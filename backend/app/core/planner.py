"""
Plan模块 v2 —— 动态任务规划

职责：
1. 根据用户需求动态拆解任务（不是模板填充）
2. 构建DAG任务图（依赖关系）
3. 静态槽位填充（实体/改写query/默认值）
4. 输出TaskGraph供ReAct执行器执行

不做：
- ask_user（假设输入已完整）
- 预设fallback策略（由执行器动态决定）

输入：rewritten_query + demands[] + resolved_entities + available_tools_schema
输出：TaskGraph
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from app.core.config import settings
from app.core.llm_client import LLMClient, TIMEOUT_STANDARD

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# 1. 数据模型
# ═══════════════════════════════════════════════════════

@dataclass
class TaskNode:
    """任务图中的单个节点"""
    task_id: str
    task_type: str  # "tool_call" | "llm_reasoning" | "aggregate"
    tool_name: Optional[str] = None  # tool_call时必填
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)  # 原始参数（含占位符）
    resolved_params: Dict[str, Any] = field(default_factory=dict)  # 解析后的实际参数
    dependencies: List[str] = field(default_factory=list)  # 依赖的task_id列表
    
    # 运行时状态
    status: str = "pending"  # pending | running | success | failed | skipped
    result: Any = None
    observation: str = ""
    is_critical: bool = False  # 是否是核心任务（失败时是否报告用户）
    started_at: float = 0.0
    finished_at: float = 0.0


@dataclass
class TaskGraph:
    """任务图（DAG）"""
    tasks: Dict[str, TaskNode] = field(default_factory=dict)
    global_status: str = "pending"  # pending | running | success | failed | needs_replan
    replan_reason: str = ""  # 触发replan的原因
    planner_thought: str = ""  # 规划思路（供调试）
    
    def get_ready_tasks(self) -> List[TaskNode]:
        """获取所有依赖已满足且状态为pending的任务"""
        ready = []
        for task in self.tasks.values():
            if task.status != "pending":
                continue
            # 检查所有依赖是否已到达终态
            deps_satisfied = all(
                self.tasks.get(dep_id, TaskNode(dep_id, "")).status in ("success", "skipped")
                for dep_id in task.dependencies
            )
            if deps_satisfied:
                ready.append(task)
        return ready
    
    def get_task(self, task_id: str) -> Optional[TaskNode]:
        return self.tasks.get(task_id)
    
    def add_task(self, task: TaskNode):
        self.tasks[task.task_id] = task
    
    def compute_parallel_groups(self) -> List[List[str]]:
        """基于依赖关系计算拓扑分层（用于调试和日志）"""
        remaining = set(self.tasks.keys())
        groups: List[List[str]] = []
        
        while remaining:
            # 找出当前没有未满足依赖的任务
            group = []
            for tid in list(remaining):
                task = self.tasks[tid]
                deps_in_remaining = set(task.dependencies) & remaining
                if not deps_in_remaining:
                    group.append(tid)
            
            if not group:
                # 有循环依赖，强制选出一个
                group = [list(remaining)[0]]
            
            groups.append(group)
            remaining -= set(group)
        
        return groups
    
    def validate(self) -> List[str]:
        """验证DAG无循环依赖"""
        errors = []
        visited = set()
        rec_stack = set()
        
        def _dfs(tid: str) -> bool:
            visited.add(tid)
            rec_stack.add(tid)
            for dep in self.tasks.get(tid, TaskNode(tid, "")).dependencies:
                if dep not in self.tasks:
                    errors.append(f"依赖不存在: {tid} -> {dep}")
                    continue
                if dep not in visited:
                    if _dfs(dep):
                        return True
                elif dep in rec_stack:
                    errors.append(f"循环依赖: {tid} -> {dep}")
                    return True
            rec_stack.remove(tid)
            return False
        
        for tid in self.tasks:
            if tid not in visited:
                _dfs(tid)
        
        return errors


# ═══════════════════════════════════════════════════════
# 2. 可用工具Schema（供Plan LLM理解）
# ═══════════════════════════════════════════════════════

AVAILABLE_TOOLS = [
    {
        "name": "kb_retrieve",
        "description": "从内部知识库检索JD信息（混合召回：70%向量+30%BM25）",
        "when_to_use": [
            "用户询问具体公司/岗位信息",
            "需要获取JD描述、岗位要求、职责等信息",
            "内部数据库可能有相关信息时"
        ],
        "when_not_to_use": [
            "用户问的是实时新闻、公司动态、行业趋势等内部库不可能有的信息",
            "已经确定内部知识库无目标公司/岗位时"
        ],
        "parameters": {
            "query": "检索关键词",
            "company": "公司名过滤（可选）",
            "position": "岗位名过滤（可选）",
            "top_k": "返回数量，默认10",
        },
    },
    {
        "name": "external_search",
        "description": "通过Brave Search搜索外部网络信息",
        "when_to_use": [
            "kb_retrieve返回结果不足（<2条或hybrid_score<0.3）",
            "用户询问包含'最新'、'最近'、'新闻'、'融资'、'裁员'、'组织架构'等实时性关键词",
            "用户提到的公司/岗位在内部知识库中未找到",
            "需要补充公司背景信息以辅助匹配分析"
        ],
        "when_not_to_use": [
            "kb_retrieve已召回高质量结果（>=3条且hybrid_score>=0.5）",
            "用户问题纯基于已有简历/JD文本的分析，无需外部信息"
        ],
        "parameters": {
            "query": "搜索关键词",
            "count": "返回数量，默认5",
        },
    },
    {
        "name": "match_analyze",
        "description": "分析简历与单个JD的匹配度，输出分数/优势/短板/建议",
        "when_to_use": [
            "需要评估用户与某个具体岗位的匹配度",
            "用户问'我匹配吗'、'够格吗'、'差距在哪'"
        ],
        "when_not_to_use": [
            "没有简历信息时",
            "没有目标JD信息时"
        ],
        "parameters": {
            "resume_text": "用户简历文本",
            "jd_text": "岗位描述文本（可从kb_retrieve或external_search结果获取）",
            "company": "公司名称（可选）",
            "position": "岗位名称（可选）",
        },
    },
    {
        "name": "global_rank",
        "description": "简历vs多JD批量对比，按匹配度排序并给出投递策略",
        "when_to_use": [
            "用户要求推荐多个岗位",
            "用户问'有哪些适合我的'、'排序一下'",
            "需要全局视角对比多个JD时"
        ],
        "when_not_to_use": [
            "用户只问某一个具体岗位（用match_analyze即可）",
            "没有多个候选JD时"
        ],
        "parameters": {
            "resume_text": "用户简历文本",
            "candidate_jds": "候选JD列表（来自kb_retrieve结果）",
            "top_k": "返回Top-K数量，默认5",
        },
    },
    {
        "name": "qa_synthesize",
        "description": "基于检索到的证据回答事实性问题",
        "when_to_use": [
            "用户询问某个岗位的具体属性（薪资/要求/福利等）",
            "需要从检索结果中综合信息回答"
        ],
        "when_not_to_use": [
            "没有检索到相关证据时",
            "用户问的是主观判断而非事实"
        ],
        "parameters": {
            "question": "用户问题",
            "evidence_chunks": "证据列表（来自kb_retrieve/external_search）",
            "qa_type": "问题类型：factual/comparative/temporal/definition",
        },
    },
    {
        "name": "interview_gen",
        "description": "基于匹配分析结果生成针对性面试题",
        "when_to_use": [
            "用户要求准备面试",
            "已经做过匹配分析，需要基于短板生成题目"
        ],
        "when_not_to_use": [
            "没有做过匹配分析时（可先做match_analyze）",
            "用户没有明确目标岗位时"
        ],
        "parameters": {
            "match_result": "match_analyze的输出结果",
            "company": "公司名称（可选）",
            "position": "岗位名称（可选）",
        },
    },
    {
        "name": "general_chat",
        "description": "通用对话/职业规划建议/行业咨询",
        "when_to_use": [
            "用户闲聊、打招呼",
            "用户问职业规划建议、行业趋势",
            "不需要检索、不需要匹配分析时"
        ],
        "parameters": {
            "user_message": "用户消息",
            "chat_type": "对话类型：career/industry/other",
        },
    },
    {
        "name": "evidence_relevance_check",
        "description": "判断上轮检索的evidence_cache是否与当前问题强相关，避免重复检索",
        "when_to_use": [
            "follow_up_type为expand或clarify时",
            "session中存在evidence_cache",
            "需要判断是否可以复用上轮检索结果"
        ],
        "when_not_to_use": [
            "follow_up_type为switch或none时",
            "evidence_cache为空时"
        ],
        "parameters": {
            "query": "当前用户query",
            "evidence_chunks": "上轮检索的evidence_cache（已截断到前N条）",
        },
    },
]


# ═══════════════════════════════════════════════════════
# 3. Plan LLM Prompt
# ═══════════════════════════════════════════════════════

PLANNER_SYSTEM_PROMPT = """你是任务规划专家。将用户需求拆解为可执行的DAG任务图。

## 核心原则
1. 动态选工具，不按模板填充
2. 同类检索合并，下游共用
3. 不生成if-else，完整序列由执行器处理
4. 最后一步必须是aggregate
5. 无简历时不生成match_analyze/global_rank

## 可用工具

{tools_desc}

## 任务类型
- tool_call: 调用外部工具（必须有tool_name）
- aggregate: 聚合上游输出，生成最终素材

## 依赖与占位符
- dependencies: 依赖的task_id列表，无依赖=[]
- {{entity.xxx}}: 意图实体
- {{resume_text}}: 简历文本
- {{Txx.output.xxx}}: 上游任务输出

## 输出格式（严格JSON）
{
  "planner_thought": "规划思路",
  "tasks": [
    {"task_id":"T0","task_type":"tool_call","tool_name":"kb_retrieve","description":"检索JD","parameters":{"query":"{{search_keywords}}"},"dependencies":[],"is_critical":false},
    {"task_id":"T1","task_type":"tool_call","tool_name":"match_analyze","description":"匹配分析","parameters":{"resume_text":"{{resume_text}}","jd_text":"{{T0.output.chunks}}"},"dependencies":["T0"],"is_critical":true},
    {"task_id":"T2","task_type":"aggregate","description":"生成回复","parameters":{},"dependencies":["T1"],"is_critical":false}
  ]
}"""


# ═══════════════════════════════════════════════════════
# 4. Plan模块核心类
# ═══════════════════════════════════════════════════════

class TaskPlanner:
    """
    动态任务规划器。
    输入：用户需求描述 + 实体 + 工具Schema
    输出：TaskGraph（DAG）
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client

    async def create_graph(
        self,
        rewritten_query: str,
        demands: List[Dict],  # [{intent_type, entities, priority}]
        resolved_entities: Dict[str, Any],
        resume_text: str = "",
        search_keywords: str = "",
        follow_up_type: str = "none",
        evidence_cache_summary: str = "",
    ) -> TaskGraph:
        """
        根据用户需求动态生成TaskGraph。
        """
        if self.llm is None:
            self.llm = LLMClient.from_config("planner")

        # 构建prompt
        tools_desc = self._build_tools_description()
        system_prompt = PLANNER_SYSTEM_PROMPT.replace("{tools_desc}", tools_desc)

        user_prompt = self._build_user_prompt(
            rewritten_query=rewritten_query,
            demands=demands,
            entities=resolved_entities,
            resume_text=resume_text,
            follow_up_type=follow_up_type,
            evidence_cache_summary=evidence_cache_summary,
        )

        try:
            raw = await self.llm.generate(
                prompt=user_prompt,
                system=system_prompt,
                temperature=0.2,
                max_tokens=2000,
                timeout=TIMEOUT_STANDARD,  # 20s，Planner 生成 DAG
            )
        except Exception as e:
            logger.error(f"[Planner] LLM规划失败: {e}")
            return self._fallback_graph(demands, resolved_entities, resume_text, rewritten_query, follow_up_type)

        # 解析
        graph = self._parse_graph(raw)
        if not graph.tasks:
            logger.warning("[Planner] LLM输出解析为空，fallback到规则图")
            return self._fallback_graph(demands, resolved_entities, resume_text, rewritten_query, follow_up_type)

        # 验证
        errors = graph.validate()
        if errors:
            logger.warning(f"[Planner] 图验证警告: {errors}")

        # 静态槽位填充
        graph = self._fill_static_slots(graph, resolved_entities, resume_text, rewritten_query, search_keywords)

        logger.info(
            f"[Planner] 规划完成 | tasks={len(graph.tasks)} | "
            f"groups={len(graph.compute_parallel_groups())} | thought={graph.planner_thought[:60]}..."
        )
        return graph

    def _build_tools_description(self) -> str:
        """构建精简工具描述文本"""
        lines = []
        for tool in AVAILABLE_TOOLS:
            lines.append(f"- {tool['name']}: {tool['description']}")
            lines.append(f"  参数：{json.dumps(tool.get('parameters', {}), ensure_ascii=False)}")
        return "\n".join(lines)

    def _build_user_prompt(
        self,
        rewritten_query: str,
        demands: List[Dict],
        entities: Dict[str, Any],
        resume_text: str,
        follow_up_type: str = "none",
        evidence_cache_summary: str = "",
    ) -> str:
        """构建用户prompt"""
        demand_lines = []
        for i, d in enumerate(demands):
            demand_lines.append(f"需求{i+1}：{d.get('intent_type', 'unknown')} | 实体：{json.dumps(d.get('entities', {}), ensure_ascii=False)}")

        lines = [
            "【用户需求】",
            f"改写后query：{rewritten_query}",
            f"提取实体：{json.dumps(entities, ensure_ascii=False)}",
            f"是否有简历：{'是' if resume_text else '否'}",
            f"追问类型：{follow_up_type}",
            "",
        ]
        if evidence_cache_summary:
            lines.extend([
                "【上轮检索证据摘要】",
                evidence_cache_summary,
                "",
                "检索策略提示：",
                "- expand/clarify → 优先判断evidence_cache是否可直接复用",
                "- switch/none → 必须重新检索",
                "",
            ])
        lines.extend([
            "【需求列表】",
        ])
        lines.extend(demand_lines)
        lines.extend([
            "",
            "请生成TaskGraph JSON：",
        ])
        return "\n".join(lines)

    def _parse_graph(self, raw: str) -> TaskGraph:
        """解析LLM输出为TaskGraph"""
        graph = TaskGraph()

        if not raw or not raw.strip():
            return graph

        text = raw.strip()
        # 去除markdown代码块
        for marker in ["```json", "```"]:
            if marker in text:
                text = text.replace(marker, "")
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("[Planner] JSON解析失败，尝试提取")
            # 尝试提取JSON片段
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(text[start:end+1])
                except json.JSONDecodeError:
                    return graph
            else:
                return graph

        graph.planner_thought = data.get("planner_thought", "")

        for t in data.get("tasks", []):
            task = TaskNode(
                task_id=t.get("task_id", f"T{len(graph.tasks)}"),
                task_type=t.get("task_type", "tool_call"),
                tool_name=t.get("tool_name") if t.get("tool_name") else None,
                description=t.get("description", ""),
                parameters=t.get("parameters", {}),
                dependencies=t.get("dependencies", []),
                is_critical=t.get("is_critical", False),
            )
            graph.add_task(task)

        return graph

    def _fill_static_slots(
        self,
        graph: TaskGraph,
        entities: Dict[str, Any],
        resume_text: str,
        rewritten_query: str,
        search_keywords: str = "",
    ) -> TaskGraph:
        """
        静态槽位填充：使用意图识别提取的实体填充参数中的占位符。
        不涉及跨任务引用（那是执行器的动态填充）。
        支持递归处理嵌套字典和列表。
        """
        def _resolve_value(val):
            if isinstance(val, str) and val.startswith("{{") and val.endswith("}}"):
                placeholder = val[2:-2].strip()
                return self._resolve_static_placeholder(
                    placeholder, entities, resume_text, rewritten_query, search_keywords
                )
            elif isinstance(val, dict):
                return {k: _resolve_value(v) for k, v in val.items()}
            elif isinstance(val, list):
                return [_resolve_value(item) for item in val]
            else:
                return val

        for task in graph.tasks.values():
            task.resolved_params = _resolve_value(task.parameters)

        return graph

    def _resolve_static_placeholder(
        self,
        placeholder: str,
        entities: Dict[str, Any],
        resume_text: str,
        rewritten_query: str,
        search_keywords: str = "",
    ) -> Any:
        """解析静态占位符（不涉及跨任务引用，Txx.output 等动态引用保持原样）"""
        # 跨任务动态引用（如 T0.output.chunks）由执行器在运行时解析，此处保持原样
        if re.match(r"^T\d+", placeholder):
            return f"{{{{{placeholder}}}}}"

        # entity.xxx
        if placeholder.startswith("entity."):
            field = placeholder.replace("entity.", "")
            return entities.get(field, "")

        # 直接匹配
        if placeholder == "resume_text":
            return resume_text
        if placeholder == "rewritten_query":
            return rewritten_query
        if placeholder == "search_keywords":
            return search_keywords if search_keywords else rewritten_query

        # 默认：尝试从entities查找
        if placeholder in entities:
            return entities[placeholder]

        logger.warning(f"[Planner] 未解析的静态占位符: {placeholder}")
        return ""

    def _fallback_graph(
        self,
        demands: List[Dict],
        entities: Dict[str, Any],
        resume_text: str,
        rewritten_query: str = "",
        follow_up_type: str = "none",
    ) -> TaskGraph:
        """
        LLM规划失败时的兜底图。
        基于demands生成最简化的任务链。
        """
        graph = TaskGraph()
        graph.planner_thought = "LLM规划失败，使用fallback规则图"

        if not demands:
            # 纯对话
            graph.add_task(TaskNode(
                task_id="T0",
                task_type="aggregate",
                description="通用对话回复",
                parameters={},
                dependencies=[],
            ))
            return graph

        # 获取主需求
        main_demand = demands[0]
        intent_type = main_demand.get("intent_type", "general_chat")
        has_resume = bool(resume_text)

        # 基于需求类型生成最简链
        # expand/clarify 且存在实体时，生成简化图（跳过kb_retrieve，优先复用evidence_cache）
        skip_retrieve = follow_up_type in ("expand", "clarify") and bool(entities)

        if intent_type == "general_chat":
            graph.add_task(TaskNode(
                task_id="T0", task_type="aggregate",
                description="通用对话回复", parameters={}, dependencies=[],
            ))

        elif intent_type in ("match_assess", "attribute_verify"):
            if skip_retrieve:
                # 复用evidence_cache，直接做分析/问答
                if intent_type == "match_assess" and has_resume:
                    graph.add_task(TaskNode(
                        task_id="T0", task_type="tool_call", tool_name="match_analyze",
                        description="分析匹配度（复用缓存证据）",
                        parameters={"resume_text": resume_text, "jd_text": "{{evidence_cache}}"},
                        dependencies=[], is_critical=True,
                    ))
                    graph.add_task(TaskNode(
                        task_id="T1", task_type="aggregate",
                        description="汇总结果", parameters={"result": "{{T0.output}}"},
                        dependencies=["T0"], is_critical=False,
                    ))
                else:
                    graph.add_task(TaskNode(
                        task_id="T0", task_type="tool_call", tool_name="qa_synthesize",
                        description="回答用户问题（复用缓存证据）",
                        parameters={"question": rewritten_query, "evidence_chunks": "{{evidence_cache}}"},
                        dependencies=[], is_critical=False,
                    ))
                    graph.add_task(TaskNode(
                        task_id="T1", task_type="aggregate",
                        description="汇总结果", parameters={"result": "{{T0.output}}"},
                        dependencies=["T0"], is_critical=False,
                    ))
            else:
                # 检索 -> 分析/问答
                graph.add_task(TaskNode(
                    task_id="T0", task_type="tool_call", tool_name="kb_retrieve",
                    description="检索目标岗位信息",
                    parameters={"query": entities.get("company", "") + " " + entities.get("position", "")},
                    dependencies=[], is_critical=False,
                ))
                if intent_type == "match_assess" and has_resume:
                    graph.add_task(TaskNode(
                        task_id="T1", task_type="tool_call", tool_name="match_analyze",
                        description="分析匹配度",
                        parameters={"resume_text": resume_text, "jd_text": "{{T0.output.chunks}}"},
                        dependencies=["T0"], is_critical=True,
                    ))
                    graph.add_task(TaskNode(
                        task_id="T2", task_type="aggregate",
                        description="汇总结果", parameters={"result": "{{T1.output}}"},
                        dependencies=["T1"], is_critical=False,
                    ))
                else:
                    graph.add_task(TaskNode(
                        task_id="T1", task_type="tool_call", tool_name="qa_synthesize",
                        description="回答用户问题",
                        parameters={"question": rewritten_query, "evidence_chunks": "{{T0.output.chunks}}"},
                        dependencies=["T0"], is_critical=False,
                    ))
                    graph.add_task(TaskNode(
                        task_id="T2", task_type="aggregate",
                        description="汇总结果", parameters={"result": "{{T1.output}}"},
                        dependencies=["T1"], is_critical=False,
                    ))

        elif intent_type == "position_explore":
            if skip_retrieve:
                if has_resume:
                    graph.add_task(TaskNode(
                        task_id="T0", task_type="tool_call", tool_name="global_rank",
                        description="全局匹配排序（复用缓存证据）",
                        parameters={"resume_text": resume_text, "candidate_jds": "{{evidence_cache}}"},
                        dependencies=[], is_critical=True,
                    ))
                    graph.add_task(TaskNode(
                        task_id="T1", task_type="aggregate",
                        description="汇总结果", parameters={"result": "{{T0.output}}"},
                        dependencies=["T0"], is_critical=False,
                    ))
                else:
                    graph.add_task(TaskNode(
                        task_id="T0", task_type="aggregate",
                        description="汇总结果（复用缓存证据）", parameters={"result": "{{evidence_cache}}"},
                        dependencies=[], is_critical=False,
                    ))
            else:
                graph.add_task(TaskNode(
                    task_id="T0", task_type="tool_call", tool_name="kb_retrieve",
                    description="检索相关岗位",
                    parameters={"query": resume_text[:100] if resume_text else "推荐岗位"},
                    dependencies=[], is_critical=False,
                ))
                if has_resume:
                    graph.add_task(TaskNode(
                        task_id="T1", task_type="tool_call", tool_name="global_rank",
                        description="全局匹配排序",
                        parameters={"resume_text": resume_text, "candidate_jds": "{{T0.output.chunks}}"},
                        dependencies=["T0"], is_critical=True,
                    ))
                    graph.add_task(TaskNode(
                        task_id="T2", task_type="aggregate",
                        description="汇总结果", parameters={"result": "{{T1.output}}"},
                        dependencies=["T1"], is_critical=False,
                    ))
                else:
                    graph.add_task(TaskNode(
                        task_id="T2", task_type="aggregate",
                        description="汇总结果", parameters={"result": "{{T0.output}}"},
                        dependencies=["T0"], is_critical=False,
                    ))

        elif intent_type == "interview_prepare":
            if skip_retrieve:
                if has_resume:
                    graph.add_task(TaskNode(
                        task_id="T0", task_type="tool_call", tool_name="match_analyze",
                        description="分析匹配度（复用缓存证据）",
                        parameters={"resume_text": resume_text, "jd_text": "{{evidence_cache}}"},
                        dependencies=[], is_critical=False,
                    ))
                    graph.add_task(TaskNode(
                        task_id="T1", task_type="tool_call", tool_name="interview_gen",
                        description="生成面试题",
                        parameters={"match_result": "{{T0.output}}"},
                        dependencies=["T0"], is_critical=False,
                    ))
                    graph.add_task(TaskNode(
                        task_id="T2", task_type="aggregate",
                        description="汇总结果", parameters={"result": "{{T1.output}}"},
                        dependencies=["T1"], is_critical=False,
                    ))
                else:
                    graph.add_task(TaskNode(
                        task_id="T0", task_type="aggregate",
                        description="汇总结果（复用缓存证据）", parameters={"result": "{{evidence_cache}}"},
                        dependencies=[], is_critical=False,
                    ))
            else:
                graph.add_task(TaskNode(
                    task_id="T0", task_type="tool_call", tool_name="kb_retrieve",
                    description="检索目标岗位JD",
                    parameters={"query": entities.get("position", "")},
                    dependencies=[], is_critical=False,
                ))
                if has_resume:
                    graph.add_task(TaskNode(
                        task_id="T1", task_type="tool_call", tool_name="match_analyze",
                        description="分析匹配度",
                        parameters={"resume_text": resume_text, "jd_text": "{{T0.output.chunks}}"},
                        dependencies=["T0"], is_critical=False,
                    ))
                    graph.add_task(TaskNode(
                        task_id="T2", task_type="tool_call", tool_name="interview_gen",
                        description="生成面试题",
                        parameters={"match_result": "{{T1.output}}"},
                        dependencies=["T1"], is_critical=False,
                    ))
                    graph.add_task(TaskNode(
                        task_id="T3", task_type="aggregate",
                        description="汇总结果", parameters={"result": "{{T2.output}}"},
                        dependencies=["T2"], is_critical=False,
                    ))
                else:
                    graph.add_task(TaskNode(
                        task_id="T2", task_type="aggregate",
                        description="汇总结果", parameters={"result": "{{T0.output}}"},
                        dependencies=["T0"], is_critical=False,
                    ))

        else:
            # 兜底：通用aggregate
            graph.add_task(TaskNode(
                task_id="T0", task_type="aggregate",
                description="通用回复", parameters={},
                dependencies=[],
            ))

        # 执行静态槽位填充
        graph = self._fill_static_slots(graph, entities, resume_text, rewritten_query, search_keywords)
        return graph
