"""
LLM Planner v2 —— 基于 TaskGraph 的任务规划器

核心职能：
1. 任务拆分器（Decomposer）：将多意图拆分为原子任务
2. 任务依赖设定器（Dependency Designer）：精确定义数据依赖、控制依赖、共享依赖
3. 状态跟踪器（State Tracker）：跟踪全局就绪状态和历史缓存复用
4. 失败处理器（Failure Handler）：为每个任务设计 fallback 策略及级联影响

与 v1 的区别：
- 输出从 TaskPlan 升级为 TaskGraph（含并行组、关键路径、fallback 级联）
- 支持多意图公共前置合并（相同参数的 kb_retrieve 合并）
- 支持历史缓存复用判断（Replan 场景）
- 支持 ask_user / abort / skip / retry 四级失败处理
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from app.core.llm_client import LLMClient
from app.core.memory import SessionMemory
from app.core.tool_registry import TOOL_REGISTRY_META, ToolRegistry
from app.core.config import settings
from app.core.task_graph import (
    TaskGraph, TaskNode, TaskFallback, ExecutionStrategy, HistoryCacheEntry,
)
from app.core.llm_intent import MultiIntentResult, LLMIntentType, IntentCandidate

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# 1. System Prompt 模板（核心框架）
# ═══════════════════════════════════════════════════════

PLANNER_SYSTEM_PROMPT = """你是「求职雷达」系统的执行编排专家。你的职责是根据用户意图和系统状态，将高层意图拆解为可执行的原子任务图（TaskGraph），并精确定义任务间的依赖关系、失败处理策略和执行策略。

## 你的四个核心职能

### 1. 任务拆分器（Decomposer）
- 一个意图可能需要拆成多个原子任务（如 ASSESS = 检索JD + 匹配分析）
- 多个意图可能合并公共前置任务（如 ASSESS 和 VERIFY 都需 JD，若检索参数相同则复用一次 kb_retrieve）
- 复杂操作需拆为"准备→执行→后处理"（如上传简历 = file_ops + 解析提取 + 更新状态）

### 2. 任务依赖设定器（Dependency Designer）
- 数据依赖：任务 B 需要任务 A 的输出字段（如 match_analyze 需要 kb_retrieve 的 chunks）
- 控制依赖：即使无数据依赖，某些任务必须串行（如 file_ops upload 必须在 parse 前）
- 共享依赖：多个下游依赖同一个上游时，上游只执行一次，输出被多处引用
- 严禁循环依赖；无依赖的任务必须 dependencies=[] 以触发并行

### 3. 状态跟踪器（State Tracker）
- 规划时必须考虑：当前全局槽位中哪些数据已就绪（如 resume_text 已存在 vs 缺失）
- 若某必要数据缺失（如用户未上传简历），不要硬编码空值，应设计 ask_user 任务中断执行
- 追问场景必须识别：上轮哪些任务已执行、输出字段是否仍有效、哪些状态已过期

### 4. 失败处理器（Failure Handler）
每个任务必须设计 fallback，且要考虑级联影响：
- retry：网络/临时错误，指数退避重试
- skip：非核心任务失败，下游使用默认值或空值继续
- ask_user：关键输入缺失，中断执行并追问
- abort：致命错误，终止整个计划

---

## 输入数据结构

### 1. 意图拓扑（由上游提供，你只消费）
[意图拓扑] 示例：
Layer 0: assess(0.95) | verify(0.92)      ← 同层并行
Layer 1: prepare(0.88)                    ← 依赖上层

### 2. 全局就绪状态
[全局就绪状态] 示例：
resume_text: ✅ 已就绪（来自附件）
company: ✅ 字节跳动
position: ✅ 算法岗

### 3. 意图专属槽位
[意图专属槽位] 示例：
assess: query="字节跳动 算法岗", attributes=["匹配度","差距"]
verify: query="字节跳动 算法岗 薪资", attributes=["薪资"], qa_type=factual
prepare: count=5, difficulty=tricky

### 4. 工具注册表
"""

PLANNER_TOOL_REGISTRY_DESC = """
kb_retrieve(query, top_k, company?, position?) → chunks[], meta, total
  成本: medium | 幂等: 是 | 可缓存: 是
match_analyze(resume_text, jd_source, attributes?, company?, position?) → score, gaps[], suggestions[]
  成本: high | 幂等: 是 | 依赖: kb_retrieve.chunks 或附件文本
qa_synthesize(question, evidence_chunks, qa_type, attributes?, company?, position?) → answer, citations[], confidence
  成本: medium | 幂等: 是 | 依赖: evidence（kb_retrieve.chunks 或附件）
interview_gen(match_result, count?, difficulty?, focus_area?, company?, position?) → questions[], rationale
  成本: medium | 幂等: 否 | 依赖: match_result（来自 match_analyze）
global_rank(resume_text, candidate_jds, sort_by?) → ranked_list[], explanation
  成本: high | 幂等: 是
file_ops(operation, file_data?, text_data?, target_id?) → status, file_id, extracted_text
  成本: low | 幂等: 否（写操作）| 副作用: 修改存储状态
parse_resume(raw_text) → resume_text, structured_info
  成本: low | 幂等: 是 | 依赖: file_ops 的 extracted_text
general_chat(user_message, chat_type?, user_profile?) → response
  成本: low
"""

PLANNER_RULES = """
---

## 编排决策规则（你必须遵守）

规则1：公共前置合并
若多个意图都需要同类数据（如 JD 文本），检查它们的检索参数是否完全相同（query + top_k + company + position）：
- 相同：只生成一个 kb_retrieve，下游共用 {{Txx.output.chunks}}
- 不同：分别生成多个 kb_retrieve

规则2：数据流显式化
每个任务的 parameters 必须声明数据来源：
- 来自全局槽位：{{global_slots.xxx}}
- 来自上游任务：{{Txx.output.chunks}}
- 来自意图专属槽位：{{intent_xxx.yyy}}
- 来自历史缓存：{{last.Txx.output.chunks}}
禁止在 parameters 中写死未经验证的字符串，除非该值确实不在任何槽位中。

规则3：缺失数据阻断
若某任务必要参数缺失且无法通过占位符解析：
- 方案A：在该任务前插入 ask_user 任务
- 方案B：将该任务 fallback 设为 ask_user
- 方案C：若缺失的是非关键参数，使用 fallback.default_params 降级执行

规则4：级联失败设计
设计 fallback 时必须写明 downstream_impact：
- T0(kb_retrieve) 失败 → abort：下游 match_analyze / qa_synthesize 全部无数据
- T1(match_analyze) 失败 → skip：interview_gen 的 focus_area 降级为通用值

规则5：追问状态感知（Replan）
收到历史缓存时，你必须判断：
- 缓存命中：上轮任务参数 hash 与本轮完全相同 → 复用 {{last.Txx}}，不生成新任务
- 缓存失效：参数不同 → 视为不同，重新生成任务
- 缓存过期：用户明确说"更新/修改/重新上传" → 缓存失效，重新执行

规则6：执行策略标注
在 execution_strategy 中显式声明：
- parallel_groups：拓扑分层，同层可并行
- critical_path：决定总耗时的最长依赖链
- estimated_cost：基于工具成本累加（low/medium/high）
"""

PLANNER_OUTPUT_SCHEMA = """
---

## 输出格式（严格 JSON，不要 markdown 代码块）

{
  "planner_thought": "规划思路：1. 哪些意图可并行 2. 公共前置是否合并 3. 缺失数据如何处理 4. 失败级联设计",
  "tasks": [
    {
      "task_id": "T0",
      "task_type": "tool_call",
      "tool_name": "kb_retrieve",
      "description": "检索JD：字节跳动算法岗（用于assess）",
      "parameters": {
        "query": "{{intent_assess.query}}",
        "top_k": 3
      },
      "dependencies": [],
      "execution_mode": "auto",
      "fallback": {
        "action": "ask_user",
        "reason": "未找到JD，请上传JD图片"
      },
      "output_schema": {
        "chunks": "List[Dict]",
        "meta": "Dict",
        "total": "int"
      }
    },
    {
      "task_id": "T1",
      "task_type": "tool_call",
      "tool_name": "match_analyze",
      "description": "分析简历与JD匹配度",
      "parameters": {
        "resume_text": "{{global_slots.resume_text}}",
        "jd_source": "kb",
        "jd_data": {"chunks": "{{T0.output.chunks}}"},
        "attributes": "{{intent_assess.attributes}}"
      },
      "dependencies": ["T0"],
      "execution_mode": "auto",
      "fallback": {
        "action": "skip",
        "reason": "匹配分析失败，下游 interview_gen 将使用通用 focus_area",
        "downstream_impact": "T2.parameters.focus_area 降级为 ['通用技术面试']",
        "default_params": {
          "focus_area": ["通用技术面试"]
        }
      },
      "output_schema": {
        "score": "float",
        "gaps": "List[str]",
        "suggestions": "List[str]"
      }
    },
    {
      "task_id": "T2",
      "task_type": "tool_call",
      "tool_name": "interview_gen",
      "description": "基于匹配短板生成面试题",
      "parameters": {
        "match_result": "{{T1.output}}",
        "count": "{{intent_prepare.count}}",
        "difficulty": "{{intent_prepare.difficulty}}"
      },
      "dependencies": ["T1"],
      "execution_mode": "auto",
      "fallback": {
        "action": "skip",
        "reason": "使用通用题库生成面试题",
        "default_params": {
          "focus_area": ["项目经验", "系统设计", "算法基础"]
        }
      },
      "output_schema": {
        "questions": "List[Dict]",
        "rationale": "str"
      }
    },
    {
      "task_id": "T3",
      "task_type": "aggregate",
      "description": "聚合所有成功任务的输出，生成回复素材",
      "parameters": {
        "assess": "{{T1.output}}",
        "prepare": "{{T2.output}}"
      },
      "dependencies": ["T1", "T2"],
      "execution_mode": "auto",
      "fallback": {
        "action": "abort",
        "reason": "聚合失败，无法生成回复"
      },
      "output_schema": {
        "synthesis_material": "Dict"
      }
    }
  ],
  "execution_strategy": {
    "parallel_groups": [["T0"], ["T1"], ["T2"], ["T3"]],
    "critical_path": ["T0", "T1", "T2", "T3"],
    "estimated_cost": "high"
  }
}

---

## 任务类型说明

- tool_call：调用外部工具（必须有 tool_name）
- llm_reasoning：纯 LLM 推理步骤，不调用外部工具（如解析、摘要、格式转换）
- aggregate：聚合上游任务输出，生成最终素材（不调用外部工具）
- ask_user：中断执行，向用户提问（parameters 中可包含 question 和 options）

## fallback.action 说明

- retry：临时错误，指数退避重试（max_retries 默认 3）
- skip：非核心任务失败，下游使用 default_params 继续
- ask_user：关键输入缺失，中断执行并追问
- abort：致命错误，终止整个计划（级联中止所有下游）

## 输出前自检清单

生成 JSON 前确认：
1. [ ] 任务拆分合理，公共前置已合并
2. [ ] 依赖正确，无循环依赖，所有 {{Txx}} 引用的 task_id 存在
3. [ ] 数据流清晰，参数来自槽位或上游输出，无硬编码臆测
4. [ ] 必要数据缺失时设计了 ask_user 或 abort
5. [ ] 每个 fallback 都考虑了 downstream_impact
6. [ ] 存在历史缓存时检查了参数匹配
7. [ ] 最后一个任务应为 aggregate
8. [ ] execution_strategy.parallel_groups 覆盖所有任务
"""


# ═══════════════════════════════════════════════════════
# 2. Planner 核心类
# ═══════════════════════════════════════════════════════

class TaskGraphPlanner:
    """
    TaskGraph 规划器。

    输入：多意图结果 + 全局槽位 + 工具注册表 + 历史缓存
    输出：TaskGraph（含任务拆分、依赖、fallback、执行策略）
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client

    # ──────────────────────────── 公共 API ────────────────────────────

    async def create_graph(
        self,
        multi_result: MultiIntentResult,
        session: SessionMemory,
        resume_text: str,
        rewrite_result: Any,
        tool_registry: Optional[ToolRegistry] = None,
        history_cache: Optional[List[HistoryCacheEntry]] = None,
    ) -> TaskGraph:
        """
        根据多意图结果生成 TaskGraph。
        """
        if self.llm is None:
            self.llm = LLMClient.from_config("planner")

        # 构建 prompt
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            multi_result=multi_result,
            session=session,
            resume_text=resume_text,
            rewrite_result=rewrite_result,
            history_cache=history_cache or [],
        )

        try:
            raw = await self.llm.generate(
                prompt=user_prompt,
                system=system_prompt,
                temperature=0.2,
                max_tokens=4000,
            )
        except Exception as e:
            logger.warning(f"[TaskGraphPlanner] LLM 规划失败: {e}，fallback 到规则")
            return self._fallback_graph(multi_result, resume_text, history_cache or [])

        # 解析
        graph = self._parse_graph(raw, multi_result, history_cache or [])

        # 后处理
        graph = self._post_process(graph, history_cache or [])

        # 验证
        errors = graph.validate()
        if errors:
            logger.warning(f"[TaskGraphPlanner] 验证警告: {errors}")

        logger.info(
            f"[TaskGraphPlanner] 规划完成 | tasks={len(graph.tasks)} | "
            f"parallel_groups={len(graph.execution_strategy.parallel_groups)} | "
            f"cost={graph.execution_strategy.estimated_cost}"
        )
        return graph

    async def replan(
        self,
        graph: TaskGraph,
        failed_task: TaskNode,
        session: SessionMemory,
    ) -> TaskGraph:
        """
        动态 replan：根据失败任务调整计划。
        """
        if failed_task.fallback:
            fb = failed_task.fallback

            if fb.action == "retry" and failed_task.retry_count < fb.max_retries:
                failed_task.retry_count += 1
                failed_task.status = "pending"
                # 指数退避
                backoff = fb.backoff_base_ms * (2 ** (failed_task.retry_count - 1))
                logger.info(
                    f"[TaskGraphPlanner] Replan: 重试 {failed_task.task_id} "
                    f"(第{failed_task.retry_count}次, 退避{backoff}ms)"
                )
                return graph

            if fb.action == "skip":
                failed_task.status = "skipped"
                self._apply_skip_impact(graph, failed_task)
                logger.info(f"[TaskGraphPlanner] Replan: 跳过 {failed_task.task_id}")
                return graph

            if fb.action == "ask_user":
                graph.global_status = "needs_clarification"
                graph.clarification_question = fb.reason
                # 收集澄清选项（从 ask_user 任务或下游缺失推断）
                opts = fb.default_params.get("options", [])
                graph.clarification_options = opts if isinstance(opts, list) else []
                logger.info(f"[TaskGraphPlanner] Replan: 触发澄清 - {fb.reason}")
                return graph

            if fb.action == "abort":
                self._cascade_abort(graph, failed_task)
                graph.global_status = "failed"
                logger.info(f"[TaskGraphPlanner] Replan: 中止 {failed_task.task_id} 及下游")
                return graph

        # 默认 abort
        self._cascade_abort(graph, failed_task)
        graph.global_status = "failed"
        return graph

    # ──────────────────────────── Prompt 构建 ────────────────────────────

    def _build_system_prompt(self) -> str:
        return (
            PLANNER_SYSTEM_PROMPT
            + PLANNER_TOOL_REGISTRY_DESC
            + PLANNER_RULES
            + PLANNER_OUTPUT_SCHEMA
        )

    def _build_user_prompt(
        self,
        multi_result: MultiIntentResult,
        session: SessionMemory,
        resume_text: str,
        rewrite_result: Any,
        history_cache: List[HistoryCacheEntry],
    ) -> str:
        """构建用户 prompt（场景 A：全新规划）"""
        parts: List[str] = []

        # 1. 意图拓扑
        parts.append("## 意图拓扑")
        for layer_idx, layer in enumerate(multi_result.execution_topology):
            items = []
            for intent in layer:
                cand = next((c for c in multi_result.candidates if c.intent_type == intent), None)
                if cand:
                    items.append(f"{intent.value}({cand.confidence:.2f})")
                else:
                    items.append(f"{intent.value}(?)")
            parts.append(f"Layer {layer_idx}: {' | '.join(items)}")
        if not multi_result.execution_topology:
            primary = multi_result.primary_intent
            primary_cand = next((c for c in multi_result.candidates if c.intent_type == primary), None)
            conf = primary_cand.confidence if primary_cand else 0.0
            parts.append(f"Layer 0: {primary.value if primary else 'chat'}({conf:.2f})")

        # 2. 全局就绪状态
        parts.append("\n## 全局就绪状态")
        has_resume = bool(resume_text) and "尚未上传" not in resume_text
        parts.append(f"resume_text: {'✅ 已就绪' if has_resume else '❌ 缺失'}")
        gs = multi_result.global_slots or {}
        for k, v in gs.items():
            if k == "resume_text":
                continue
            status = "✅" if v is not None else "❌"
            parts.append(f"{k}: {status} {v}")
        if not gs:
            parts.append("（无其他全局槽位）")

        # 3. 意图专属槽位
        parts.append("\n## 意图专属槽位")
        for cand in multi_result.candidates:
            slots = cand.slots or {}
            slot_lines = [f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in slots.items()]
            parts.append(f"{cand.intent_type.value}: " + ", ".join(slot_lines) if slot_lines else f"{cand.intent_type.value}: （无专属槽位）")

        # 4. 改写后查询
        parts.append(f"\n## 改写后查询")
        parts.append(f"rewritten_query: {getattr(rewrite_result, 'rewritten_query', '')}")
        parts.append(f"search_keywords: {getattr(rewrite_result, 'search_keywords', '')}")
        parts.append(f"is_follow_up: {getattr(rewrite_result, 'is_follow_up', False)}")
        parts.append(f"follow_up_type: {getattr(rewrite_result, 'follow_up_type', 'none')}")

        # 5. 历史缓存（Replan 场景）
        if history_cache:
            parts.append("\n## 历史执行缓存")
            for h in history_cache:
                status_icon = "✅" if h.status == "success" else "❌"
                parts.append(f"{status_icon} {h.task_id}({h.tool_name}): params_hash={h.params_hash}, status={h.status}")
            parts.append("\n### 缓存复用规则")
            parts.append("- 若本轮某任务参数 hash 与历史缓存相同 → 复用 {{last.Txx}}，不生成新任务")
            parts.append("- 若参数不同 → 新建任务")
            parts.append("- 若历史任务失败 → 不可复用，必须重新执行")

        parts.append("\n请生成 TaskGraph JSON。")
        return "\n".join(parts)

    # ──────────────────────────── 解析 ────────────────────────────

    def _parse_graph(
        self,
        raw: str,
        multi_result: MultiIntentResult,
        history_cache: List[HistoryCacheEntry],
    ) -> TaskGraph:
        """解析 LLM 输出为 TaskGraph"""
        text = raw.strip() if raw else ""
        if not text:
            return self._fallback_graph(multi_result, "", history_cache)

        # 去除 markdown 代码块
        for marker in ["```json", "```"]:
            if marker in text:
                text = re.sub(rf"^{re.escape(marker)}\s*|\s*{re.escape(marker)}$", "", text, flags=re.MULTILINE).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    logger.warning("[TaskGraphPlanner] JSON 解析失败，fallback")
                    return self._fallback_graph(multi_result, "", history_cache)
            else:
                return self._fallback_graph(multi_result, "", history_cache)

        planner_thought = data.get("planner_thought", "")
        tasks_raw = data.get("tasks", [])
        strategy_raw = data.get("execution_strategy", {})

        tasks: List[TaskNode] = []
        for tr in tasks_raw:
            if not isinstance(tr, dict):
                continue

            fb_raw = tr.get("fallback")
            fallback = None
            if fb_raw:
                fallback = TaskFallback(
                    action=fb_raw.get("action", "abort"),
                    reason=fb_raw.get("reason", ""),
                    downstream_impact=fb_raw.get("downstream_impact", ""),
                    default_params=fb_raw.get("default_params", {}),
                    max_retries=fb_raw.get("max_retries", 3),
                )

            node = TaskNode(
                task_id=str(tr.get("task_id", f"T{len(tasks)}")),
                task_type=tr.get("task_type", "tool_call"),
                tool_name=tr.get("tool_name"),
                description=tr.get("description", ""),
                parameters=tr.get("parameters", {}),
                dependencies=tr.get("dependencies", []),
                execution_mode=tr.get("execution_mode", "auto"),
                fallback=fallback,
                output_schema=tr.get("output_schema", {}),
            )
            node.params_hash = node.compute_params_hash()
            tasks.append(node)

        strategy = ExecutionStrategy(
            parallel_groups=strategy_raw.get("parallel_groups", []),
            critical_path=strategy_raw.get("critical_path", []),
            estimated_cost=strategy_raw.get("estimated_cost", "medium"),
        )

        return TaskGraph(
            planner_thought=planner_thought,
            tasks=tasks,
            execution_strategy=strategy,
            context={
                "primary_intent": multi_result.primary_intent.value if multi_result.primary_intent else "chat",
                "candidate_intents": [c.intent_type.value for c in multi_result.candidates],
                "global_slots": multi_result.global_slots,
            },
            history_cache=history_cache,
        )

    # ──────────────────────────── 后处理 ────────────────────────────

    def _post_process(
        self,
        graph: TaskGraph,
        history_cache: List[HistoryCacheEntry],
    ) -> TaskGraph:
        """
        后处理：
        1. 公共前置合并（相同 tool_name + params_hash 的 tool_call 合并）
        2. 历史缓存复用标注
        3. 自动计算 parallel_groups（若 LLM 未提供或不全）
        4. 自动计算 critical_path
        """
        # 1. 历史缓存复用
        for t in graph.tasks:
            if t.task_type != "tool_call":
                continue
            for h in history_cache:
                if h.tool_name == t.tool_name and h.is_equivalent(t.params_hash):
                    t.reused_from_history = True
                    t.history_task_id = h.task_id
                    t.status = "success"
                    t.result = h.output
                    t.observation = f"复用历史缓存 {h.task_id}"
                    logger.info(f"[TaskGraphPlanner] 复用历史缓存: {t.task_id} <- {h.task_id}")
                    break

        # 2. 公共前置合并
        graph = self._merge_common_prefixes(graph)

        # 3. 自动计算 parallel_groups（若缺失或不完整）
        declared_flat = set()
        for g in graph.execution_strategy.parallel_groups:
            declared_flat.update(g)
        task_ids = set(t.task_id for t in graph.tasks)
        if not graph.execution_strategy.parallel_groups or declared_flat != task_ids:
            graph.execution_strategy.parallel_groups = graph.compute_parallel_groups()
            logger.info(f"[TaskGraphPlanner] 自动计算 parallel_groups: {graph.execution_strategy.parallel_groups}")

        # 4. 自动计算 critical_path（若缺失）
        if not graph.execution_strategy.critical_path:
            graph.execution_strategy.critical_path = graph.compute_critical_path()

        return graph

    def _merge_common_prefixes(self, graph: TaskGraph) -> TaskGraph:
        """
        合并公共前置任务：
        若多个 tool_call 任务使用相同 tool_name 和 params_hash，只保留第一个，
        其余任务的依赖重定向到保留的任务。
        """
        seen: Dict[str, TaskNode] = {}  # key=(tool_name, params_hash) -> TaskNode
        merge_map: Dict[str, str] = {}  # old_task_id -> kept_task_id

        for t in graph.tasks:
            if t.task_type != "tool_call" or not t.tool_name:
                continue
            key = f"{t.tool_name}#{t.params_hash}"
            if key in seen:
                # 合并：当前任务被已有任务替代
                kept = seen[key]
                merge_map[t.task_id] = kept.task_id
                logger.info(f"[TaskGraphPlanner] 合并公共前置: {t.task_id} -> {kept.task_id} ({t.tool_name})")
            else:
                seen[key] = t

        if not merge_map:
            return graph

        # 更新所有任务的依赖
        new_tasks: List[TaskNode] = []
        for t in graph.tasks:
            if t.task_id in merge_map:
                # 被合并的任务不再保留
                continue
            # 替换依赖中的被合并任务
            new_deps = []
            for dep in t.dependencies:
                # 如果 dep 被合并了，且当前任务不是合并后的保留任务，则替换
                if dep in merge_map:
                    final_dep = merge_map[dep]
                    # 避免自依赖
                    if final_dep != t.task_id:
                        new_deps.append(final_dep)
                    else:
                        # 若替换后变成自依赖，去掉该依赖
                        pass
                else:
                    new_deps.append(dep)
            t.dependencies = new_deps
            new_tasks.append(t)

        graph.tasks = new_tasks
        return graph

    # ──────────────────────────── 失败处理 ────────────────────────────

    def _apply_skip_impact(self, graph: TaskGraph, skipped_task: TaskNode):
        """应用 skip 的级联影响到下游任务"""
        downstream = graph.get_downstream_tasks(skipped_task.task_id)
        if not skipped_task.fallback:
            return
        default_params = skipped_task.fallback.default_params or {}

        for dt in downstream:
            # 将下游任务参数中引用被 skip 任务输出的占位符，替换为 default_params
            dt.parameters = self._inject_defaults_into_params(
                dt.parameters, skipped_task.task_id, default_params
            )

    def _inject_defaults_into_params(
        self,
        params: Any,
        skipped_task_id: str,
        defaults: Dict[str, Any],
    ) -> Any:
        """递归将参数中引用被 skip 任务的占位符替换为默认值"""
        if isinstance(params, str):
            # 匹配 {{skipped_task_id.output.xxx}} 或 {{skipped_task_id.output}}
            pattern = r"\{\{" + re.escape(skipped_task_id) + r"\.output(\.[\w\[\]]+)?\}\}"
            if re.search(pattern, params):
                # 简单替换：如果 defaults 只有一个键且值非空，用该值替换
                if len(defaults) == 1:
                    return list(defaults.values())[0]
                # 否则返回 defaults 的 JSON 字符串
                return json.dumps(defaults, ensure_ascii=False)
            return params
        if isinstance(params, dict):
            return {
                k: self._inject_defaults_into_params(v, skipped_task_id, defaults)
                for k, v in params.items()
            }
        if isinstance(params, list):
            return [
                self._inject_defaults_into_params(item, skipped_task_id, defaults)
                for item in params
            ]
        return params

    def _cascade_abort(self, graph: TaskGraph, aborted_task: TaskNode):
        """级联中止所有下游任务"""
        downstream = graph.get_downstream_tasks(aborted_task.task_id)
        for dt in downstream:
            if dt.status in ("pending", "running"):
                dt.status = "aborted"
                dt.observation = f"上游 {aborted_task.task_id} 中止，级联中止"
        aborted_task.status = "failed"

    # ──────────────────────────── Fallback ────────────────────────────

    def _fallback_graph(
        self,
        multi_result: MultiIntentResult,
        resume_text: str,
        history_cache: List[HistoryCacheEntry],
    ) -> TaskGraph:
        """规则兜底：根据意图直接映射到固定任务模板"""
        tasks: List[TaskNode] = []
        primary = multi_result.primary_intent or LLMIntentType.CHAT
        gs = multi_result.global_slots or {}
        has_resume = bool(resume_text) and "尚未上传" not in resume_text

        def add_task(**kwargs) -> TaskNode:
            t = TaskNode(task_id=f"T{len(tasks)}", **kwargs)
            t.params_hash = t.compute_params_hash()
            tasks.append(t)
            return t

        if primary == LLMIntentType.EXPLORE:
            add_task(
                task_type="tool_call", tool_name="kb_retrieve",
                description="检索候选岗位",
                parameters={"query": "{{global_slots.search_keywords}}", "top_k": settings.EXPLORE_TOP_K},
                fallback=TaskFallback(action="ask_user", reason="未找到相关岗位，请提供更多信息"),
            )
            if has_resume:
                add_task(
                    task_type="tool_call", tool_name="global_rank",
                    description="全局匹配排序",
                    parameters={"resume_text": "{{global_slots.resume_text}}", "candidate_jds": "{{T0.output.chunks}}"},
                    dependencies=["T0"],
                    fallback=TaskFallback(action="skip", reason="排序失败，返回原始检索结果", default_params={"rankings": []}),
                )
        elif primary == LLMIntentType.ASSESS:
            add_task(
                task_type="tool_call", tool_name="kb_retrieve",
                description="检索目标岗位JD",
                parameters={"query": "{{global_slots.search_keywords}}", "company": "{{global_slots.company}}", "top_k": settings.ASSESS_TOP_K},
                fallback=TaskFallback(action="ask_user", reason="未找到目标岗位JD，请上传JD图片"),
            )
            if has_resume:
                add_task(
                    task_type="tool_call", tool_name="match_analyze",
                    description="简历匹配分析",
                    parameters={
                        "resume_text": "{{global_slots.resume_text}}",
                        "jd_source": "kb",
                        "jd_data": {"chunks": "{{T0.output.chunks}}"},
                    },
                    dependencies=["T0"],
                    fallback=TaskFallback(action="skip", reason="匹配分析失败", default_params={"score": 0, "gaps": ["分析失败"]}),
                )
        elif primary == LLMIntentType.VERIFY:
            add_task(
                task_type="tool_call", tool_name="kb_retrieve",
                description="检索核实信息",
                parameters={"query": "{{global_slots.search_keywords}}", "top_k": settings.VERIFY_TOP_K},
                fallback=TaskFallback(action="ask_user", reason="未找到相关信息"),
            )
            add_task(
                task_type="tool_call", tool_name="qa_synthesize",
                description="综合回答",
                parameters={"question": "{{global_slots.query}}", "evidence_chunks": "{{T0.output.chunks}}"},
                dependencies=["T0"],
                fallback=TaskFallback(action="skip", reason="问答合成失败", default_params={"answer": "抱歉，暂时无法回答"}),
            )
        elif primary == LLMIntentType.PREPARE:
            add_task(
                task_type="tool_call", tool_name="interview_gen",
                description="生成面试题",
                parameters={"count": 5, "difficulty": "mixed"},
                fallback=TaskFallback(action="skip", reason="面试题生成失败", default_params={"questions": []}),
            )
        elif primary == LLMIntentType.MANAGE:
            add_task(
                task_type="tool_call", tool_name="file_ops",
                description="资料管理",
                parameters={"operation": "{{global_slots.operation}}"},
                fallback=TaskFallback(action="abort", reason="资料管理操作失败"),
            )
        elif primary == LLMIntentType.CHAT:
            add_task(
                task_type="tool_call", tool_name="general_chat",
                description="通用对话",
                parameters={"user_message": "{{global_slots.query}}"},
                fallback=TaskFallback(action="skip", reason="对话失败"),
            )

        # 总是添加 aggregate
        deps = [t.task_id for t in tasks]
        add_task(
            task_type="aggregate",
            description="聚合所有输出生成回复素材",
            parameters={"results": "{{all_outputs}}"},
            dependencies=deps,
            fallback=TaskFallback(action="abort", reason="聚合失败"),
        )

        graph = TaskGraph(
            planner_thought="规则 fallback：根据主意图直接映射到固定任务模板",
            tasks=tasks,
            execution_strategy=ExecutionStrategy(
                parallel_groups=TaskGraph(tasks=tasks).compute_parallel_groups(),
                estimated_cost="medium",
            ),
            context={
                "primary_intent": primary.value,
                "global_slots": gs,
            },
            history_cache=history_cache,
        )
        graph.execution_strategy.critical_path = graph.compute_critical_path()
        return graph
