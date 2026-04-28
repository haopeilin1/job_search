"""
Plan 模块 —— 基于上下文的 step-by-step 执行计划生成与执行

设计原则：
1. 计划生成是确定性的（规则驱动），但保留 LLM 扩展接口
2. 步骤间支持参数传递（通过占位符机制 {{xxx}}）
3. 条件执行：根据运行时上下文（简历有无、检索决策类型）动态跳过步骤
4. 意图白名单：每种意图只能使用指定工具，在 system prompt 中显式约束
5. 与 EnhancedAgentOrchestrator 解耦：PlanExecutor 返回独立结果，由调用方组装 AgentContext
"""

import logging
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

from app.core.intent import IntentResult, IntentType
from app.core.memory import SessionMemory, RetrievalDecision, RetrievalAction
from app.core.tools import ToolCall, ToolResult, execute_tool

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# 1. 数据模型
# ═══════════════════════════════════════════════════════

class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SKIPPED = "skipped"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class Step:
    """计划中的单一步骤"""
    step_id: int
    name: str
    description: str
    tool: Optional[str] = None          # None 表示虚拟步骤（如证据复用、汇总）
    parameters: dict = field(default_factory=dict)
    condition: Optional[str] = None     # 执行条件标识，如 "has_resume"
    depends_on: List[int] = field(default_factory=list)
    result_key: Optional[str] = None    # 结果存入 runtime_ctx 的 key
    status: StepStatus = StepStatus.PENDING
    tool_result: Optional[ToolResult] = None
    resolved_params: Optional[dict] = None   # 解析占位符后的实际参数


@dataclass
class Plan:
    """完整的执行计划"""
    plan_id: str
    intent: str
    retrieval_decision: str
    allowed_tools: List[str] = field(default_factory=list)
    steps: List[Step] = field(default_factory=list)
    context: dict = field(default_factory=dict)   # 生成时的静态上下文

    def get_step(self, step_id: int) -> Optional[Step]:
        for s in self.steps:
            if s.step_id == step_id:
                return s
        return None

    def get_executable_sequence(self) -> List[Step]:
        """按 step_id 排序的可执行步骤序列（依赖已在前）"""
        return sorted(self.steps, key=lambda s: s.step_id)

    def get_step_summary(self) -> str:
        """人类可读的计划摘要，用于 prompt/debug"""
        lines = [f"执行计划（意图={self.intent}, 决策={self.retrieval_decision}）："]
        for s in self.steps:
            status_icon = "⏳" if s.status == StepStatus.PENDING else (
                "⏭️" if s.status == StepStatus.SKIPPED else (
                    "✅" if s.status == StepStatus.SUCCESS else "❌"
                )
            )
            tool_info = f"[{s.tool}]" if s.tool else "[推理]"
            lines.append(f"  {status_icon} Step{s.step_id}: {s.name} {tool_info}")
        return "\n".join(lines)


@dataclass
class PlanExecutionResult:
    """PlanExecutor 的返回结果，由调用方组装为 AgentContext"""
    kb_chunks: List[dict] = field(default_factory=list)
    jd_text: str = ""
    tool_results: List[ToolResult] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    steps: List[Step] = field(default_factory=list)
    runtime_ctx: dict = field(default_factory=dict)
    evidence_cache_updated: bool = False
    evidence_cache_query: str = ""


# ═══════════════════════════════════════════════════════
# 2. PlanGenerator —— 规则驱动的计划生成器
# ═══════════════════════════════════════════════════════

class PlanGenerator:
    """
    基于上下文生成 step-by-step 执行计划。

    核心逻辑（LLM 驱动）：
    1. 将完整上下文（意图、决策、简历状态、对话历史、可用工具白名单）注入 LLM
    2. LLM 输出结构化 JSON 计划（步骤列表、工具、参数、条件、依赖）
    3. 验证工具名在白名单中，参数合法
    4. 若 LLM 失败或验证不通过，fallback 到规则模板

    扩展性：
    - 可关闭 LLM（llm_client=None）纯规则运行
    - 可调整 temperature 改变计划创造性
    """

    # ── 意图 → 可用工具白名单 ──
    INTENT_TOOL_WHITELIST: Dict[str, List[str]] = {
        IntentType.MATCH_SINGLE.value: ["kb_retrieve", "match_analyze", "interview_questions"],
        IntentType.GLOBAL_MATCH.value: ["kb_retrieve", "match_analyze", "interview_questions"],
        IntentType.RAG_QA.value: ["kb_retrieve"],
        IntentType.GENERAL.value: [],
    }

    # ── 工具参数 Schema（用于 LLM prompt 和验证）──
    TOOL_SCHEMA: Dict[str, dict] = {
        "kb_retrieve": {
            "description": "从知识库检索与查询相关的 JD chunks（混合召回：70%向量 + 30%BM25）",
            "parameters": {
                "query": "检索查询文本（必填）",
                "company": "可选：按公司名过滤",
                "position": "可选：按岗位名过滤",
                "top_k": "返回数量，默认10",
            },
        },
        "match_analyze": {
            "description": "分析简历与 JD 的匹配度，输出分数/优势/短板/建议",
            "parameters": {
                "resume_text": "用户简历文本（必填）",
                "jd_text": "岗位描述文本（必填，可用 {{jd_text}} 占位符）",
                "company": "可选：公司名称",
                "position": "可选：岗位名称",
            },
        },
        "interview_questions": {
            "description": "基于匹配分析结果生成针对性面试题",
            "parameters": {
                "match_result": "match_analyze 的输出对象（可用 {{match_result}} 占位符）",
                "company": "可选：公司名称",
                "position": "可选：岗位名称",
            },
        },
    }

    # ── 规则 Fallback 步骤模板（LLM 失败时使用）──
    RULE_TEMPLATES: Dict[str, List[dict]] = {
        IntentType.MATCH_SINGLE.value: [
            {
                "step_id": 1,
                "name": "检索目标岗位信息",
                "description": "从知识库检索与目标岗位相关的 JD chunks",
                "tool": "kb_retrieve",
                "parameters": {"query": "{{keywords}}", "company": "{{company}}", "top_k": 10},
                "condition": "decision_requires_retrieval",
                "result_key": "kb_retrieve",
            },
            {
                "step_id": 2,
                "name": "分析简历匹配度",
                "description": "将用户简历与检索到的 JD 进行匹配分析",
                "tool": "match_analyze",
                "parameters": {"resume_text": "{{resume_text}}", "jd_text": "{{jd_text}}", "company": "{{company}}", "position": "{{position}}"},
                "condition": "has_resume",
                "depends_on": [1],
                "result_key": "match_analyze",
            },
            {
                "step_id": 3,
                "name": "生成针对性面试题",
                "description": "基于匹配分析结果生成面试题",
                "tool": "interview_questions",
                "parameters": {"match_result": "{{match_result}}", "company": "{{company}}", "position": "{{position}}"},
                "condition": "has_resume_and_match",
                "depends_on": [2],
                "result_key": "interview_questions",
            },
        ],
        IntentType.GLOBAL_MATCH.value: [
            {
                "step_id": 1,
                "name": "检索相关岗位集合",
                "description": "从知识库检索多个相关岗位",
                "tool": "kb_retrieve",
                "parameters": {"query": "{{keywords}}", "top_k": 10},
                "condition": "decision_requires_retrieval",
                "result_key": "kb_retrieve",
            },
            {
                "step_id": 2,
                "name": "全局匹配分析",
                "description": "将简历与多个岗位进行全局对比分析",
                "tool": "match_analyze",
                "parameters": {"resume_text": "{{resume_text}}", "jd_text": "{{jd_text}}"},
                "condition": "has_resume",
                "depends_on": [1],
                "result_key": "match_analyze",
            },
        ],
        IntentType.RAG_QA.value: [
            {
                "step_id": 1,
                "name": "检索知识库",
                "description": "从知识库检索与用户问题相关的信息",
                "tool": "kb_retrieve",
                "parameters": {"query": "{{keywords}}", "company": "{{company}}", "top_k": 5},
                "condition": "decision_requires_retrieval",
                "result_key": "kb_retrieve",
            },
        ],
        IntentType.GENERAL.value: [],
    }

    def __init__(self, llm_client=None):
        self.llm = llm_client
        self.use_llm = llm_client is not None

    # ──────────────────────────── 主入口 ────────────────────────────

    async def generate(
        self,
        intent_result: IntentResult,
        decision: RetrievalDecision,
        session: SessionMemory,
        resume_text: str,
        attachments: list,
        original_message: str = "",
        rewritten_query: str = "",
        search_keywords: str = "",
        is_follow_up: bool = False,
        follow_up_type: str = "",
    ) -> Plan:
        """
        生成执行计划。
        策略：优先 LLM 生成（Query 改写已在 chat.py Step 0 统一完成），失败/验证不通过则 fallback 到规则模板。
        """
        intent = intent_result.intent.value
        meta = intent_result.metadata or {}
        company = meta.get("company", "")
        position = meta.get("position", "")
        has_resume = bool(resume_text) and "尚未上传" not in resume_text

        # ── 构建静态上下文 ──
        plan_context = {
            "original_message": original_message,
            "rewritten_query": rewritten_query or original_message,
            "keywords": search_keywords or original_message,
            "company": company,
            "position": position,
            "has_resume": has_resume,
            "has_attachments": bool(attachments),
            "decision_action": decision.action.value,
            "decision_requires_retrieval": decision.action in (RetrievalAction.FULL, RetrievalAction.INCREMENTAL),
            "has_evidence": bool(session.evidence_cache),
            "resume_text": resume_text,
            "user_provided_jd": getattr(session, "user_provided_jd", ""),
            "is_follow_up": is_follow_up,
            "follow_up_type": follow_up_type,
        }

        # ── 尝试 LLM 生成 ──
        if self.use_llm:
            try:
                llm_plan = await self._generate_with_llm(
                    intent_result=intent_result,
                    decision=decision,
                    session=session,
                    plan_context=plan_context,
                )
                if llm_plan and self._validate_plan(llm_plan, intent):
                    logger.info(f"[PlanGenerator] LLM 生成计划成功 | intent={intent} | steps={len(llm_plan.steps)}")
                    for s in llm_plan.steps:
                        logger.info(f"  Step {s.step_id}: {s.name} | tool={s.tool} | condition={s.condition}")
                    return llm_plan
            except Exception as e:
                logger.warning(f"[PlanGenerator] LLM 生成计划异常: {e}")

        # ── Fallback：规则模板 ──
        logger.info(f"[PlanGenerator] Fallback 到规则模板 | intent={intent}")
        return self._generate_rule_based(intent, decision, plan_context, session)

    # ──────────────────────────── LLM 生成 ────────────────────────────

    async def _generate_with_llm(
        self,
        intent_result: IntentResult,
        decision: RetrievalDecision,
        session: SessionMemory,
        plan_context: dict,
    ) -> Optional[Plan]:
        """调用 LLM 生成结构化执行计划"""

        system_prompt = """你是一位智能助手的计划生成专家。请根据上下文生成一个 step-by-step 的工具执行计划。"""

        user_prompt = self._build_plan_prompt(intent_result, decision, session, plan_context)

        raw = await self.llm.generate(
            prompt=user_prompt,
            system=system_prompt,
            temperature=0.2,
            max_tokens=1200,
        )

        return self._parse_llm_plan(raw, plan_context, session, decision)

    def _build_plan_prompt(
        self,
        intent_result: IntentResult,
        decision: RetrievalDecision,
        session: SessionMemory,
        plan_context: dict,
    ) -> str:
        """构建给 LLM 的计划生成 prompt"""

        intent = intent_result.intent.value
        allowed = self.INTENT_TOOL_WHITELIST.get(intent, [])
        meta = intent_result.metadata or {}

        # 工具白名单描述
        tool_desc_lines = []
        for t in allowed:
            schema = self.TOOL_SCHEMA.get(t, {})
            tool_desc_lines.append(f"- {t}: {schema.get('description', '')}")
            params = schema.get("parameters", {})
            for pname, pdesc in params.items():
                tool_desc_lines.append(f"    · {pname}: {pdesc}")

        # 最近对话历史（用于 LLM 理解上下文）
        history = ""
        if session.working_memory.turns:
            history = session.working_memory.get_recent_context(2)

        lines = [
            "【任务】生成一个 step-by-step 工具执行计划（JSON 格式）。",
            "",
            "【可用工具白名单】（你只能从中选择，严禁使用白名单外的工具）",
        ]
        lines.extend(tool_desc_lines if tool_desc_lines else ["（当前意图不允许使用任何工具）"])
        lines.extend([
            "",
            "【参数占位符规范】",
            "参数值可以是具体字符串，或以下占位符（执行时会自动解析）：",
            "  - {{keywords}}     : 检索关键词",
            "  - {{company}}      : 公司名（从意图 metadata 提取）",
            "  - {{position}}     : 岗位名（从意图 metadata 提取）",
            "  - {{resume_text}}  : 用户简历文本",
            "  - {{jd_text}}      : JD 文本（自动从 kb_retrieve 结果拼接）",
            "  - {{match_result}} : match_analyze 的输出对象",
            "",
            "【条件标识规范】",
            "  - always                     : 无条件执行",
            "  - has_resume                 : 仅当用户已上传简历时执行",
            "  - decision_requires_retrieval: 仅当检索决策为 FULL/INCREMENTAL 时执行",
            "  - has_resume_and_match       : 需简历且匹配分析已完成",
            "",
            "【上下文信息】",
            f"- 意图类型：{intent}",
            f"- 检索决策：{decision.action.value}",
            f"  · FULL        = 全新检索",
            f"  · REUSE       = 复用旧证据（不要调用 kb_retrieve）",
            f"  · INCREMENTAL = 增量检索",
            f"  · NO_RETRIEVAL= 无需检索（不要调用任何工具）",
            f"- 是否有简历：{'是' if plan_context.get('has_resume') else '否'}",
            f"- 用户原始问题：{plan_context.get('original_message', '')}",
            f"- 改写后的问题：{plan_context.get('rewritten_query', '')}",
            f"- 检索关键词：{plan_context.get('keywords', '')}",
            f"- 是否追问：{'是' if plan_context.get('is_follow_up') else '否'} ({plan_context.get('follow_up_type', '')})",
            f"- 已知公司：{plan_context.get('company') or '无'}",
            f"- 已知岗位：{plan_context.get('position') or '无'}",
        ])

        if history:
            lines.extend([
                "",
                "【最近对话历史】",
                history,
            ])

        lines.extend([
            "",
            "【Query 改写状态】",
            "Query 改写已由上游模块完成（指代消解 + 口语降噪 + 追问标记），请直接使用【上下文信息】中的改写后问题和检索关键词生成计划，无需再次改写。",
            "",
            "【计划约束】",
            "1. 只能使用白名单中的工具",
            "2. 若 decision=REUSE，不要调用 kb_retrieve（证据已在缓存中）",
            "3. 若 decision=NO_RETRIEVAL，不要调用任何工具，仅保留一个汇总步骤",
            "4. 若 has_resume=false，match_analyze / interview_questions 必须设 condition='has_resume' 以便跳过",
            "5. 步骤间通过 depends_on 表达依赖（如 match_analyze 依赖 kb_retrieve）",
            "6. 最后必须有一个 tool=null 的汇总步骤（name='汇总生成回复'）",
            "7. 参数中涉及前序步骤产出的，使用占位符（如 {{jd_text}}, {{match_result}}）",
            "",
            "【输出格式】",
            "严格 JSON，不要 markdown 代码块：",
            '{',
            '  "rewritten_query": "改写后的结构化查询",',
            '  "search_keywords": "检索关键词 用空格分隔",',
            '  "steps": [',
            '    {',
            '      "step_id": 1,',
            '      "name": "步骤名称",',
            '      "description": "步骤说明",',
            '      "tool": "工具名或null",',
            '      "parameters": {"参数名": "参数值或占位符"},',
            '      "condition": "条件标识",',
            '      "depends_on": [],',
            '      "result_key": "结果存储key"',
            '    }',
            '  ]',
            '}',
            "",
            "请直接输出 JSON：",
        ])

        return "\n".join(lines)

    def _parse_llm_plan(
        self,
        raw: str,
        plan_context: dict,
        session: SessionMemory,
        decision: RetrievalDecision,
    ) -> Optional[Plan]:
        """解析 LLM 返回的 JSON 为 Plan 对象"""
        import json, re

        if not raw or not raw.strip():
            return None

        text = raw.strip()
        # 去除 markdown 代码块
        for marker in ["```json", "```"]:
            if marker in text:
                text = re.sub(rf"^{re.escape(marker)}\s*|\s*{re.escape(marker)}$", "", text, flags=re.MULTILINE).strip()

        # 提取 JSON
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    return None
            else:
                return None

        if not isinstance(data, dict) or "steps" not in data:
            return None

        # 提取 Query 改写结果
        rewritten_query = data.get("rewritten_query", "")
        search_keywords = data.get("search_keywords", "")

        steps_raw = data["steps"]
        if not isinstance(steps_raw, list):
            return None

        steps: List[Step] = []
        for sr in steps_raw:
            if not isinstance(sr, dict):
                continue
            step = Step(
                step_id=int(sr.get("step_id", len(steps) + 1)),
                name=sr.get("name", ""),
                description=sr.get("description", ""),
                tool=sr.get("tool") if sr.get("tool") else None,
                parameters=sr.get("parameters", {}),
                condition=sr.get("condition") if sr.get("condition") else None,
                depends_on=sr.get("depends_on", []),
                result_key=sr.get("result_key") if sr.get("result_key") else None,
            )
            steps.append(step)

        # 确保有汇总步骤
        if not any(s.tool is None for s in steps):
            steps.append(Step(
                step_id=len(steps) + 1,
                name="汇总生成回复",
                description="基于所有工具输出生成最终回复",
                tool=None,
                result_key="final",
            ))

        # 将 Query 改写结果存入 plan_context
        plan_context["rewritten_query"] = rewritten_query or plan_context.get("original_message", "")
        plan_context["keywords"] = search_keywords or plan_context.get("original_message", "")

        intent = plan_context.get("intent", "general")
        plan = Plan(
            plan_id=f"plan_{session.session_id}_{len(session.working_memory.turns) + 1}",
            intent=intent,
            retrieval_decision=decision.action.value,
            allowed_tools=self.INTENT_TOOL_WHITELIST.get(intent, []),
            steps=steps,
            context=plan_context,
        )
        return plan

    def _validate_plan(self, plan: Plan, intent: str) -> bool:
        """验证 LLM 生成的计划是否合法"""
        allowed = set(self.INTENT_TOOL_WHITELIST.get(intent, []))

        for step in plan.steps:
            if step.tool and step.tool not in allowed:
                logger.warning(f"[PlanGenerator] 验证失败: 步骤 '{step.name}' 使用了不在白名单中的工具 '{step.tool}'")
                return False
            # 验证参数不是空的 dict（除了无工具步骤）
            if step.tool and not step.parameters:
                logger.warning(f"[PlanGenerator] 验证失败: 步骤 '{step.name}' 参数为空")
                return False

        return True

    # ──────────────────────────── 规则 Fallback ────────────────────────────

    def _generate_rule_based(
        self,
        intent: str,
        decision: RetrievalDecision,
        plan_context: dict,
        session: SessionMemory,
    ) -> Plan:
        """规则模板生成（LLM 失败时的兜底），同时完成规则 Query 改写"""
        templates = self.RULE_TEMPLATES.get(intent, []).copy()
        steps: List[Step] = []

        if decision.action == RetrievalAction.NO_RETRIEVAL:
            templates = []

        # 如果用户已上传 JD（OCR 提取），跳过 kb_retrieve 步骤
        has_user_jd = bool(getattr(session, "user_provided_jd", ""))

        # ── 规则 Query 改写 ──
        original_message = plan_context.get("original_message", "")
        company = plan_context.get("company", "")
        position = plan_context.get("position", "")
        has_resume = plan_context.get("has_resume", False)

        if intent == "match_single":
            target = position or company or "当前 JD"
            if company and position:
                target = f"{company} · {position}"
            rewritten = f"分析用户简历与【{target}】的匹配度" if has_resume else f"【{target}】岗位匹配分析"
            keywords = f"{company} {position} 匹配度".strip()
        elif intent == "global_match":
            rewritten = "基于用户简历对比知识库全部岗位，给出匹配排序和投递建议" if has_resume else "全局岗位推荐"
            keywords = "岗位对比 匹配排序 推荐"
        elif intent == "rag_qa":
            attrs = plan_context.get("attributes", [])
            attr_str = "、".join(attrs) if attrs else "具体要求"
            rewritten = f"检索{company or '知识库'}的{attr_str}" if company else "检索知识库"
            keywords = f"{company} {position} {attr_str}".strip()
        else:
            rewritten = original_message
            keywords = original_message

        plan_context["rewritten_query"] = rewritten
        plan_context["keywords"] = keywords

        for tmpl in templates:
            # 用户自带 JD 时，跳过知识库检索步骤
            if has_user_jd and tmpl.get("tool") == "kb_retrieve":
                continue
            step = Step(
                step_id=tmpl["step_id"],
                name=tmpl["name"],
                description=tmpl["description"],
                tool=tmpl.get("tool"),
                parameters=tmpl.get("parameters", {}).copy(),
                condition=tmpl.get("condition"),
                depends_on=tmpl.get("depends_on", []),
                result_key=tmpl.get("result_key"),
            )
            # 用户自带 JD 时，match_analyze 的 depends_on 清空（不需要依赖 kb_retrieve）
            if has_user_jd and step.tool == "match_analyze":
                step.depends_on = []
                step.description = "将用户简历与用户上传的 JD 图片内容进行匹配分析"
            if decision.action == RetrievalAction.REUSE and step.tool == "kb_retrieve":
                step.tool = None
                step.condition = None
                step.description = "复用历史检索证据（REUSE 决策）：从 evidence_cache 加载上轮检索结果"
            steps.append(step)

        steps.append(Step(
            step_id=len(steps) + 1,
            name="汇总生成回复",
            description="基于所有工具输出、记忆上下文和约束条件生成最终回复",
            tool=None,
            result_key="final",
        ))

        return Plan(
            plan_id=f"plan_{session.session_id}_{len(session.working_memory.turns) + 1}",
            intent=intent,
            retrieval_decision=decision.action.value,
            allowed_tools=self.INTENT_TOOL_WHITELIST.get(intent, []),
            steps=steps,
            context=plan_context,
        )

    # ──────────────────────────── 辅助方法 ────────────────────────────

    def get_allowed_tools(self, intent: str) -> List[str]:
        return self.INTENT_TOOL_WHITELIST.get(intent, [])

    def get_tool_constraint_prompt(self, intent: str) -> str:
        allowed = self.get_allowed_tools(intent)
        if not allowed:
            return "【工具约束】当前意图为通用对话（general），不允许调用任何工具，请直接基于已有知识和记忆回复。"

        lines = ["【工具约束】根据当前用户意图，你只能使用以下工具："]
        for t in allowed:
            schema = self.TOOL_SCHEMA.get(t, {})
            lines.append(f"  - {t}：{schema.get('description', '')}")
        lines.append("严禁使用未在白名单中的工具。若需要的信息超出工具能力，请明确告知用户。")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════
# 3. PlanExecutor —— 计划执行器
# ═══════════════════════════════════════════════════════

class PlanExecutor:
    """
    按步骤顺序执行计划，支持：
    - 条件跳过（基于 runtime context）
    - 参数占位符解析（{{xxx}} → 实际值）
    - 步骤间结果传递（前序步骤的结果注入到后续步骤参数）
    - evidence_cache 自动更新
    """

    # ── 条件评估函数表 ──
    CONDITION_EVALUATORS: Dict[str, Callable[[Dict, SessionMemory, RetrievalDecision], bool]] = {
        "has_resume": lambda ctx, session, decision: ctx.get("has_resume", False),
        "has_resume_and_match": lambda ctx, session, decision: ctx.get("has_resume", False),
        "decision_requires_retrieval": lambda ctx, session, decision: decision.action in (RetrievalAction.FULL, RetrievalAction.INCREMENTAL),
        "has_evidence": lambda ctx, session, decision: bool(session.evidence_cache),
    }

    async def execute(
        self,
        plan: Plan,
        session: SessionMemory,
        decision: RetrievalDecision,
    ) -> PlanExecutionResult:
        """
        执行完整计划，返回 PlanExecutionResult。
        调用方（EnhancedAgentOrchestrator）用此结果组装 AgentContext。
        """
        # 运行时上下文：存储变量 + 步骤结果
        runtime_ctx = plan.context.copy()
        runtime_ctx["steps"] = {}   # step_id -> step_result_data

        result = PlanExecutionResult()

        for step in plan.get_executable_sequence():
            # ── 1. 条件判断 ──
            if step.condition and not self._evaluate_condition(
                step.condition, runtime_ctx, session, decision
            ):
                step.status = StepStatus.SKIPPED
                logger.info(f"[PlanExecutor] Step {step.step_id} SKIPPED | condition={step.condition}")
                result.steps.append(step)
                continue

            # ── 2. 解析参数占位符 ──
            if step.tool:
                resolved_params = self._resolve_parameters(step, runtime_ctx, session, decision)
                step.resolved_params = resolved_params
            else:
                resolved_params = {}
                step.resolved_params = {}

            # ── 3. 执行 ──
            if step.tool:
                step.status = StepStatus.RUNNING
                tool_call = ToolCall(name=step.tool, parameters=resolved_params)
                result.tool_calls.append(tool_call)

                try:
                    tool_result = await execute_tool(tool_call)
                    step.tool_result = tool_result
                    step.status = StepStatus.SUCCESS if tool_result.success else StepStatus.FAILED
                    result.tool_results.append(tool_result)

                    # 收集到 runtime_ctx
                    runtime_ctx["steps"][step.step_id] = {
                        "tool": step.tool,
                        "result": tool_result,
                        "success": tool_result.success,
                    }

                    # 按工具类型提取结构化数据
                    if step.tool == "kb_retrieve" and tool_result.success:
                        chunks = tool_result.data.get("chunks", [])
                        result.kb_chunks.extend(chunks)
                        runtime_ctx["steps"][step.step_id]["chunks"] = chunks

                    if step.tool == "match_analyze" and tool_result.success:
                        result.jd_text = tool_result.data.get("jd_text", "")
                        runtime_ctx["steps"][step.step_id]["data"] = tool_result.data

                    if step.tool == "interview_questions" and tool_result.success:
                        runtime_ctx["steps"][step.step_id]["data"] = tool_result.data

                    logger.info(
                        f"[PlanExecutor] Step {step.step_id} {step.status.value} | "
                        f"tool={step.tool} | params_keys={list(resolved_params.keys())}"
                    )

                except Exception as e:
                    step.status = StepStatus.FAILED
                    step.tool_result = ToolResult(success=False, error=str(e))
                    result.tool_results.append(step.tool_result)
                    logger.error(f"[PlanExecutor] Step {step.step_id} FAILED | tool={step.tool} | error={e}")

            else:
                # ── 虚拟步骤处理 ──
                if step.result_key == "kb_retrieve" and decision.action == RetrievalAction.REUSE:
                    # 从 evidence_cache 加载历史证据
                    cached = decision.reused_evidence or session.evidence_cache or []
                    result.kb_chunks = list(cached)
                    runtime_ctx["steps"][step.step_id] = {
                        "tool": "kb_retrieve",
                        "from_cache": True,
                        "chunks": result.kb_chunks,
                    }
                    step.status = StepStatus.SUCCESS
                    logger.info(
                        f"[PlanExecutor] Step {step.step_id} SUCCESS | "
                        f"loaded {len(result.kb_chunks)} chunks from evidence_cache (REUSE)"
                    )
                else:
                    step.status = StepStatus.SUCCESS

            result.steps.append(step)

        # ── 4. 更新证据缓存 ──
        if any(
            s.tool == "kb_retrieve" and s.status == StepStatus.SUCCESS
            for s in result.steps
        ):
            result.evidence_cache_updated = True
            result.evidence_cache_query = plan.context.get("rewritten_query", "")

        result.runtime_ctx = runtime_ctx
        return result

    # ── 内部方法 ──

    def _evaluate_condition(
        self,
        condition: str,
        runtime_ctx: Dict,
        session: SessionMemory,
        decision: RetrievalDecision,
    ) -> bool:
        """评估步骤执行条件，支持 AND 组合"""
        parts = [p.strip() for p in condition.split(" AND ")]
        for part in parts:
            evaluator = self.CONDITION_EVALUATORS.get(part)
            if evaluator is None:
                logger.warning(f"[PlanExecutor] 未知条件: {part}，默认通过")
                continue
            if not evaluator(runtime_ctx, session, decision):
                return False
        return True

    def _resolve_parameters(
        self,
        step: Step,
        runtime_ctx: Dict,
        session: SessionMemory,
        decision: RetrievalDecision,
    ) -> Dict[str, Any]:
        """解析参数中的 {{placeholder}} 占位符"""
        resolved = {}
        for key, val in step.parameters.items():
            if isinstance(val, str) and val.startswith("{{") and val.endswith("}}"):
                placeholder = val[2:-2].strip()
                resolved[key] = self._resolve_placeholder(
                    placeholder, step, runtime_ctx, session, decision
                )
            else:
                resolved[key] = val
        return resolved

    def _resolve_placeholder(
        self,
        placeholder: str,
        step: Step,
        runtime_ctx: Dict,
        session: SessionMemory,
        decision: RetrievalDecision,
    ) -> Any:
        """解析单个占位符为实际值"""

        # 1. 直接匹配 runtime_ctx 中的变量
        if placeholder in runtime_ctx:
            return runtime_ctx[placeholder]

        # 2. 语义化占位符
        if placeholder == "kb_chunks_text":
            return self._build_chunks_text(runtime_ctx, session)

        if placeholder == "jd_text":
            # 如果用户已上传 JD（OCR 提取），直接使用，跳过知识库检索
            if getattr(session, "user_provided_jd", ""):
                return session.user_provided_jd
            # 优先从已执行的 kb_retrieve 步骤获取文本
            text = self._build_chunks_text(runtime_ctx, session)
            if text and text != "（无检索结果）":
                return text
            # fallback：从当前 evidence_cache
            if session.evidence_cache:
                return self._build_chunks_text_from_list(session.evidence_cache)
            return "（暂无 JD 文本）"

        if placeholder == "match_result":
            # 从已执行的 match_analyze 步骤获取结果数据
            for sid, data in runtime_ctx.get("steps", {}).items():
                if data.get("tool") == "match_analyze" and data.get("success"):
                    return data.get("data", {})
            return {}

        if placeholder == "evidence_cache":
            return session.evidence_cache

        # 3. 步骤结果引用：step_N.field
        if placeholder.startswith("step_"):
            parts = placeholder.split(".", 1)
            if len(parts) == 2:
                step_id_str, field = parts
                try:
                    sid = int(step_id_str.replace("step_", ""))
                    step_data = runtime_ctx.get("steps", {}).get(sid, {})
                    return step_data.get(field, "")
                except ValueError:
                    pass

        logger.warning(f"[PlanExecutor] 未解析的占位符: {placeholder}，保留原样")
        return f"{{{{{placeholder}}}}}"

    def _build_chunks_text(self, runtime_ctx: Dict, session: SessionMemory) -> str:
        """从 runtime_ctx 的 kb_retrieve 步骤或 evidence_cache 构建 chunks 文本"""
        for sid, data in runtime_ctx.get("steps", {}).items():
            if "chunks" in data:
                return self._build_chunks_text_from_list(data["chunks"])
        if session.evidence_cache:
            return self._build_chunks_text_from_list(session.evidence_cache)
        return "（无检索结果）"

    @staticmethod
    def _build_chunks_text_from_list(chunks: List[dict]) -> str:
        texts = []
        for c in chunks:
            meta = c.get("metadata", {})
            company = meta.get("company", "未知")
            section = meta.get("section", "")
            content = c.get("content", "")
            header = f"【{company}" + (f" - {section}" if section else "") + "】"
            texts.append(f"{header}\n{content}")
        return "\n\n---\n\n".join(texts)
