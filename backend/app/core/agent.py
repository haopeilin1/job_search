"""
Agent 协调器 —— 意图识别 → 工具填充 → 问题改写 → 工具执行

核心流程：
  1. 接收 IntentResult + 用户原始请求
  2. 根据意图选择可用工具（Tool Planning）
  3. 填充工具参数（Tool Parameter Filling）
  4. 改写用户问题为结构化查询（Query Rewriting）
  5. 顺序执行工具链（Tool Execution）
  6. 汇总工具输出，构造给 LLM 的最终上下文
"""

import logging
from typing import List, Optional, Any
from dataclasses import dataclass, field

from app.core.intent import IntentResult, IntentType, RouteLayer
from app.core.tools import (
    ToolCall, ToolResult,
    execute_tool, TOOL_DEFINITIONS,
)
from app.core.llm_client import LLMClient
from app.core.config import settings
from app.core.memory import (
    SessionMemory, DialogueTurn, WorkingMemory, MemoryManager,
    RetrievalDecision,
)
from app.core.retrieval_decision import RetrievalDecisionEngine, FollowUpType, RetrievalAction
from app.core.plan import PlanGenerator, PlanExecutor, PlanExecutionResult



logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# 1. Agent 输出数据模型
# ═══════════════════════════════════════════════════════

@dataclass
class AgentContext:
    """Agent 执行后的完整上下文，用于构造 LLM prompt"""
    original_message: str
    rewritten_query: str
    intent: str
    intent_confidence: float
    intent_layer: str
    selected_tools: List[ToolCall] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)
    kb_chunks: List[dict] = field(default_factory=list)  # 检索召回的 chunks
    resume_text: str = ""
    jd_text: str = ""


@dataclass
class AgentPlan:
    """Agent 执行计划"""
    rewritten_query: str
    tools: List[ToolCall] = field(default_factory=list)
    context: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════
# 2. 意图 → 工具映射规则
# ═══════════════════════════════════════════════════════

INTENT_TOOL_MAP = {
    IntentType.MATCH_SINGLE: {
        "description": "单 JD 匹配：分析简历与具体 JD 的契合度",
        "tools_by_subtype": {
            "referenced_match": ["kb_retrieve", "match_analyze"],      # 引用知识库中的 JD
            "jd_preview": ["match_analyze"],                           # 用户新上传 JD（附件/长文本）
            "attachment_match": ["match_analyze"],                     # 附件+匹配意图
        },
        "default_tools": ["match_analyze"],
    },
    IntentType.GLOBAL_MATCH: {
        "description": "全局对比：将简历与知识库中所有 JD 对比，给出排序和推荐",
        "tools": ["kb_retrieve", "match_analyze"],
    },
    IntentType.RAG_QA: {
        "description": "知识库问答：基于知识库内容回答用户问题",
        "tools": ["kb_retrieve"],
    },
    IntentType.GENERAL: {
        "description": "通用对话：求职咨询、行业闲聊、打招呼等",
        "tools": [],
    },
}


# ═══════════════════════════════════════════════════════
# 3. Agent 协调器
# ═══════════════════════════════════════════════════════

class AgentOrchestrator:
    """
    Agent 协调器：负责"意图识别 → 工具填充 → 问题改写 → 工具执行"的完整链路。
    """

    def __init__(self):
        pass

    # ──────────────────────────── Step 1: 工具规划 ────────────────────────────

    async def plan_tools(self, intent_result: IntentResult, message: str,
                         attachments: list, resume_text: str = "") -> AgentPlan:
        """
        根据意图结果，选择需要调用的工具列表，并填充初始参数。
        """
        tools = []
        context = {
            "message": message,
            "attachments": attachments,
            "resume_text": resume_text,
            "intent": intent_result.intent.value,
            "rule": intent_result.rule,
        }

        intent = intent_result.intent

        # 提取检索关键词（放入 context 供工具规划使用）
        meta = intent_result.metadata or {}
        search_keywords = self._extract_search_keywords(
            message,
            meta.get("company", ""),
            meta.get("position", ""),
            meta.get("attributes", [])
        )
        context["search_keywords"] = search_keywords

        if intent == IntentType.MATCH_SINGLE:
            tools = self._plan_match_single(intent_result, context)
        elif intent == IntentType.GLOBAL_MATCH:
            tools = self._plan_global_match(intent_result, context)
        elif intent == IntentType.RAG_QA:
            tools = self._plan_rag_qa(intent_result, context)
        elif intent == IntentType.GENERAL:
            tools = []

        # 问题改写（LLM 语义改写 + fallback 规则模板）
        rewritten, search_keywords = await self._rewrite_query(intent_result, message, context)
        context["search_keywords"] = search_keywords

        return AgentPlan(
            rewritten_query=rewritten,
            tools=tools,
            context=context,
        )

    def _plan_match_single(self, intent_result: IntentResult, context: dict) -> List[ToolCall]:
        """
        match_single 的工具规划：
        - 引用知识库 JD → 先检索知识库，再分析匹配度
        - 用户新上传 JD → 直接分析匹配度
        """
        tools = []
        meta = intent_result.metadata or {}
        sub_type = meta.get("sub_type", "")
        company = meta.get("company", "")

        # 判断是"引用知识库"还是"新上传 JD"
        is_referenced = bool(company) or sub_type in ("referenced_match",)
        has_attachments = bool(context.get("attachments"))

        if is_referenced:
            # 引用知识库 JD：先检索（使用提取的关键词而非原始消息）
            tools.append(ToolCall(
                name="kb_retrieve",
                parameters={
                    "query": context.get("search_keywords", context["message"]),
                    "company": company,
                    "top_k": settings.RETRIEVAL_TOP_K,
                },
            ))

        # 无论哪种情况，都需要 match_analyze
        tools.append(ToolCall(
            name="match_analyze",
            parameters={
                "resume_text": context.get("resume_text", ""),
                "jd_text": context["message"],  # 临时：后续从检索结果或附件中提取
                "company": company,
            },
        ))

        return tools

    def _plan_global_match(self, intent_result: IntentResult, context: dict) -> List[ToolCall]:
        """
        global_match 的工具规划：
        - 先检索知识库中所有相关 JD
        - 再逐个分析匹配度
        """
        tools = []
        has_resume = bool(context.get("resume_text")) and "尚未上传" not in context.get("resume_text", "")

        # 1. 检索知识库（不加公司过滤，跨所有 JD）
        # 有简历时用简历关键词检索，无简历时用用户问题关键词
        query = context.get("resume_text") if has_resume else context.get("search_keywords", context["message"])
        tools.append(ToolCall(
            name="kb_retrieve",
            parameters={
                "query": query,
                "top_k": settings.RETRIEVAL_TOP_K,
            },
        ))

        # 2. 匹配分析（基于检索结果做全局排序）
        tools.append(ToolCall(
            name="match_analyze",
            parameters={
                "resume_text": context.get("resume_text", ""),
                "jd_text": "[待从检索结果填充]",
            },
        ))

        return tools

    def _plan_rag_qa(self, intent_result: IntentResult, context: dict) -> List[ToolCall]:
        """
        rag_qa 的工具规划：
        - 仅检索知识库，召回相关 chunks
        """
        meta = intent_result.metadata or {}
        company = meta.get("company", "")

        return [ToolCall(
            name="kb_retrieve",
            parameters={
                "query": context.get("search_keywords", context["message"]),
                "company": company,
                "top_k": 5,
            },
        )]

    # ──────────────────────────── Step 2: 问题改写 ────────────────────────────

    async def _rewrite_query(self, intent_result: IntentResult, message: str,
                             context: dict) -> tuple[str, str]:
        """
        将用户的自然语言问题改写为更适合 LLM + 工具执行的结构化查询。

        优先使用 LLM 语义改写，失败时 fallback 到规则模板。

        Returns:
            (structured_query, search_keywords)
        """
        # 1. 优先尝试 LLM 语义改写
        llm_result = await self._llm_rewrite_query(intent_result, message, context)
        if llm_result:
            return llm_result

        # 2. Fallback：规则模板改写
        return self._rule_rewrite_query(intent_result, message, context)

    async def _llm_rewrite_query(self, intent_result: IntentResult, message: str,
                                  context: dict) -> Optional[tuple[str, str]]:
        """
        使用 LLM 进行语义级 query 改写。
        返回 (structured_query, search_keywords)，失败返回 None。
        """
        intent = intent_result.intent
        meta = intent_result.metadata or {}
        company = meta.get("company", "")
        position = meta.get("position", "")
        attrs = meta.get("attributes", [])
        resume_text = context.get("resume_text", "")
        has_resume = bool(resume_text) and "尚未上传" not in resume_text

        intent_desc = {
            IntentType.MATCH_SINGLE: "单JD匹配：用户想分析简历与某个岗位的匹配度",
            IntentType.GLOBAL_MATCH: "全局对比：用户想从知识库中找出最适合自己的岗位并排序",
            IntentType.RAG_QA: "知识库问答：用户想查询知识库中某岗位/公司的具体信息",
            IntentType.GENERAL: "通用对话：打招呼、闲聊、泛泛咨询",
        }.get(intent, "未知意图")

        system_prompt = """你是一位 query 改写专家。请将用户的自然语言问题改写为两个版本：

1. 【structured_query】：去除口语化表达，明确核心意图，适合作为后续 LLM 的输入问题
2. 【search_keywords】：提取 3-8 个适合向量检索的关键词（公司名、岗位名、属性词、技能词等），用空格分隔

改写要求：
- match_single 意图：聚焦"简历与某岗位在技能、经验、项目上的匹配分析"
- global_match 意图：聚焦"多岗位对比，找出最适合用户的并排序"
- rag_qa 意图：聚焦"检索某实体的具体属性信息（薪资、要求、技能等）"
- 如果用户没上传简历，structured_query 中应提示"请先上传简历"
- 保留原文中的核心实体，去除冗余语气词

输出格式（严格 JSON，不要 markdown 代码块，不要解释）：
{"structured_query": "...", "search_keywords": "..."}"""

        user_prompt = f"""用户原始问题：{message}
意图类型：{intent.value}（{intent_desc}）
已知实体：公司={company or '无'}, 岗位={position or '无'}, 属性={', '.join(attrs) if attrs else '无'}
简历状态：{'已上传' if has_resume else '未上传'}

请输出 JSON："""

        try:
            llm = LLMClient.from_config("planner")
            raw = await llm.generate(prompt=user_prompt, system=system_prompt, temperature=0.3, max_tokens=300)

            # 解析 JSON
            import json, re
            raw = raw.strip()
            # 去除可能的 markdown 代码块
            if raw.startswith("```"):
                raw = re.sub(r"^```json\s*|\s*```$", "", raw).strip()
            data = json.loads(raw)

            structured = data.get("structured_query", message)
            keywords = data.get("search_keywords", message)

            logger.info(f"[Agent:LLMRewrite] '{message[:30]}...' -> structured='{structured[:50]}...' keywords='{keywords}'")
            return (structured, keywords)

        except Exception as e:
            logger.warning(f"[Agent:LLMRewrite] LLM 改写失败，fallback 到规则模板: {e}")
            return None

    def _rule_rewrite_query(self, intent_result: IntentResult, message: str,
                            context: dict) -> tuple[str, str]:
        """
        规则模板改写（fallback）。
        Returns: (structured_query, search_keywords)
        """
        intent = intent_result.intent
        meta = intent_result.metadata or {}
        company = meta.get("company", "")
        position = meta.get("position", "")
        attrs = meta.get("attributes", [])
        resume_text = context.get("resume_text", "")
        has_resume = bool(resume_text) and "尚未上传" not in resume_text

        search_keywords = self._extract_search_keywords(message, company, position, attrs)

        if intent == IntentType.MATCH_SINGLE:
            target = position or company or "当前 JD"
            if company and position:
                target = f"{company} · {position}"
            elif company:
                target = company

            if has_resume:
                structured = f"分析用户简历与【{target}】的匹配度：识别核心优势、潜在短板，并给出面试准备建议"
            else:
                structured = f"【{target}】岗位匹配分析：请用户上传简历后进行精准匹配"

        elif intent == IntentType.GLOBAL_MATCH:
            if has_resume:
                structured = "基于用户简历，从知识库中检索并对比最匹配的岗位，给出投递优先级排序及推荐理由"
            else:
                structured = "全局岗位推荐：请用户上传简历后进行多岗位匹配对比"

        elif intent == IntentType.RAG_QA:
            attr_str = "、".join(attrs) if attrs else "具体要求"
            if company:
                structured = f"检索【{company}】的{attr_str}：{search_keywords}"
            else:
                structured = f"检索知识库：{search_keywords}"

        else:
            structured = message

        return (structured, search_keywords)

    def _extract_search_keywords(self, message: str, company: str = "",
                                  position: str = "", attrs: list = None) -> str:
        """
        从用户问题中提取适合向量检索的关键词。
        去除口语化表达，保留公司名、岗位名、属性词等核心实体。
        """
        import re
        noise_words = ["一下", "请问", "我想知道", "能不能", "可以", "吗", "呢", "吧",
                       "帮我", "给我", "看一下", "查一下", "告诉我", "请问"]
        keywords = message
        for w in noise_words:
            keywords = keywords.replace(w, "")
        keywords = keywords.strip()

        parts = []
        if company and company not in keywords:
            parts.append(company)
        if position and position not in keywords:
            parts.append(position)
        if attrs:
            for a in attrs:
                if a not in keywords:
                    parts.append(a)
        parts.append(keywords)
        return " ".join(parts)

    # ──────────────────────────── Step 3: 执行工具链 ────────────────────────────

    async def execute(self, plan: AgentPlan) -> AgentContext:
        """
        顺序执行工具链，收集结果。
        """
        ctx = AgentContext(
            original_message=plan.context["message"],
            rewritten_query=plan.rewritten_query,
            intent=plan.context["intent"],
            intent_confidence=0.0,
            intent_layer="agent",
            selected_tools=plan.tools,
            resume_text=plan.context.get("resume_text", ""),
        )

        for tool_call in plan.tools:
            logger.info(f"[Agent] 执行工具: {tool_call.name} | params={tool_call.parameters}")
            result = await execute_tool(tool_call)
            ctx.tool_results.append(result)

            # 如果是 kb_retrieve，把召回的 chunks 存到上下文
            if tool_call.name == "kb_retrieve" and result.success:
                chunks = result.data.get("chunks", [])
                ctx.kb_chunks.extend(chunks)

            # 如果是 match_analyze，把匹配结果存到上下文
            if tool_call.name == "match_analyze" and result.success:
                ctx.jd_text = result.data.get("jd_text", "")

        return ctx

    # ──────────────────────────── 主入口 ────────────────────────────

    async def run(self, intent_result: IntentResult, message: str,
                  attachments: list, resume_text: str = "") -> AgentContext:
        """
        Agent 完整执行入口。
        """
        logger.info(f"[Agent] 开始执行 | intent={intent_result.intent.value} | rule={intent_result.rule}")

        # 1. 规划工具
        plan = await self.plan_tools(intent_result, message, attachments, resume_text)
        logger.info(f"[Agent] 工具规划完成 | tools={[t.name for t in plan.tools]} | rewritten='{plan.rewritten_query}'")

        # 2. 执行工具链
        ctx = await self.execute(plan)

        logger.info(f"[Agent] 执行完成 | tools_executed={len(ctx.tool_results)} | kb_chunks={len(ctx.kb_chunks)}")
        return ctx


# ═══════════════════════════════════════════════════════
# 4. 辅助：构造 LLM Prompt
# ═══════════════════════════════════════════════════════

def build_llm_prompt(ctx: AgentContext, user_message: str) -> tuple[str, str]:
    """
    根据 Agent 执行结果，构造给 LLM 的 system_prompt 和 user_prompt。

    Returns:
        (system_prompt, user_prompt)
    """
    # System prompt：根据意图类型给出角色设定
    intent = ctx.intent

    if intent == IntentType.MATCH_SINGLE.value:
        system = """你是一位资深猎头顾问，擅长分析简历与岗位描述的匹配度。
请基于下方的【检索信息】和【匹配分析结果】，给出专业、客观、有建设性的分析。
如果提供了匹配分析数据，请直接引用其中的分数、优势和短板。"""
    elif intent == IntentType.GLOBAL_MATCH.value:
        system = """你是一位资深猎头顾问。请基于下方的【检索信息】和【匹配分析结果】，
按匹配度从高到低排序，给出投递策略建议。对每个岗位给出匹配分数和一句话推荐理由。"""
    elif intent == IntentType.RAG_QA.value:
        system = """你是一位求职信息顾问。请基于下方的【知识库检索结果】，准确回答用户问题。
如果检索结果中没有相关信息，请明确告知用户"知识库中暂无该信息"，不要编造。"""
    else:
        system = """你是「求职雷达」AI 助手小橘 🍊，一位专业的求职顾问。
语气友好、专业、有温度。"""

    # User prompt：整合检索结果 + 工具输出 + 用户问题
    parts = []
    parts.append(f"【用户问题】\n{user_message}")
    parts.append(f"【改写后的问题】\n{ctx.rewritten_query}")

    if ctx.kb_chunks:
        parts.append("【知识库检索结果】\n" + "\n---\n".join(
            f"[来源: {c.get('metadata', {}).get('company', '未知')} - {c.get('metadata', {}).get('section', '未知')}]\n{c.get('content', '')[:500]}"
            for c in ctx.kb_chunks
        ))

    for idx, (tool, result) in enumerate(zip(ctx.selected_tools, ctx.tool_results)):
        if result.success and result.data:
            parts.append(f"【工具输出: {tool.name}】\n{json.dumps(result.data, ensure_ascii=False, indent=2)[:1500]}")

    if ctx.resume_text:
        parts.append(f"【用户简历摘要】\n{ctx.resume_text[:2000]}")

    user = "\n\n".join(parts)
    return system, user


# 避免循环引用，放在文件末尾导入
import json


# ═══════════════════════════════════════════════════════
# 5. 增强版 AgentOrchestrator（多轮对话 + 记忆机制）
# ═══════════════════════════════════════════════════════

class EnhancedAgentOrchestrator:
    """
    增强版Agent协调器：
    - 引入 SessionMemory 维护多轮状态
    - 引入 RetrievalDecisionEngine 做检索决策
    - 引入 MemoryManager 做记忆压缩与实体提取
    """

    def __init__(self):
        self.llm = LLMClient.from_config("planner")
        self.retrieval_engine = RetrievalDecisionEngine(llm_client=self.llm)
        self.memory_manager = MemoryManager(llm_client=self.llm)
        self.plan_generator = PlanGenerator(llm_client=self.llm)
        self.plan_executor = PlanExecutor()
        self.base_orchestrator = AgentOrchestrator()  # 保留兼容

    # ──────────────────────────── 核心入口 ────────────────────────────

    async def run(self,
                  intent_result: IntentResult,
                  message: str,
                  attachments: list,
                  resume_text: str = "",
                  session: Optional[SessionMemory] = None,
                  rewritten_query: str = "",
                  search_keywords: str = "",
                  is_follow_up: bool = False,
                  follow_up_type: str = "",
                  ) -> tuple[Any, SessionMemory]:
        """
        完整执行入口（多轮兼容版）

        Args:
            intent_result: 意图识别结果
            message: 用户原始输入
            attachments: 附件列表
            resume_text: 简历文本
            session: 会话记忆
            rewritten_query: Query 改写后的查询
            search_keywords: 检索关键词
            is_follow_up: 是否追问
            follow_up_type: 追问类型

        Returns:
            (AgentContext, SessionMemory)
        """
        # 1. 初始化或复用session
        if session is None:
            import time
            session = SessionMemory(session_id=f"s_{int(time.time() * 1000)}")

        # 2. 话题切换检测（若切换，强制全新检索并清空缓存）
        is_shift = await self.memory_manager.detect_topic_shift(message, session)
        if is_shift:
            logger.info("[EnhancedAgent] 检测到话题切换，清空证据缓存，强制全新检索")

        # 3. 检索决策（非首轮时，使用改写后的 query）
        decision = await self._make_retrieval_decision(rewritten_query or message, session, is_shift)

        # 4. Plan 生成（Query 改写已在 chat.py Step 0 统一完成，直接传入）
        plan = await self.plan_generator.generate(
            intent_result=intent_result,
            decision=decision,
            session=session,
            resume_text=resume_text,
            attachments=attachments,
            original_message=message,
            rewritten_query=rewritten_query,
            search_keywords=search_keywords,
            is_follow_up=is_follow_up,
            follow_up_type=follow_up_type,
        )

        # PlanGenerator 已在 LLM 规划阶段完成 Query 改写，直接读取结果
        final_rewritten = plan.context.get("rewritten_query", message)
        final_keywords = plan.context.get("keywords", message)
        logger.info(f"[EnhancedAgent] Plan 生成完成 | query_rewrite='{final_rewritten[:50]}...' | keywords='{final_keywords}'")

        # 5. Plan 执行：按步骤顺序执行，支持条件跳过和参数传递
        exec_result = await self.plan_executor.execute(plan, session, decision)

        # 5.5 组装 AgentContext
        final_rewritten = plan.context.get("rewritten_query", message)
        logger.info(f"[EnhancedAgent] exec_result | tool_calls={len(exec_result.tool_calls)} | tool_results={len(exec_result.tool_results)} | kb_chunks={len(exec_result.kb_chunks)}")
        ctx = AgentContext(
            original_message=message,
            rewritten_query=final_rewritten,
            intent=intent_result.intent.value,
            intent_confidence=intent_result.confidence,
            intent_layer=intent_result.layer,
            selected_tools=exec_result.tool_calls,
            tool_results=exec_result.tool_results,
            kb_chunks=exec_result.kb_chunks,
            resume_text=resume_text,
            jd_text=exec_result.jd_text,
        )
        # 附加 plan 摘要，供 prompt 构造使用
        ctx._plan_summary = plan.get_step_summary()

        # 更新会话证据缓存
        if exec_result.evidence_cache_updated:
            session.evidence_cache = exec_result.kb_chunks
            session.evidence_cache_query = exec_result.evidence_cache_query
        elif decision.action == RetrievalAction.REUSE and decision.reused_evidence:
            ctx.kb_chunks = decision.reused_evidence

        # 7. 构造本轮对话记录并加入工作内存
        turn = DialogueTurn(
            turn_id=len(session.working_memory.turns) + 1,
            user_message=message,
            assistant_reply="",  # 待LLM生成后回填
            intent=intent_result.intent.value,
            rewritten_query=rewritten,
            tool_calls=exec_result.tool_calls,
            tool_results=exec_result.tool_results,
            retrieved_chunks=ctx.kb_chunks,
            evidence_score=decision.threshold_score if decision else 0.0,
        )
        session.working_memory.append(turn)

        # 8. 记忆轮转（压缩溢出轮次、提取长期记忆、持久化）
        await self.memory_manager.rotate_memory(session, turn)

        # 9. 构造LLM prompt（包含三层记忆上下文 + 执行计划）
        system_prompt, user_prompt = self._build_enriched_prompt(ctx, session, message, decision, plan)
        ctx.system_prompt = system_prompt
        ctx.user_prompt = user_prompt

        return ctx, session

    # ──────────────────────────── 检索决策 ────────────────────────────

    async def _make_retrieval_decision(self, query: str, session: SessionMemory, is_shift: bool) -> RetrievalDecision:
        if is_shift or not session.working_memory.turns:
            return RetrievalDecision(
                action=RetrievalAction.FULL,
                reason="话题切换或首轮对话",
                incremental_query=query,
            )
        return await self.retrieval_engine.decide(query, session)

    # ──────────────────────────── 多轮Query改写 ────────────────────────────

    async def _rewrite_query_with_history(self, intent_result, message, session, decision):
        """
        多轮感知的query改写：
        - 结合工作内存（最近3轮原始对话）
        - 结合压缩记忆（关键事实）
        - 结合长期记忆（用户偏好）
        """
        # 组装历史上下文
        history_parts = []

        if session.working_memory.turns:
            history_parts.append("【最近对话】\n" + session.working_memory.get_recent_context(3))

        if session.compressed_memories:
            # 取最近 4-10 个压缩记忆块（用户要求 4-10 轮压缩记忆）
            recent_compressed = session.compressed_memories[-10:]
            facts = []
            for cm in recent_compressed:
                facts.extend(cm.key_facts)
            if facts:
                history_parts.append("【历史关键事实】\n" + "\n".join(f"- {f}" for f in facts))

        if session.long_term:
            lt = session.long_term
            pref_str = ", ".join([f"{k}={v}" for k, v in lt.preferences.items() if v])
            if pref_str:
                history_parts.append(f"【用户偏好】{pref_str}")
            if lt.entities.get("技能"):
                history_parts.append(f"【已识别技能】{', '.join(lt.entities['技能'])}")

        history_context = "\n\n".join(history_parts)

        system_prompt = f"""你是一位多轮对话query改写专家。当前是第{len(session.working_memory.turns) + 1}轮对话。

历史上下文：
{history_context}

改写要求：
1. 如果用户问题包含指代（"这个"、"那个"、"刚才"），请根据历史上下文将其替换为具体实体
2. 如果用户追问细节（展开型），请在原query基础上补充限定词
3. 如果用户要求新信息（扩展型），请明确新引入的实体/时间
4. 提取3-8个检索关键词（含公司、岗位、技能、属性）

输出JSON：{{"structured_query": "...", "search_keywords": "..."}}"""

        user_prompt = f"用户原始问题：{message}\n上轮检索动作：{decision.action.value if decision else 'full'}"

        try:
            raw = await self.llm.generate(prompt=user_prompt, system=system_prompt, temperature=0.3, max_tokens=400)
            raw = raw.strip().strip("```json").strip("```").strip()
            data = json.loads(raw)
            return data.get("structured_query", message), data.get("search_keywords", message)
        except Exception as e:
            logger.warning(f"[EnhancedAgent] 多轮query改写失败，fallback到规则: {e}")
            return message, message

    # ──────────────────────────── 工具规划（带决策） ────────────────────────────

    async def _plan_tools_with_decision(self, intent_result, message, attachments, resume_text,
                                        rewritten, keywords, decision: RetrievalDecision):
        """
        根据检索决策调整工具链：
        - REUSE: 跳过kb_retrieve，直接match_analyze或qa
        - NO_RETRIEVAL: 仅调用通用工具
        - INCREMENTAL: kb_retrieve + 合并旧证据
        - FULL: 正常走原有逻辑
        """
        tools = []
        context = {
            "message": message,
            "rewritten": rewritten,
            "search_keywords": keywords,
            "resume_text": resume_text,
            "intent": intent_result.intent.value,
        }

        intent = intent_result.intent

        # ── match_single / global_match 必须先检查简历 ──
        has_resume = bool(resume_text) and "尚未上传" not in resume_text
        if intent.value in [IntentType.MATCH_SINGLE.value, IntentType.GLOBAL_MATCH.value] and not has_resume:
            # 没有简历：只检索JD信息，不调用 match_analyze
            if decision.action in (RetrievalAction.FULL, RetrievalAction.INCREMENTAL):
                meta = intent_result.metadata or {}
                tools.append(ToolCall(
                    name="kb_retrieve",
                    parameters={
                        "query": keywords,
                        "company": meta.get("company", ""),
                        "top_k": settings.RETRIEVAL_TOP_K,
                    },
                ))
            return AgentPlan(rewritten_query=rewritten, tools=tools, context=context)

        if decision.action == RetrievalAction.NO_RETRIEVAL:
            pass

        elif decision.action == RetrievalAction.REUSE:
            if intent.value in [IntentType.MATCH_SINGLE.value, IntentType.GLOBAL_MATCH.value]:
                tools.append(ToolCall(
                    name="match_analyze",
                    parameters={
                        "resume_text": resume_text,
                        "jd_text": "[从缓存证据填充]",
                    },
                ))

        elif decision.action == RetrievalAction.INCREMENTAL:
            meta = intent_result.metadata or {}
            tools.append(ToolCall(
                name="kb_retrieve",
                parameters={
                    "query": keywords,
                    "company": meta.get("company", ""),
                    "top_k": settings.RETRIEVAL_TOP_K,
                },
            ))
            if intent.value in [IntentType.MATCH_SINGLE.value, IntentType.GLOBAL_MATCH.value]:
                tools.append(ToolCall(
                    name="match_analyze",
                    parameters={
                        "resume_text": resume_text,
                        "jd_text": "[待从检索结果填充]",
                    },
                ))

        else:  # FULL
            plan = await self.base_orchestrator.plan_tools(intent_result, message, attachments, resume_text)
            tools = plan.tools
            context["search_keywords"] = plan.context.get("search_keywords", keywords)

        return AgentPlan(rewritten_query=rewritten, tools=tools, context=context)

    # ──────────────────────────── 执行计划 ────────────────────────────

    async def _execute_plan(self, plan, session, decision):
        """执行工具链，并根据决策缓存证据"""
        ctx = AgentContext(
            original_message=plan.context["message"],
            rewritten_query=plan.rewritten_query,
            intent=plan.context["intent"],
            intent_confidence=0.0,
            intent_layer="agent",
            selected_tools=plan.tools,
            resume_text=plan.context.get("resume_text", ""),
        )

        for tool_call in plan.tools:
            result = await execute_tool(tool_call)
            ctx.tool_results.append(result)

            if tool_call.name == "kb_retrieve" and result.success:
                ctx.kb_chunks.extend(result.data.get("chunks", []))

            if tool_call.name == "match_analyze" and result.success:
                ctx.jd_text = result.data.get("jd_text", "")

        # 更新证据缓存（仅当执行了检索时）
        if any(t.name == "kb_retrieve" for t in plan.tools) and ctx.kb_chunks:
            session.evidence_cache = ctx.kb_chunks
            session.evidence_cache_query = plan.rewritten_query
        elif decision.action == RetrievalAction.REUSE and decision.reused_evidence:
            ctx.kb_chunks = decision.reused_evidence

        return ctx

    # ──────────────────────────── 增强Prompt构造 ────────────────────────────

    def _build_enriched_prompt(self, ctx, session, message, decision, plan=None):
        """
        构造给LLM的prompt，注入三层记忆：
        1. 工作内存（最近3轮完整对话）
        2. 压缩记忆（4-10轮摘要）
        3. 长期记忆（实体/偏好）
        4. 检索决策说明
        5. 执行计划步骤（新增）
        """
        parts = []

        # ── 1. 工作内存：最近3轮完整对话（排除当前轮，避免把当前问题当历史） ──
        if len(session.working_memory.turns) > 1:
            parts.append("【最近对话历史】\n" + session.working_memory.get_recent_context(3, exclude_last=True))
        elif session.working_memory.turns:
            # 首轮对话，只有当前轮，不显示历史
            pass

        # ── 2. 压缩记忆：历史摘要和关键事实 ──
        if session.compressed_memories:
            # 取最近 4-10 个压缩记忆块（用户要求 4-10 轮压缩记忆）
            recent_cm = session.compressed_memories[-10:]
            cm_lines = ["【更早对话摘要】"]
            for cm in recent_cm:
                cm_lines.append(f"轮次 {cm.start_turn}-{cm.end_turn}：{cm.summary[:300]}")
                for fact in cm.key_facts[:5]:
                    cm_lines.append(f"  - {fact}")
            parts.append("\n".join(cm_lines))

        # ── 3. 当前轮次信息 ──
        parts.append(f"【用户问题】\n{message}")
        parts.append(f"【改写后的问题】\n{ctx.rewritten_query}")

        # ── 4. 检索决策说明 ──
        if decision:
            parts.append(f"【检索策略】{decision.reason}")
            if decision.action == RetrievalAction.REUSE:
                parts.append("⚠️ 本轮基于历史证据直接回答，未执行新检索。")
            elif decision.action == RetrievalAction.NO_RETRIEVAL:
                parts.append("⚠️ 本轮仅改变输出格式，请基于已有信息重新组织。")

        # ── 4.5 执行计划步骤（新增）──
        plan_summary = getattr(ctx, '_plan_summary', '')
        if plan_summary:
            parts.append(plan_summary)

        # ── 5. 检索证据 ──
        if ctx.kb_chunks:
            parts.append("【本轮检索证据】\n" + self._format_chunks(ctx.kb_chunks))
        elif decision and decision.reused_evidence:
            parts.append("【复用历史证据】\n" + self._format_chunks(decision.reused_evidence))

        # ── 6. 工具输出 ──
        for tool, result in zip(ctx.selected_tools, ctx.tool_results):
            if result.success and result.data:
                parts.append(f"【工具输出: {tool.name}】\n{json.dumps(result.data, ensure_ascii=False, indent=2)[:1500]}")

        # ── 7. 简历摘要 ──
        if ctx.resume_text:
            parts.append(f"【用户简历摘要】\n{ctx.resume_text[:2000]}")

        # ── 8. 长期记忆：用户画像 ──
        lt_parts = []
        if session.long_term:
            lt = session.long_term
            if lt.preferences:
                lt_parts.append(f"用户偏好：{lt.preferences}")
            if lt.entities.get("技能"):
                lt_parts.append(f"用户技能：{', '.join(lt.entities['技能'])}")
            if lt.entities.get("公司"):
                lt_parts.append(f"关注公司：{', '.join(lt.entities['公司'])}")
            if lt.entities.get("岗位"):
                lt_parts.append(f"意向岗位：{', '.join(lt.entities['岗位'])}")
        if lt_parts:
            parts.append("【用户画像】\n" + "\n".join(lt_parts))

        system = self._build_system_prompt(ctx.intent, session)
        user = "\n\n".join(parts)

        return system, user

    def _format_chunks(self, chunks):
        return "\n---\n".join(
            f"[来源: {c.get('metadata', {}).get('company', '未知')} - {c.get('metadata', {}).get('section', '未知')}]\n{c.get('content', '')[:500]}"
            for c in chunks
        )

    def _build_system_prompt(self, intent, session):
        """根据意图和长期记忆构建system prompt，并注入工具白名单约束"""
        base_prompts = {
            IntentType.MATCH_SINGLE.value: "你是一位资深猎头顾问，擅长分析简历与岗位描述的匹配度。请基于下方的【检索信息】和【匹配分析结果】，给出专业、客观、有建设性的分析。如果提供了匹配分析数据，请直接引用其中的分数、优势和短板。",
            IntentType.GLOBAL_MATCH.value: "你是一位资深猎头顾问。请基于下方的【检索信息】和【匹配分析结果】，按匹配度从高到低排序，给出投递策略建议。对每个岗位给出匹配分数和一句话推荐理由。",
            IntentType.RAG_QA.value: "你是一位求职信息顾问。请基于下方的【知识库检索结果】，准确回答用户问题。如果检索结果中没有相关信息，请明确告知用户'知识库中暂无该信息'，不要编造。",
        }
        base = base_prompts.get(intent, "你是「求职雷达」AI助手小橘🍊，一位专业的求职顾问。语气友好、专业、有温度。")

        # ── 注入工具白名单约束（Plan 模块生成）──
        tool_constraint = self.plan_generator.get_tool_constraint_prompt(intent)
        base += "\n\n" + tool_constraint

        if session.long_term and session.long_term.preferences.get("行业"):
            base += f"\n用户关注行业：{session.long_term.preferences['行业']}，请优先推荐相关岗位。"

        return base
