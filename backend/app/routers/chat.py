import json
import logging
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from typing import Optional, List, Any
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from app.core.llm_client import LLMClient, TIMEOUT_HEAVY, TIMEOUT_STANDARD, TIMEOUT_LIGHT
from app.core.memory import SessionMemory, MemoryManager, DialogueTurn
from app.core.db import load_session_meta, load_long_term_memory
from app.core.config import settings
from app.core.query_rewrite import QueryRewriter, QueryRewriteResult
from app.core.intent_recognition import IntentRecognizer, IntentResult as NewIntentResult
from app.core.planner import TaskPlanner
from app.core.react_executor import ReActExecutor
from app.services.ocr_service import OCRService, OCRResult
from app.services.handlers import (
    ChatRequest,
    ChatReply,
    _get_resume_text,
)

# 旧架构模块（v1，待废弃）
from app.core.task_graph import HistoryCacheEntry
from app.core.tool_registry import ToolCall

# 规则路线专用（LLM 模式不共用）
from app.core.intent import IntentRouter, IntentResult, IntentType
from app.core.agent import EnhancedAgentOrchestrator
from app.services.handlers import (
    handle_match_single,
    handle_global_match,
    handle_rag_qa,
    handle_general,
)

# ──────────────────────────── 全局 Session 存储（内存 + DB 双写） ────────────────────────────

# session_id -> SessionMemory 的映射（内存缓存）
# 长期记忆持久化到 SQLite，内存重启后从数据库恢复
_session_store: dict[str, SessionMemory] = {}


def _get_or_create_session(session_id: Optional[str], user_id: Optional[str] = None) -> tuple[str, SessionMemory]:
    """
    根据 session_id 获取或创建 SessionMemory。
    若 session_id 存在且内存中没有，尝试从数据库恢复元数据。
    长期记忆始终从数据库加载（按 user_id 或 session_id）。
    """
    import time
    if not session_id:
        session_id = f"s_{int(time.time() * 1000)}"

    if session_id in _session_store:
        return session_id, _session_store[session_id]

    # 新建 SessionMemory
    session = SessionMemory(session_id=session_id)

    # 尝试恢复会话元数据
    meta = load_session_meta(session_id)
    if meta:
        session.current_topic = meta.get("current_topic", "general")
        session.evidence_cache_query = meta.get("evidence_cache_query", "")
        # 如果数据库中有 user_id，优先用数据库的
        if meta.get("user_id"):
            user_id = meta["user_id"]

    # 加载长期记忆（按 user_id，如果没有则用 session_id 作为临时标识）
    lt_key = user_id or session_id
    lt = load_long_term_memory(lt_key)
    if lt:
        session.long_term = lt
        logger.info(f"[Chat] 长期记忆恢复成功 | user_id={lt_key}")
    else:
        # 初始化一个空的长期记忆
        session.long_term = MemoryManager.load_long_term(lt_key) or None

    # 恢复最近对话历史（从数据库加载到 WorkingMemory）
    from app.core.db import load_session_dialogue_history
    history = load_session_dialogue_history(session_id)
    if history:
        from app.core.memory import DialogueTurn
        for h in history:
            turn = DialogueTurn(
                turn_id=h.get("turn_id", 0),
                user_message=h.get("user_message", ""),
                assistant_reply=h.get("assistant_reply", ""),
                intent=h.get("intent", ""),
                rewritten_query=h.get("rewritten_query", ""),
                evidence_score=h.get("evidence_score", 0.0),
                task_graph_snapshot=h.get("task_graph_snapshot"),
            )
            session.working_memory.append(turn)
        logger.info(f"[Chat] 对话历史恢复成功 | turns={len(session.working_memory.turns)}")

    _session_store[session_id] = session
    return session_id, session


# ──────────────────────────── 现有接口（保留兼容） ────────────────────────────

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


class ChatMessageRequest(BaseModel):
    message: str = Field(..., description="用户发送的消息")
    session_id: Optional[str] = Field(None, description="对话会话 ID")


class ChatMessageResponse(BaseModel):
    role: str = Field("bot", description="消息角色")
    text: str = Field(..., description="回复文本")


@router.post("/message", response_model=ChatMessageResponse)
async def chat_message(req: ChatMessageRequest):
    """处理用户对话消息，基于内容返回知识库查询或通用回复（旧版接口，保留兼容）"""
    lower = req.message.lower()
    
    if "bytedance" in lower or "字节" in req.message:
        return ChatMessageResponse(
            text="📋 知识库匹配结果：\n\n**ByteDance** · AI 平台产品经理\n负责大模型基础设施产品设计。\n\n📍 北京　💰 30k-60k"
        )
    
    if "baidu" in lower or "百度" in req.message:
        return ChatMessageResponse(
            text="📋 知识库匹配结果：\n\n**Baidu** · 大模型应用 PM\n探索文心一言在垂直行业的落地场景。\n\n📍 上海　💰 25k-50k"
        )
    
    if "jd" in lower or "职位" in req.message or "工作" in req.message or "岗位" in req.message:
        return ChatMessageResponse(
            text="📚 目前知识库共有 **4** 个职位。你可以：\n1. 直接输入公司名称查询详情\n2. 点击顶部「知识库」图标进入管理页\n3. 发送 JD 截图给我，直接开始匹配"
        )
    
    return ChatMessageResponse(
        text="收到！如果是关于职位的问题，可以直接问我知识库内容；如果想看匹配度，上传 JD 截图是最快的办法 📎"
    )


# ──────────────────────────── 统一对话入口（意图识别 + Agent 工具链 + 多轮记忆） ────────────────────────────

api_router = APIRouter(tags=["chat"])

# 规则路线：Handler 映射表（fallback 兼容）
RULE_HANDLERS = {
    IntentType.MATCH_SINGLE: handle_match_single,
    IntentType.GLOBAL_MATCH: handle_global_match,
    IntentType.RAG_QA: handle_rag_qa,
    IntentType.GENERAL: handle_general,
}


def _create_intent_router():
    """动态创建 IntentRouter（规则路线专用）"""
    try:
        llm = LLMClient.from_config("planner")
    except Exception:
        llm = None
    return IntentRouter(llm_client=llm)


@api_router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    统一对话入口（双路线独立，不共用组件）。

    路线选择（配置驱动）：
    - AGENT_MODE="rule" → 规则路线（完整 IntentRouter → Plan → 执行 → LLM 回复）
    - AGENT_MODE="llm"  → LLM 路线（Query 改写 → 直接 LLM 回复，其余模块待填充）
    """
    # 1. 会话管理
    session_id, session = _get_or_create_session(request.session_id, request.user_id)
    resume_text = _get_resume_text()

    # 1.5 处理图片附件：OCR 提取文字后附加到 message
    message_with_ocr = request.message
    ocr_results = []
    if request.attachments:
        for att in request.attachments:
            if att.content_type and att.content_type.startswith("image/"):
                if att.data:
                    try:
                        import base64
                        image_bytes = base64.b64decode(att.data)
                        ocr_service = OCRService()
                        ocr_result = await ocr_service.extract(image_bytes, filename=att.filename)
                        ocr_results.append(ocr_result)
                        logger.info(
                            f"[Chat] OCR 提取 | {att.filename} | backend={ocr_result.backend} | "
                            f"chars={len(ocr_result.text)} | conf={ocr_result.confidence:.2f}"
                        )
                    except Exception as e:
                        logger.error(f"[Chat] OCR 失败 | {att.filename}: {e}")

    user_provided_jd = None
    if ocr_results:
        ocr_text = "\n\n".join(
            f"【图片提取内容 - {r.backend}】\n{r.text}" for r in ocr_results
        )
        message_with_ocr = f"{request.message}\n\n{ocr_text}"
        # 提取纯 JD 文本用于直接匹配（跳过知识库检索）
        user_provided_jd = "\n\n".join(r.text for r in ocr_results)
        request.user_provided_jd = user_provided_jd
        session.user_provided_jd = user_provided_jd
        logger.info(f"[Chat] 用户上传 JD 图片，OCR 提取 {len(user_provided_jd)} 字符，将跳过知识库检索")

    # ── 路线选择：配置驱动 ──
    agent_mode = settings.AGENT_MODE
    if agent_mode == "auto":
        effective_mode = settings.DEFAULT_AGENT_MODE
    else:
        effective_mode = agent_mode

    use_llm_agent = effective_mode == "llm"
    logger.info(f"[Chat] 路线选择 | config_mode={agent_mode} | effective_mode={effective_mode} | use_llm_agent={use_llm_agent} | msg='{request.message[:40]}...'")

    # ════════════════════════════════════════
    # Step 0: Query 改写（两条路线共用）
    # ════════════════════════════════════════
    rewriter = QueryRewriter()
    rewrite_result = await rewriter.rewrite(raw_query=message_with_ocr, session=session)
    logger.info(
        f"[Chat] Query改写 | original='{request.message[:30]}...' "
        f"| rewritten='{rewrite_result.rewritten_query[:40]}...' "
        f"| follow_up={rewrite_result.is_follow_up}/{rewrite_result.follow_up_type}"
    )

    if use_llm_agent:
        # ════════════════════════════════════════
        # LLM 路线 v2（新架构：Query改写→意图识别→Plan→ReAct执行）
        # ════════════════════════════════════════
        import time
        from app.core.telemetry import create_tracker

        request_start_time = time.time()
        turn_id = len(session.working_memory.turns) + 1
        tracker = create_tracker(
            session_id=session_id,
            turn_id=turn_id,
            eval_context=request.eval_context or {},
        )

        return await _handle_llm_route_v2(
            request=request,
            session=session,
            session_id=session_id,
            message_with_ocr=message_with_ocr,
            rewrite_result=rewrite_result,
            resume_text=resume_text,
            tracker=tracker,
            request_start_time=request_start_time,
        )
    else:
        # ════════════════════════════════════════
        # 规则路线（完整链路，独立组件）
        # ════════════════════════════════════════
        return await _handle_rule_route(
            request=request,
            session=session,
            session_id=session_id,
            message_with_ocr=message_with_ocr,
            rewrite_result=rewrite_result,
            resume_text=resume_text,
        )


# ──────────────────────────── LLM 路线（独立，待填充） ────────────────────────────

async def _execute_single_intent_tools(
    intent: "LLMIntentType",
    slots: dict,
    request: "ChatRequest",
    session: "SessionMemory",
    rewrite_result: "QueryRewriteResult",
    resume_text: str,
    registry,
) -> tuple[list, list]:
    """执行单一意图的工具链，返回 (tool_results, tool_summary)"""
    tool_results = []
    tool_summary = []

    if intent.value == "explore":
        explore_top_k = slots.get("top_k", settings.EXPLORE_TOP_K)
        explore_filters = slots.get("filters", {})
        t1 = await registry.execute(ToolCall(
            name="kb_retrieve",
            params={
                "query": slots.get("search_keywords") or rewrite_result.search_keywords or rewrite_result.rewritten_query,
                "top_k": explore_top_k,
                "filters": explore_filters,
            },
            task_id=0,
        ))
        tool_results.append(t1)
        if t1.success and t1.data.get("chunks"):
            tool_summary.append({"tool": "kb_retrieve", "chunks": len(t1.data.get("chunks", []))})
            chunks = t1.data.get("chunks", [])
            candidate_jds = []
            seen = set()
            for c in chunks:
                meta = c.get("metadata", {})
                key = f"{meta.get('company','')}#{meta.get('position','')}"
                if key not in seen:
                    seen.add(key)
                    candidate_jds.append({
                        "jd_id": c.get("id", ""),
                        "company": meta.get("company", "未知"),
                        "position": meta.get("position", "未知"),
                        "chunks": [c],
                        "salary": meta.get("salary", None),
                    })
            t2 = await registry.execute(ToolCall(
                name="global_rank",
                params={"resume_text": resume_text, "candidate_jds": candidate_jds[:10], "top_k": slots.get("top_k", settings.MATCH_TOP_K)},
                task_id=1,
            ))
            tool_results.append(t2)
            if t2.success:
                tool_summary.append({"tool": "global_rank", "rankings": len(t2.data.get("rankings", []))})

    elif intent.value == "assess":
        user_jd = getattr(session, "user_provided_jd", None)
        if user_jd:
            t1 = await registry.execute(ToolCall(
                name="match_analyze",
                params={
                    "resume_text": resume_text,
                    "jd_source": slots.get("jd_source", "text"),
                    "jd_data": {"full_text": user_jd},
                    "company": slots.get("company"),
                },
                task_id=0,
            ))
            tool_results.append(t1)
            tool_summary.append({"tool": "match_analyze", "source": "user_jd"})
        else:
            company = slots.get("company")
            t1 = await registry.execute(ToolCall(
                name="kb_retrieve",
                params={
                    "query": slots.get("search_keywords") or rewrite_result.search_keywords or rewrite_result.rewritten_query,
                    "company": company,
                    "top_k": settings.ASSESS_TOP_K,
                },
                task_id=0,
            ))
            tool_results.append(t1)
            if t1.success and t1.data.get("chunks"):
                tool_summary.append({"tool": "kb_retrieve", "chunks": len(t1.data.get("chunks", []))})
                t2 = await registry.execute(ToolCall(
                    name="match_analyze",
                    params={
                        "resume_text": resume_text,
                        "jd_source": slots.get("jd_source", "kb"),
                        "jd_data": {"chunks": t1.data.get("chunks", [])},
                        "company": company,
                        "attributes": slots.get("attributes", ["匹配度"]),
                    },
                    task_id=1,
                ))
                tool_results.append(t2)
                if t2.success:
                    tool_summary.append({"tool": "match_analyze", "score": t2.data.get("match_score")})

    elif intent.value == "verify":
        t1 = await registry.execute(ToolCall(
            name="kb_retrieve",
            params={"query": slots.get("search_keywords") or rewrite_result.search_keywords or rewrite_result.rewritten_query, "top_k": settings.VERIFY_TOP_K},
            task_id=0,
        ))
        tool_results.append(t1)
        if t1.success and t1.data.get("chunks"):
            tool_summary.append({"tool": "kb_retrieve", "chunks": len(t1.data.get("chunks", []))})
            t2 = await registry.execute(ToolCall(
                name="qa_synthesize",
                params={
                    "question": request.message,
                    "rewritten_question": rewrite_result.rewritten_query,
                    "evidence_chunks": t1.data.get("chunks", []),
                    "qa_type": slots.get("qa_type", "factual"),
                    "attributes": slots.get("attributes", []),
                },
                task_id=1,
            ))
            tool_results.append(t2)
            if t2.success:
                tool_summary.append({"tool": "qa_synthesize", "confidence": t2.data.get("confidence")})

    elif intent.value == "prepare":
        t1 = await registry.execute(ToolCall(
            name="interview_gen",
            params={
                "match_result": {
                    "gaps": slots.get("gaps", ["待补充"]),
                    "interview_focus": ["技术深度", "项目经验"],
                    "jd_summary": rewrite_result.rewritten_query,
                    "company": slots.get("company"),
                    "position": slots.get("position"),
                },
                "count": slots.get("count", 5),
                "difficulty": slots.get("difficulty", "mixed"),
                "focus_area": slots.get("focus_area", "gap"),
            },
            task_id=0,
        ))
        tool_results.append(t1)
        if t1.success:
            tool_summary.append({"tool": "interview_gen", "questions": len(t1.data.get("questions", []))})

    elif intent.value == "manage":
        operation = slots.get("operation", "list_jds")
        if operation == "upload_resume":
            # 简历上传已迁移至「我的简历」页面，对话内不再处理
            from app.core.tool_registry import ToolResult
            t1 = ToolResult(
                success=False,
                error="简历上传请前往「我的简历」页面完成，支持 PDF / DOCX / TXT 格式。",
                data={"redirect_tip": "请切换到底部「我的简历」标签上传"},
            )
            tool_results.append(t1)
            tool_summary.append({"tool": "file_ops", "operation": "upload_resume", "status": "引导至简历页面"})
        else:
            t1 = await registry.execute(ToolCall(
                name="file_ops",
                params={
                    "operation": operation,
                    "file_data": slots.get("file_data"),
                    "text_data": slots.get("text_data"),
                    "target_id": slots.get("target_id"),
                },
                task_id=0,
            ))
            tool_results.append(t1)
            tool_summary.append({"tool": "file_ops", "operation": operation})

    elif intent.value == "chat":
        t1 = await registry.execute(ToolCall(
            name="general_chat",
            params={"user_message": request.message, "chat_type": slots.get("general_type", "other"), "topic_hint": slots.get("topic_hint")},
            task_id=0,
        ))
        tool_results.append(t1)
        if t1.success:
            tool_summary.append({"tool": "general_chat", "response_preview": t1.data.get("response", "")[:50]})

    return tool_results, tool_summary


def _format_tool_evidence(tool_results: list) -> str:
    """将工具结果格式化为文本证据"""
    tool_evidence = ""
    for i, tr in enumerate(tool_results):
        if not tr.success or not tr.data:
            continue
        if tr.data.get("chunks"):
            tool_evidence += f"\n【检索结果 {i+1}】\n"
            for c in tr.data.get("chunks", [])[:5]:
                tool_evidence += f"- {c.get('content', '')[:200]}...\n"
        if tr.data.get("rankings"):
            tool_evidence += f"\n【排序结果】\n"
            for r in tr.data.get("rankings", [])[:5]:
                tool_evidence += f"{r.get('rank')}. {r.get('company')} · {r.get('position')} (匹配度{r.get('match_score')})\n"
        if tr.data.get("answer"):
            tool_evidence += f"\n【问答结果】\n{tr.data.get('answer')}\n"
        if tr.data.get("questions"):
            tool_evidence += f"\n【面试题】\n"
            for q in tr.data.get("questions", [])[:5]:
                tool_evidence += f"{q.get('id')}. [{q.get('category')}] {q.get('question')}\n"
        if tr.data.get("response"):
            tool_evidence += f"\n【对话回复】\n{tr.data.get('response')}\n"
    return tool_evidence


def _build_history_cache(session: SessionMemory) -> List[HistoryCacheEntry]:
    """
    从历史对话轮次中提取可复用的执行缓存。

    遍历最近的工作记忆轮次，提取上轮成功执行的 tool_call 任务，
    构建 HistoryCacheEntry 列表供 planner 做 Replan 复用判断。
    """
    history_cache: List[HistoryCacheEntry] = []
    recent_turns = list(reversed(session.working_memory.turns[-3:]))
    for turn in recent_turns:
        snapshot = turn.task_graph_snapshot
        if not snapshot or not isinstance(snapshot, dict):
            continue
        tasks = snapshot.get("tasks", [])
        for t in tasks:
            if not isinstance(t, dict):
                continue
            if t.get("task_type") == "tool_call" and t.get("status") == "success":
                entry = HistoryCacheEntry(
                    task_id=t.get("task_id", ""),
                    tool_name=t.get("tool_name"),
                    params_hash=t.get("params_hash", ""),
                    status="success",
                    output=t.get("result"),
                    timestamp=turn.timestamp,
                )
                history_cache.append(entry)
                logger.info(
                    f"[HistoryCache] 提取缓存 | turn={turn.turn_id} | "
                    f"task={entry.task_id}({entry.tool_name}) | hash={entry.params_hash[:8]}..."
                )
    logger.info(f"[HistoryCache] 共提取 {len(history_cache)} 条历史缓存")
    return history_cache


# ──────────────────────────── 规则路线（完整链路，独立组件） ────────────────────────────

async def _handle_rule_route(
    request: ChatRequest,
    session: SessionMemory,
    session_id: str,
    message_with_ocr: str,
    rewrite_result: QueryRewriteResult,
    resume_text: str,
) -> dict:
    """规则路线处理入口（完整链路：意图识别 → Plan → 执行 → LLM 回复）"""
    intent_router = _create_intent_router()
    history_context = session.working_memory.get_recent_context(2) if session.working_memory.turns else ""
    route_meta = await intent_router.route(
        message=rewrite_result.rewritten_query,
        attachments=request.attachments or [],
        context=request.context,
        history_context=history_context,
    )

    agent = EnhancedAgentOrchestrator()
    agent_ctx, session = await agent.run(
        intent_result=route_meta,
        message=message_with_ocr,
        rewritten_query=rewrite_result.rewritten_query,
        search_keywords=rewrite_result.search_keywords,
        is_follow_up=rewrite_result.is_follow_up,
        follow_up_type=rewrite_result.follow_up_type,
        attachments=request.attachments or [],
        resume_text=resume_text,
        session=session,
    )

    # 回复生成
    has_resume = bool(resume_text) and "尚未上传" not in resume_text
    if route_meta.intent in (IntentType.MATCH_SINGLE, IntentType.GLOBAL_MATCH) and not has_resume:
        reply = ChatReply(
            text="📎 要进行岗位匹配分析，我需要先了解你的简历背景。\n\n请上传你的简历（支持 PDF/图片/文本），我会帮你分析：\n1. 与目标岗位的匹配度\n2. 核心优势和潜在短板\n3. 面试准备建议",
        )
    else:
        try:
            llm = LLMClient.from_config("chat")
            reply_text = await llm.generate(
                prompt=agent_ctx.user_prompt,
                system=agent_ctx.system_prompt,
                temperature=0.7,
                max_tokens=1500,
                timeout=TIMEOUT_HEAVY,  # 30s，最终聚合
            )
            reply = ChatReply(text=reply_text)
        except Exception as e:
            handler = RULE_HANDLERS.get(route_meta.intent, handle_general)
            reply = await handler(request, route_meta)

    # 回填记忆
    if session.working_memory.turns:
        session.working_memory.turns[-1].assistant_reply = reply.text

    # 构造 Agent 执行摘要
    tool_summary = []
    for tool_call, tool_result in zip(agent_ctx.selected_tools, agent_ctx.tool_results):
        status = "✅" if tool_result.success else "❌"
        tool_summary.append({
            "tool": tool_call.name,
            "status": status,
            "params": tool_call.parameters,
            "result_preview": str(tool_result.data)[:200] if tool_result.data else str(tool_result.error)[:200],
        })

    memory_state = _build_memory_state(session)
    return {
        "session_id": session_id,
        "intent": route_meta.intent.value,
        "route_meta": route_meta.model_dump(),
        "agent_mode": "rule",
        "agent": {
            "rewritten_query": agent_ctx.rewritten_query,
            "tools": tool_summary,
            "kb_chunks_count": len(agent_ctx.kb_chunks),
            "system_prompt_preview": agent_ctx.system_prompt[:200] if hasattr(agent_ctx, "system_prompt") else "",
        },
        "memory": memory_state,
        "reply": reply.model_dump(),
    }


def _build_memory_state(session: SessionMemory) -> dict:
    """构造记忆状态摘要（两条路线共用）"""
    state = {
        "working_turns": len(session.working_memory.turns),
        "compressed_blocks": len(session.compressed_memories),
        "current_topic": session.current_topic,
        "evidence_cache_size": len(session.evidence_cache),
    }
    if session.long_term:
        state["long_term"] = {
            "entities": session.long_term.entities,
            "preferences": session.long_term.preferences,
        }
    return state


# ═══════════════════════════════════════════════════════
# LLM 路线 v2 —— 新架构
# ═══════════════════════════════════════════════════════

async def _handle_llm_route_v2(
    request: ChatRequest,
    session: SessionMemory,
    session_id: str,
    message_with_ocr: str,
    rewrite_result: QueryRewriteResult,
    resume_text: str,
    tracker: Any = None,
    request_start_time: float = 0.0,
) -> dict:
    """
    LLM 路线 v2（新架构）。

    链路：Query改写（已完成）→ 三层级联意图识别 → Plan动态规划 → ReAct执行 → LLM聚合回复
    """
    import time

    # ── 辅助：turn_completed 埋点 ──
    def _emit_turn_completed(reply_text: str = "", is_clarification: bool = False, has_error: bool = False, graph=None):
        if not tracker:
            return
        total_tools = sum(
            1 for t in graph.tasks.values() if t.tool_name and t.status == "success"
        ) if graph else 0
        tracker.track("turn_completed", {
            "session_id": session_id,
            "turn_id": len(session.working_memory.turns),
            "intent": intent_result.demands[0].intent_type if intent_result.demands else "chat",
            "is_clarification": is_clarification,
            "has_error": has_error,
            "reply_length": len(reply_text),
            "total_latency_ms": int((time.time() - request_start_time) * 1000) if request_start_time else 0,
            "total_tools_called": total_tools,
            "completed": not is_clarification and not has_error and len(reply_text) > 50,
            "relevance_score": (request.eval_context or {}).get("relevance_score", 0),
        })

    # ── ① 三层级联意图识别 ──
    intent_recognizer = IntentRecognizer()
    intent_result: NewIntentResult = await intent_recognizer.recognize(
        rewrite_result=rewrite_result,
        session=session,
    )
    logger.info(
        f"[Chat-v2] 意图识别 | demands={[d.intent_type for d in intent_result.demands]} | "
        f"complete={intent_result.is_complete} | clarify={intent_result.needs_clarification}"
    )

    # 埋点：intent_classified
    eval_ctx = request.eval_context or {}
    if tracker:
        tracker.track("intent_classified", {
            "session_id": session_id,
            "turn_id": len(session.working_memory.turns) + 1,
            "is_follow_up": rewrite_result.is_follow_up,
            "follow_up_type": rewrite_result.follow_up_type,
            "predicted_intents": [d.intent_type for d in intent_result.demands],
            "primary_intent": intent_result.demands[0].intent_type if intent_result.demands else "chat",
            "needs_clarification": intent_result.needs_clarification,
            "resolved_entities": intent_result.resolved_entities,
            "intent_skipped_due_to_timeout": getattr(intent_result, "skipped_due_to_timeout", False),
            "gold_intents": eval_ctx.get("gold_intents", []),
            "gold_slots": eval_ctx.get("gold_slots", {}),
        })

    # ── ② 澄清场景 ──
    if intent_result.needs_clarification:
        reply_text = intent_result.clarification_question or "抱歉，我没有完全理解您的意思，能再详细说明一下吗？"
        reply = ChatReply(text=reply_text)
        turn = DialogueTurn(
            turn_id=len(session.working_memory.turns) + 1,
            user_message=request.message,
            assistant_reply=reply.text,
            intent=intent_result.demands[0].intent_type if intent_result.demands else "chat",
            rewritten_query=rewrite_result.rewritten_query,
            evidence_score=0.0,
        )
        session.working_memory.append(turn)
        try:
            mm = MemoryManager(llm_client=LLMClient.from_config("memory"))
            await mm.rotate_memory(session, turn)
        except Exception as e:
            logger.warning(f"[Chat-v2] 记忆轮转失败: {e}")

        memory_state = _build_memory_state(session)
        _emit_turn_completed(reply_text=reply_text, is_clarification=True)
        return {
            "session_id": session_id,
            "intent": intent_result.demands[0].intent_type if intent_result.demands else "chat",
            "route_meta": {
                "layer": "clarification",
                "intent": intent_result.demands[0].intent_type if intent_result.demands else "chat",
                "confidence": 0.0,
                "missing_entities": intent_result.missing_entities,
                "reason": intent_result.clarification_question,
            },
            "agent_mode": "llm",
            "is_clarification": True,
            "llm_agent": {
                "rewritten_query": rewrite_result.rewritten_query,
                "search_keywords": rewrite_result.search_keywords,
                "is_follow_up": rewrite_result.is_follow_up,
                "follow_up_type": rewrite_result.follow_up_type,
                "clarification_question": intent_result.clarification_question,
                "global_slots": intent_result.resolved_entities,
            },
            "memory": memory_state,
            "reply": reply.model_dump(),
        }

    # ── ③ Plan模块：动态生成任务图 ──
    # 构建 evidence_cache_summary
    evidence_cache_summary = ""
    if session.evidence_cache and settings.EVIDENCE_CACHE_ENABLED:
        top_chunks = session.evidence_cache[:settings.EVIDENCE_CACHE_MAX_SIZE]
        summary_lines = []
        for c in top_chunks:
            title = c.get("metadata", {}).get("position", "") or c.get("title", "") or "未知岗位"
            content = c.get("content", "")[:100]
            summary_lines.append(f"- {title}: {content}")
        evidence_cache_summary = "\n".join(summary_lines)

    planner = TaskPlanner()
    graph = await planner.create_graph(
        rewritten_query=rewrite_result.rewritten_query,
        demands=[{"intent_type": d.intent_type, "entities": d.entities, "priority": d.priority} for d in intent_result.demands],
        resolved_entities=intent_result.resolved_entities,
        resume_text=resume_text,
        search_keywords=rewrite_result.search_keywords,
        follow_up_type=rewrite_result.follow_up_type,
        evidence_cache_summary=evidence_cache_summary,
    )
    logger.info(
        f"[Chat-v2] Plan完成 | tasks={len(graph.tasks)} | "
        f"groups={len(graph.compute_parallel_groups())}"
    )

    # 埋点：plan_generated
    errors = graph.validate()
    if tracker:
        tracker.track("plan_generated", {
            "session_id": session_id,
            "turn_id": len(session.working_memory.turns) + 1,
            "task_count": len(graph.tasks),
            "errors": errors,
            "passed": len(errors) == 0,
            "has_circular_dep": any("循环依赖" in str(e) for e in errors),
            "has_missing_dep": any("依赖不存在" in str(e) for e in errors),
        })

    # ── ④ ReAct执行器：执行任务图 ──
    executor = ReActExecutor()
    graph = await executor.execute(graph, session)
    logger.info(
        f"[Chat-v2] ReAct执行完成 | status={graph.global_status} | "
        f"tasks={[t.task_id + ':' + t.status for t in graph.tasks.values()]}"
    )

    # Update evidence cache after execution
    for task in graph.tasks.values():
        if task.tool_name == "kb_retrieve" and task.status == "success" and task.result:
            chunks = task.result.get("chunks", []) if isinstance(task.result, dict) else []
            if chunks:
                session.evidence_cache = chunks[:settings.EVIDENCE_CACHE_MAX_SIZE]
                session.evidence_cache_query = rewrite_result.rewritten_query
                break

    # 埋点：task_graph_executed
    if tracker:
        tracker.track("task_graph_executed", {
            "session_id": session_id,
            "turn_id": len(session.working_memory.turns) + 1,
            "total_tasks": len(graph.tasks),
            "success_count": sum(1 for t in graph.tasks.values() if t.status == "success"),
            "failed_count": sum(1 for t in graph.tasks.values() if t.status == "failed"),
            "skipped_count": sum(1 for t in graph.tasks.values() if t.status == "skipped"),
            "aborted_count": sum(1 for t in graph.tasks.values() if t.status == "aborted"),
            "global_status": graph.global_status,
            "executed_intents": [d.intent_type for d in intent_result.demands],
        })

    # ── ⑤ LLM聚合：生成最终回复 ──
    # 收集所有成功任务的输出
    tool_outputs = []
    for task in graph.tasks.values():
        if task.status == "success" and task.result:
            tool_outputs.append({
                "task_id": task.task_id,
                "tool": task.tool_name,
                "description": task.description,
                "result": task.result,
            })

    system_prompt = (
        "你是一位专业的求职顾问。请基于以下工具执行结果，给用户一个清晰、有帮助的回复。\n"
        "要求：\n"
        "1. 直接回答用户的问题，不要重复用户的原话\n"
        "2. 如果有匹配分析结果，给出具体的分数和建议\n"
        "3. 如果有检索结果，基于事实回答，不要编造\n"
        "4. 语气友好、专业、结构化"
    )

    user_prompt = (
        f"【用户问题】\n{rewrite_result.rewritten_query}\n\n"
        f"【工具执行结果】\n{json.dumps(tool_outputs, ensure_ascii=False, default=str)[:3000]}\n\n"
        f"请生成回复："
    )

    # 最终聚合：chat → core → planner → memory 模型降级
    # 超时递减：主模型给 30s，fallback 模型各给 15s（服务已不稳定，快速尝试）
    reply_text = ""
    fallback_configs = [
        ("chat", TIMEOUT_HEAVY),      # 30s
        ("core", TIMEOUT_STANDARD),   # 20s
        ("planner", TIMEOUT_LIGHT),   # 10s
        ("memory", TIMEOUT_LIGHT),    # 10s
    ]
    last_error = None
    for model_name, model_timeout in fallback_configs:
        try:
            llm = LLMClient.from_config(model_name)
            reply_text = await llm.generate(
                prompt=user_prompt,
                system=system_prompt,
                temperature=0.5,
                max_tokens=1500,
                timeout=model_timeout,
            )
            if model_name != "chat":
                logger.info(f"[Chat-v2] 最终聚合使用降级模型: {model_name}")
            break
        except Exception as e:
            last_error = e
            logger.warning(f"[Chat-v2] 聚合模型 {model_name} 失败: {e}")
    else:
        logger.error(f"[Chat-v2] 所有聚合模型均失败: {last_error}")
        reply_text = "抱歉，我在处理您的请求时遇到了问题，请稍后重试。"

    reply = ChatReply(text=reply_text.strip())

    # ── ⑥ 保存对话历史 ──
    turn = DialogueTurn(
        turn_id=len(session.working_memory.turns) + 1,
        user_message=request.message,
        assistant_reply=reply.text,
        intent=intent_result.demands[0].intent_type if intent_result.demands else "chat",
        rewritten_query=rewrite_result.rewritten_query,
        evidence_score=intent_result.demands[0].confidence if intent_result.demands else 0.0,
    )
    session.working_memory.append(turn)
    try:
        mm = MemoryManager(llm_client=LLMClient.from_config("memory"))
        await mm.rotate_memory(session, turn)
    except Exception as e:
        logger.warning(f"[Chat-v2] 记忆轮转失败: {e}")

    # ── ⑦ 构造返回 ──
    memory_state = _build_memory_state(session)
    _emit_turn_completed(reply_text=reply.text, has_error=(graph.global_status == "failed"), graph=graph)

    return {
        "session_id": session_id,
        "intent": intent_result.demands[0].intent_type if intent_result.demands else "chat",
        "route_meta": {
            "layer": "llm",
            "intent": intent_result.demands[0].intent_type if intent_result.demands else "chat",
            "confidence": intent_result.demands[0].confidence if intent_result.demands else 0.0,
            "demands": [{"intent": d.intent_type, "entities": d.entities} for d in intent_result.demands],
            "plan_tasks": len(graph.tasks),
            "plan_errors": errors,
        },
        "agent_mode": "llm",
        "is_clarification": False,
        "llm_agent": {
            "rewritten_query": rewrite_result.rewritten_query,
            "search_keywords": rewrite_result.search_keywords,
            "is_follow_up": rewrite_result.is_follow_up,
            "follow_up_type": rewrite_result.follow_up_type,
            "global_slots": intent_result.resolved_entities,
            "tool_outputs": [{"tool": o["tool"], "task_id": o["task_id"]} for o in tool_outputs],
        },
        "memory": memory_state,
        "reply": reply.model_dump(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SSE 流式输出（v2 路线）
# ═══════════════════════════════════════════════════════════════════════════════

def _sse(event_type: str, data: dict) -> str:
    """构造 SSE 事件行"""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _handle_llm_route_v2_stream(
    request: ChatRequest,
    session: SessionMemory,
    session_id: str,
    message_with_ocr: str,
    rewrite_result: QueryRewriteResult,
    resume_text: str,
):
    """
    v2 路线的 SSE 流式处理生成器。
    在关键步骤推送进度事件，最终聚合使用 LLM 流式生成。
    """
    import time

    # ── ① 意图识别 ──
    yield _sse("status", {"step": "intent", "message": "正在识别您的意图..."})
    intent_recognizer = IntentRecognizer()
    intent_result = await intent_recognizer.recognize(
        rewrite_result=rewrite_result,
        session=session,
    )

    # 澄清场景
    if intent_result.needs_clarification:
        yield _sse("clarification", {"question": intent_result.clarification_question})
        yield _sse("done", {
            "session_id": session_id,
            "intent": intent_result.demands[0].intent_type if intent_result.demands else "chat",
            "is_clarification": True,
            "reply": {"text": intent_result.clarification_question},
        })
        return

    # ── ② Plan ──
    yield _sse("status", {"step": "plan", "message": "正在规划任务..."})
    evidence_cache_summary = ""
    if session.evidence_cache and settings.EVIDENCE_CACHE_ENABLED:
        top_chunks = session.evidence_cache[:settings.EVIDENCE_CACHE_MAX_SIZE]
        summary_lines = []
        for c in top_chunks:
            title = c.get("metadata", {}).get("position", "") or c.get("title", "") or "未知岗位"
            content = c.get("content", "")[:100]
            summary_lines.append(f"- {title}: {content}")
        evidence_cache_summary = "\n".join(summary_lines)

    planner = TaskPlanner()
    graph = await planner.create_graph(
        rewritten_query=rewrite_result.rewritten_query,
        demands=[{"intent_type": d.intent_type, "entities": d.entities, "priority": d.priority}
                 for d in intent_result.demands],
        resolved_entities=intent_result.resolved_entities,
        resume_text=resume_text,
        search_keywords=rewrite_result.search_keywords,
        follow_up_type=rewrite_result.follow_up_type,
        evidence_cache_summary=evidence_cache_summary,
    )

    # ── ③ ReAct 执行 ──
    yield _sse("status", {"step": "execute", "message": "正在执行分析..."})
    executor = ReActExecutor()
    graph = await executor.execute(graph, session)

    # Update evidence cache
    for task in graph.tasks.values():
        if task.tool_name == "kb_retrieve" and task.status == "success" and task.result:
            chunks = task.result.get("chunks", []) if isinstance(task.result, dict) else []
            if chunks:
                session.evidence_cache = chunks[:settings.EVIDENCE_CACHE_MAX_SIZE]
                session.evidence_cache_query = rewrite_result.rewritten_query
                break

    # ── ④ 收集工具输出 ──
    tool_outputs = []
    for task in graph.tasks.values():
        if task.status == "success" and task.result:
            tool_outputs.append({
                "task_id": task.task_id,
                "tool": task.tool_name,
                "description": task.description,
                "result": task.result,
            })

    system_prompt = (
        "你是一位专业的求职顾问。请基于以下工具执行结果，给用户一个清晰、有帮助的回复。\n"
        "要求：\n"
        "1. 直接回答用户的问题，不要重复用户的原话\n"
        "2. 如果有匹配分析结果，给出具体的分数和建议\n"
        "3. 如果有检索结果，基于事实回答，不要编造\n"
        "4. 语气友好、专业、结构化"
    )
    user_prompt = (
        f"【用户问题】\n{rewrite_result.rewritten_query}\n\n"
        f"【工具执行结果】\n{json.dumps(tool_outputs, ensure_ascii=False, default=str)[:3000]}\n\n"
        f"请生成回复："
    )

    # ── ⑤ 流式聚合 ──
    yield _sse("status", {"step": "aggregate", "message": "正在生成回复..."})

    reply_text = ""
    fallback_configs = [
        ("chat", TIMEOUT_HEAVY),
        ("core", TIMEOUT_STANDARD),
        ("planner", TIMEOUT_LIGHT),
        ("memory", TIMEOUT_LIGHT),
    ]
    for model_name, model_timeout in fallback_configs:
        try:
            llm = LLMClient.from_config(model_name)
            async for token in llm.generate_stream(
                prompt=user_prompt,
                system=system_prompt,
                temperature=0.5,
                max_tokens=1500,
                timeout=model_timeout,
            ):
                reply_text += token
                yield _sse("delta", {"text": token})
            if model_name != "chat":
                logger.info(f"[Chat-v2-stream] 聚合使用降级模型: {model_name}")
            break
        except Exception as e:
            logger.warning(f"[Chat-v2-stream] 聚合模型 {model_name} 失败: {e}")
    else:
        logger.error("[Chat-v2-stream] 所有聚合模型均失败")
        fail_text = "抱歉，我在处理您的请求时遇到了问题，请稍后重试。"
        reply_text = fail_text
        yield _sse("delta", {"text": fail_text})

    # ── ⑥ 保存对话历史 ──
    reply = ChatReply(text=reply_text.strip())
    turn = DialogueTurn(
        turn_id=len(session.working_memory.turns) + 1,
        user_message=request.message,
        assistant_reply=reply.text,
        intent=intent_result.demands[0].intent_type if intent_result.demands else "chat",
        rewritten_query=rewrite_result.rewritten_query,
        evidence_score=intent_result.demands[0].confidence if intent_result.demands else 0.0,
    )
    session.working_memory.append(turn)
    try:
        mm = MemoryManager(llm_client=LLMClient.from_config("memory"))
        await mm.rotate_memory(session, turn)
    except Exception as e:
        logger.warning(f"[Chat-v2-stream] 记忆轮转失败: {e}")

    memory_state = _build_memory_state(session)
    yield _sse("done", {
        "session_id": session_id,
        "intent": intent_result.demands[0].intent_type if intent_result.demands else "chat",
        "route_meta": {
            "layer": "llm",
            "intent": intent_result.demands[0].intent_type if intent_result.demands else "chat",
            "confidence": intent_result.demands[0].confidence if intent_result.demands else 0.0,
            "demands": [{"intent": d.intent_type, "entities": d.entities} for d in intent_result.demands],
            "plan_tasks": len(graph.tasks),
        },
        "agent_mode": "llm",
        "is_clarification": False,
        "memory": memory_state,
        "reply": reply.model_dump(),
    })


@api_router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    v2 路线 SSE 流式对话入口。
    返回 text/event-stream，事件类型：status / delta / clarification / done
    """
    session_id, session = _get_or_create_session(request.session_id, request.user_id)
    resume_text = _get_resume_text()

    # 处理图片附件
    message_with_ocr = request.message
    ocr_results = []
    if request.attachments:
        for att in request.attachments:
            if att.content_type and att.content_type.startswith("image/") and att.data:
                try:
                    import base64
                    image_bytes = base64.b64decode(att.data)
                    ocr_service = OCRService()
                    ocr_result = await ocr_service.extract(image_bytes, filename=att.filename)
                    ocr_results.append(ocr_result)
                except Exception as e:
                    logger.error(f"[ChatStream] OCR 失败: {e}")
    if ocr_results:
        ocr_text = "\n\n".join(f"【图片提取】\n{r.text}" for r in ocr_results)
        message_with_ocr = f"{request.message}\n\n{ocr_text}"
        request.user_provided_jd = "\n\n".join(r.text for r in ocr_results)
        session.user_provided_jd = request.user_provided_jd

    # Query 改写
    rewriter = QueryRewriter()
    rewrite_result = await rewriter.rewrite(raw_query=message_with_ocr, session=session)

    agent_mode = settings.AGENT_MODE
    if agent_mode == "auto":
        effective_mode = settings.DEFAULT_AGENT_MODE
    else:
        effective_mode = agent_mode
    use_llm_agent = effective_mode == "llm"

    async def event_generator():
        if not use_llm_agent:
            # 规则路线暂不支持流式，直接返回结果
            result = await _handle_rule_route(
                request=request, session=session, session_id=session_id,
                message_with_ocr=message_with_ocr, rewrite_result=rewrite_result,
                resume_text=resume_text,
            )
            yield _sse("done", result)
            return

        # v2 流式路线
        async for event in _handle_llm_route_v2_stream(
            request=request,
            session=session,
            session_id=session_id,
            message_with_ocr=message_with_ocr,
            rewrite_result=rewrite_result,
            resume_text=resume_text,
        ):
            yield event

    return StreamingResponse(event_generator(), media_type="text/event-stream")
