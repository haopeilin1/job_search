import json
import logging
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from typing import Optional, List, Any
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from app.core.llm_client import LLMClient, TIMEOUT_HEAVY, TIMEOUT_STANDARD, TIMEOUT_LIGHT
from app.core.memory import SessionMemory, MemoryManager, DialogueTurn, PendingClarification
from app.core.db import load_session_meta, load_long_term_memory, delete_session_meta, delete_long_term_memory
from app.core.config import settings
from app.core.state import active_resume_id, resumes_db
from app.core.query_rewrite import QueryRewriter, QueryRewriteResult
from app.core.intent_recognition import IntentRecognizer, IntentResult as NewIntentResult
from app.core.planner import TaskPlanner
from app.core.react_executor import ReActExecutor

# 新体系（意图识别 + 规划）
from app.core.llm_intent import LLMIntentRouter
from app.core.llm_planner import TaskGraphPlanner
from app.core.new_arch_adapter import (
    multi_intent_result_to_intent_result,
    convert_task_graph,
)
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
    # 评测模式：若请求重置 session，先删除已有 session（内存 + 数据库）
    eval_ctx = request.eval_context or {}
    if eval_ctx.get("reset_session") and request.session_id:
        if request.session_id in _session_store:
            del _session_store[request.session_id]
        delete_session_meta(request.session_id)
        delete_long_term_memory(request.session_id)
        logger.info(f"[Chat] 评测模式重置 session | session_id={request.session_id}")

    # 1. 会话管理
    # 评测模式：若请求重置 session，先删除已有 session（内存 + 数据库）
    eval_ctx = request.eval_context or {}
    if eval_ctx.get("reset_session") and request.session_id:
        if request.session_id in _session_store:
            del _session_store[request.session_id]
        delete_session_meta(request.session_id)
        delete_long_term_memory(request.session_id)
        logger.info(f"[ChatStream] 评测模式重置 session | session_id={request.session_id}")

    session_id, session = _get_or_create_session(request.session_id, request.user_id)

    # 评测模式：支持 eval_context 直接指定 resume_id，避免依赖全局 active_resume_id
    eval_resume_id = eval_ctx.get("resume_id")
    if eval_resume_id and eval_resume_id in resumes_db:
        import app.core.state as _state_module
        _state_module.active_resume_id = eval_resume_id
        logger.info(f"[Chat] 评测模式切换简历 | resume_id={eval_resume_id}")

    resume_text = _get_resume_text()

    # 同步简历可用状态到 session.global_slots，供意图识别层 _check_clarification_need 使用
    if not hasattr(session, "global_slots"):
        session.global_slots = {}
    has_resume = resume_text != "（用户尚未上传简历）"
    session.global_slots["resume_available"] = has_resume
    if has_resume:
        session.global_slots["resume_text"] = resume_text
    logger.info(f"[Chat] 简历状态同步 | active_resume_id={active_resume_id} | has_resume={has_resume} | session.global_slots={session.global_slots}")

    # 评测模式：注入模拟多轮上下文
    injected_slots = eval_ctx.get("injected_history_slots")
    if injected_slots and isinstance(injected_slots, dict):
        if session.long_term is None:
            from app.core.memory import LongTermMemory
            session.long_term = LongTermMemory()
        session.long_term.entities.update(injected_slots)
    injected_cache = eval_ctx.get("injected_evidence_cache")
    if injected_cache and isinstance(injected_cache, list):
        session.evidence_cache = list(injected_cache)
        session.evidence_cache_query = eval_ctx.get("injected_evidence_query", "")

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
        f"| follow_up={rewrite_result.is_follow_up}/{rewrite_result.follow_up_type} "
        f"| source_pref={rewrite_result.source_preference}"
    )

    # 【关键】同步 QueryRewrite 结果到 session.global_slots，确保 route_multi 中的
    # _merge_global_slots 能获取到最新 query（解决多轮对话中上下文引用检测失效）
    if rewrite_result.search_keywords:
        session.global_slots["search_keywords"] = rewrite_result.search_keywords
    if rewrite_result.rewritten_query:
        session.global_slots["query"] = rewrite_result.rewritten_query
        session.global_slots["rewritten_query"] = rewrite_result.rewritten_query
    
    # Step 1: 意图识别（基于改写后的 query，更好地处理多轮对话）
    from app.core.llm_intent import LLMIntentRouter
    llm_for_intent = LLMClient.from_config("chat")
    intent_router = LLMIntentRouter(chat_llm=llm_for_intent)
    multi_result = await intent_router.route_multi(
        rewrite_result=rewrite_result,
        session=session,
        attachments=request.attachments or [],
    )
    logger.info(
        f"[Chat] 意图识别 | primary={multi_result.primary_intent.value if multi_result.primary_intent else 'None'} "
        f"| candidates={[c.intent_type.value for c in multi_result.candidates]}"
    )

    # ════════════════════════════════════════
    # 评测模式：仅意图识别，掐断后续流程
    # ════════════════════════════════════════
    if eval_ctx.get("intent_only"):
        # 将解析出的槽位同步到 session.global_slots，供多轮对话后续轮使用
        for c in multi_result.candidates:
            if c.slots.get("company"):
                session.global_slots["company"] = c.slots["company"]
            if c.slots.get("position"):
                session.global_slots["position"] = c.slots["position"]
        demands = []
        for c in multi_result.candidates:
            demands.append({
                "intent_type": c.intent_type.value,
                "confidence": c.confidence,
                "slots": c.slots,
            })
        return {
            "session_id": session_id,
            "intent": multi_result.primary_intent.value if multi_result.primary_intent else "none",
            "route_meta": {
                "demands": demands,
                "needs_clarification": multi_result.needs_clarification,
            },
            "is_clarification": multi_result.needs_clarification,
            "reply": "[intent_only_eval]",
        }

    if use_llm_agent:
        # ════════════════════════════════════════
        # LLM 路线 v2
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
            multi_result=multi_result,
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
    response = {
        "session_id": session_id,
        "intent": route_meta.intent.value,
        "agent_mode": "rule",
        "reply": reply.model_dump(),
    }
    # 旧路线暂无条件判断 eval_context，默认保留精简结构
    # 如需测试旧路线，可在此展开 agent/memory 等字段
    return response


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
    multi_result: "MultiIntentResult",
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

    # ── 话题切换检测 ──
    if session.working_memory.turns:
        try:
            mm = MemoryManager(llm_client=LLMClient.from_config("memory"))
            is_shift = await mm.detect_topic_shift(rewrite_result.rewritten_query or request.message, session)
            if is_shift:
                logger.info(f"[Chat-v2] 检测到话题切换，已清除 evidence_cache | session={session_id}")
        except Exception as e:
            logger.warning(f"[Chat-v2] 话题切换检测失败: {e}")

    # ── ① 新体系：多意图识别已在入口完成，multi_result 从外部传入 ──
    pc = session.pending_clarification
    current_turn_id = len(session.working_memory.turns) + 1

    # 澄清状态机：若上轮触发澄清且本轮是 clarify 型追问，覆盖 multi_result
    # 触发条件：有pending_clarification且输入很短（补充缺失槽位），或rewrite_result标记为clarify
    is_clarify_follow_up = (
        rewrite_result.follow_up_type == "clarify"
        or (
            pc
            and not pc.is_expired(current_turn_id, max_gap=2)
            and len(request.message) < 20
        )
    )
    if (
        pc
        and not pc.is_expired(current_turn_id, max_gap=2)
        and is_clarify_follow_up
    ):
        logger.info(
            f"[Chat-v2] 澄清状态恢复 | pending_intent={pc.pending_intent} | "
            f"resolved_refs={rewrite_result.resolved_references} | "
            f"rewritten={rewrite_result.rewritten_query}"
        )
        # 合并 QueryRewriter 解析出的槽位 + session 全局槽位继承
        merged_slots = dict(pc.resolved_slots)
        merged_slots.update(rewrite_result.resolved_references or {})
        # slot 继承：从 session.global_slots 继承上轮已解析的槽位
        if hasattr(session, "global_slots") and session.global_slots:
            for k, v in session.global_slots.items():
                if v is not None and k not in merged_slots and k not in ["resume_text", "resume_available", "search_keywords", "query", "rewritten_query"]:
                    merged_slots[k] = v
        # 如果 rewritten_query 中有更完整的信息，也尝试提取
        if rewrite_result.rewritten_query and rewrite_result.rewritten_query != request.message:
            merged_slots["search_keywords"] = rewrite_result.rewritten_query

        # 从用户原始输入中直接提取实体（公司名/岗位名），补充缺失槽位
        if pc.missing_slots:
            from app.core.llm_intent import _load_kb_entities
            kb_companies, kb_positions = _load_kb_entities()
            msg = request.message or ""
            # 按长度降序匹配，优先命中更长的实体（如 "字节跳动" 优先于 "字节"）
            if "company" in pc.missing_slots and "company" not in merged_slots:
                for c in sorted(kb_companies, key=len, reverse=True):
                    if c in msg:
                        merged_slots["company"] = c
                        break
            if "position" in pc.missing_slots and "position" not in merged_slots:
                for p in sorted(kb_positions, key=len, reverse=True):
                    if p in msg:
                        merged_slots["position"] = p
                        break

        from app.core.llm_intent import MultiIntentResult, IntentCandidate, LLMIntentType, _create_rule_registry
        from app.core.new_arch_adapter import map_intent_name_to_new
        new_intent_name = map_intent_name_to_new(pc.pending_intent)
        intent_type = LLMIntentType(new_intent_name) if new_intent_name else LLMIntentType.CHAT

        # 当 pending_intent=chat（意图模糊）时，基于原始 query 重新推断真实意图
        if intent_type == LLMIntentType.CHAT and request.message:
            if pc and not pc.is_expired(current_turn_id, max_gap=2):
                llm_for_intent = LLMClient.from_config("chat")
                intent_router = LLMIntentRouter(chat_llm=llm_for_intent)
                rewriter = QueryRewriter()
                re_rewrite = await rewriter.rewrite(raw_query=request.message, session=session)
                re_multi_result = await intent_router.route_multi(
                    rewrite_result=re_rewrite,
                    session=session,
                    attachments=request.attachments or [],
                )
                if re_multi_result.candidates:
                    intent_type = re_multi_result.primary_intent or re_multi_result.candidates[0].intent_type
                    for cand in re_multi_result.candidates:
                        merged_slots.update(cand.slots or {})
                    logger.info(f"[Chat-v2] 澄清后重新识别意图 | inferred={intent_type.value} | candidates={[c.intent_type.value for c in re_multi_result.candidates]}")
            else:
                # 无 pending_clarification 时：用规则匹配简单推断
                registry = _create_rule_registry()
                rule_matches = registry.classify_all(request.message, [])
                non_miss = [r for r in rule_matches if r.intent is not None]
                if non_miss:
                    best = non_miss[0]
                    intent_type = best.intent
                    merged_slots.update(best.metadata or {})
                    logger.info(f"[Chat-v2] 澄清状态推断意图 | inferred={intent_type.value} | rule={best.rule_name}")

        restored_candidate = IntentCandidate(
            intent_type=intent_type,
            confidence=0.85,
            reason=f"澄清状态恢复: 上轮缺失槽位 {pc.missing_slots}",
            slots=merged_slots,
            slot_sources={k: "clarification_recovery" for k in merged_slots.keys()},
            missing_slots=[s for s in pc.missing_slots if s not in merged_slots],
            needs_clarification=False,
            source="clarification_recovery",
            rule_agreement=True,
        )
        multi_result = MultiIntentResult(
            candidates=[restored_candidate],
            primary_intent=intent_type,
            needs_clarification=False,
            global_slots=merged_slots,
            execution_topology=[[intent_type]],
        )
        # 如果仍有缺失槽位，判断是否真正需要再次澄清
        # VERIFY: 有 company 或 position 之一即可执行；ASSESS: 必须 resume
        if restored_candidate.missing_slots:
            needs_clarify = True
            if intent_type == LLMIntentType.VERIFY:
                if merged_slots.get("company") or merged_slots.get("position"):
                    needs_clarify = False
            elif intent_type == LLMIntentType.ASSESS:
                if merged_slots.get("resume_available"):
                    needs_clarify = False
            if needs_clarify:
                multi_result.needs_clarification = True
                multi_result.clarification_reason = (
                    f"还需要补充以下信息：{', '.join(restored_candidate.missing_slots)}"
                )
    else:
        # 非 clarify 场景：直接使用传入的 multi_result（已在入口完成意图识别）
        if pc and pc.is_expired(current_turn_id, max_gap=2):
            logger.info(f"[Chat-v2] 澄清状态已过期，丢弃 | pending_intent={pc.pending_intent}")
            session.pending_clarification = None

    intent_result = multi_intent_result_to_intent_result(multi_result)
    logger.info(
        f"[Chat-v2] 新体系意图识别 | candidates={[c.intent_type.value for c in multi_result.candidates]} | "
        f"primary={multi_result.primary_intent.value if multi_result.primary_intent else 'None'} | "
        f"clarify={multi_result.needs_clarification}"
    )

    # 将意图识别解析的槽位同步到 session.global_slots，供 ReActExecutor 动态解析占位符使用
    if not hasattr(session, "global_slots"):
        session.global_slots = {}
    if multi_result.global_slots:
        for k, v in multi_result.global_slots.items():
            if v is not None:
                session.global_slots[k] = v
    # 补充 search_keywords 和 rewritten_query（Planner 占位符需要）
    if rewrite_result.search_keywords:
        session.global_slots["search_keywords"] = rewrite_result.search_keywords
    if rewrite_result.rewritten_query:
        session.global_slots["query"] = rewrite_result.rewritten_query
        session.global_slots["rewritten_query"] = rewrite_result.rewritten_query

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
        turn_id = len(session.working_memory.turns) + 1
        turn = DialogueTurn(
            turn_id=turn_id,
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

        # 保存澄清状态机
        primary_intent = intent_result.demands[0].intent_type if intent_result.demands else "chat"
        session.pending_clarification = PendingClarification(
            pending_intent=primary_intent,
            missing_slots=intent_result.missing_entities or [],
            clarification_question=reply_text,
            expected_slot_types=intent_result.missing_entities or [],
            created_turn_id=turn_id,
            resolved_slots=dict(intent_result.resolved_entities or {}),
        )
        logger.info(
            f"[Chat-v2] 澄清状态已保存 | intent={primary_intent} | "
            f"missing={intent_result.missing_entities} | turn_id={turn_id}"
        )

        memory_state = _build_memory_state(session)
        _emit_turn_completed(reply_text=reply_text, is_clarification=True)
        response = {
            "session_id": session_id,
            "intent": intent_result.demands[0].intent_type if intent_result.demands else "chat",
            "is_clarification": True,
            "reply": reply.model_dump(),
        }
        # 评测模式：附加详细 debug_info（澄清场景也需要）
        if request.eval_context:
            response.update({
                "route_meta": {
                    "layer": "clarification",
                    "intent": intent_result.demands[0].intent_type if intent_result.demands else "chat",
                    "confidence": 0.0,
                    "missing_entities": intent_result.missing_entities,
                    "reason": intent_result.clarification_question,
                },
                "agent_mode": "llm",
                "llm_agent": {
                    "rewritten_query": rewrite_result.rewritten_query,
                    "search_keywords": rewrite_result.search_keywords,
                    "is_follow_up": rewrite_result.is_follow_up,
                    "follow_up_type": rewrite_result.follow_up_type,
                    "clarification_question": intent_result.clarification_question,
                    "global_slots": intent_result.resolved_entities,
                },
                "memory": memory_state,
            })
            response["debug_info"] = {
                "intent": {
                    "demands": [
                        {"intent": d.intent_type, "entities": d.entities, "confidence": getattr(d, "confidence", 0.0), "source": getattr(d, "source", "unknown")}
                        for d in intent_result.demands
                    ],
                    "needs_clarification": intent_result.needs_clarification,
                    "clarification_question": intent_result.clarification_question,
                    "missing_entities": intent_result.missing_entities,
                    "resolved_entities": intent_result.resolved_entities,
                },
                "rewrite": {
                    "rewritten_query": rewrite_result.rewritten_query,
                    "search_keywords": rewrite_result.search_keywords,
                    "is_follow_up": rewrite_result.is_follow_up,
                    "follow_up_type": rewrite_result.follow_up_type,
                    "resolved_references": rewrite_result.resolved_references,
                },
                "session_history": [
                    {
                        "turn_id": t.turn_id,
                        "user_message": t.user_message,
                        "assistant_reply": t.assistant_reply,
                        "intent": t.intent,
                        "rewritten_query": t.rewritten_query,
                    }
                    for t in session.working_memory.turns
                ],
                "evidence_cache": session.evidence_cache,
                "evidence_cache_query": session.evidence_cache_query,
            }
        return response

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

    new_planner = TaskGraphPlanner()
    new_graph = await new_planner.create_graph(
        multi_result=multi_result,
        session=session,
        resume_text=resume_text,
        rewrite_result=rewrite_result,
        history_cache=[],
    )
    graph = convert_task_graph(new_graph)
    logger.info(
        f"[Chat-v2] 新体系Plan完成 | tasks={len(new_graph.tasks)} | "
        f"groups={len(new_graph.execution_strategy.parallel_groups)} | "
        f"converted_tasks={len(graph.tasks)}"
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
        if task.status == "success" and task.result:
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

    # ── ⑤ 检查 qa_synthesize 直接回复（跳过聚合）──
    qa_answer = None
    qa_task = None
    for task in graph.tasks.values():
        if task.tool_name == "qa_synthesize" and task.status == "success" and task.result:
            # qa_synthesize 的 result 直接是 data dict（不是嵌套在 data 键下）
            qa_answer = task.result.get("answer") if isinstance(task.result, dict) else None
            qa_task = task
            if qa_answer:
                logger.info(f"[Chat-v2] qa_synthesize 直接回复 | length={len(qa_answer)} | preview={qa_answer[:100]}")
                break

    if qa_answer:
        # qa_synthesize 已生成结构化回答，直接作为最终回复，跳过聚合 LLM
        reply_text = qa_answer
        reply = ChatReply(type="text", content=reply_text.strip())
        logger.info(f"[Chat-v2] 最终回复（qa_synthesize 直出）| length={len(reply_text)} | preview={reply_text[:100]}")
        # 初始化 response 构造需要的变量
        tool_outputs = []
        tool_summary = []
        system_prompt = ""
        if qa_task:
            tool_outputs.append({
                "task_id": qa_task.task_id,
                "tool": qa_task.tool_name,
                "description": qa_task.description,
                "result": qa_task.result,
            })
            result_preview = ""
            if qa_task.result:
                if isinstance(qa_task.result, dict):
                    result_preview = str(qa_task.result.get("data", qa_task.result))[:200]
                else:
                    result_preview = str(qa_task.result)[:200]
            tool_summary.append({
                "tool": qa_task.tool_name,
                "status": "✅",
                "params": qa_task.resolved_params if hasattr(qa_task, "resolved_params") else qa_task.parameters,
                "result_preview": result_preview,
            })
    else:
        # ── ⑥ LLM聚合：生成最终回复 ──
        # 收集所有成功任务的输出
        tool_outputs = []
        tool_summary = []
        for task in graph.tasks.values():
            if task.tool_name and task.status in ("success", "failed"):
                result_preview = ""
                if task.result:
                    if isinstance(task.result, dict):
                        result_preview = str(task.result.get("data", task.result))[:200]
                    else:
                        result_preview = str(task.result)[:200]
                elif task.observation:
                    result_preview = str(task.observation)[:200]
                tool_summary.append({
                    "tool": task.tool_name,
                    "status": "✅" if task.status == "success" else "❌",
                    "params": task.resolved_params if hasattr(task, "resolved_params") else task.parameters,
                    "result_preview": result_preview,
                })
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
            "1. 先直接回答用户的问题（是/否/有/没有），不要绕弯子，不要重复用户的原话\n"
            "2. 如果有匹配分析结果，给出具体的分数和建议\n"
            "3. 如果有检索结果，基于事实回答，不要编造\n"
            "4. 引用信息时必须标注来源，格式如：[来源：本地知识库-字节跳动] 或 [来源：外部搜索-新华网]\n"
            "5. 如果涉及'最近''最新'等时效性问题，请明确说明信息的时间范围\n"
            "6. 当本地知识库与外部搜索结果冲突时，请并列呈现两种说法并标注各自来源，不要替用户做判断\n"
            "7. 语气友好、专业、结构化，但不要过度展开与问题无关的建议\n"
            "8. 严格围绕用户当前问题回答，禁止发散到无关话题\n"
            "9. 控制回复长度：优先给出核心结论（2-3句话），细节只在用户明确要求时才展开\n"
            "10. 每个要点不超过2行，避免大段文字堆砌\n"
            "11. 【语气要求】你的语气应该像一位有5年经验的求职顾问在和朋友聊天：专业但有亲和力，自然不机械。"
            "避免模板化的开场白（如'根据您的问题，我将从以下几个方面进行分析'），适当使用口语化过渡。"
            "回复要有'人味'，让用户感觉是在和真人顾问对话，而不是在读说明书。"
        )

        # 限制 tool_outputs 长度，避免 prompt 过长导致 LLM 响应慢
        _truncated_outputs = []
        for to in tool_outputs:
            truncated = dict(to)
            if "result" in truncated and truncated["result"]:
                res = truncated["result"]
                if isinstance(res, dict):
                    # 对 result 中的长字段进行截断
                    for k in list(res.keys()):
                        v = res[k]
                        if isinstance(v, str) and len(v) > 500:
                            res[k] = v[:500] + "...[截断]"
                        elif isinstance(v, list) and len(v) > 10:
                            res[k] = v[:10] + [f"...共{len(v)}项，已截断"]
                truncated["result"] = res
            _truncated_outputs.append(truncated)

        tool_outputs_json = json.dumps(_truncated_outputs, ensure_ascii=False, default=str)[:1500]

        user_prompt = (
            f"【用户问题】\n{rewrite_result.rewritten_query}\n\n"
            f"【工具执行结果】\n{tool_outputs_json}\n\n"
            f"请严格围绕用户问题生成简洁回复（控制在300字以内）："
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
                    temperature=0.3,  # 降低温度，减少发散
                    max_tokens=800,   # 限制输出长度
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

        reply_text_stripped = reply_text.strip()
        logger.info(f"[Chat-v2] 最终聚合回复 | length={len(reply_text_stripped)} | preview={reply_text_stripped[:100]}")
        reply = ChatReply(type="text", content=reply_text_stripped)

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

    # 清理过期的澄清状态
    if session.pending_clarification:
        current_turn_id = len(session.working_memory.turns)
        if session.pending_clarification.is_expired(current_turn_id, max_gap=2):
            logger.info(
                f"[Chat-v2] 澄清状态过期清理 | intent={session.pending_clarification.pending_intent} | "
                f"created_turn={session.pending_clarification.created_turn_id} | current={current_turn_id}"
            )
            session.pending_clarification = None

    # ── ⑦ 构造返回 ──
    memory_state = _build_memory_state(session)
    _emit_turn_completed(reply_text=reply.text, has_error=(graph.global_status == "failed"), graph=graph)

    # 评测模式：附加详细 debug_info
    debug_info = None
    if request.eval_context:
        debug_info = {
            "task_graph": {
                "tasks": {
                    tid: {
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "tool_name": task.tool_name,
                        "status": task.status,
                        "parameters": task.parameters if hasattr(task, "parameters") else {},
                        "resolved_params": task.resolved_params if hasattr(task, "resolved_params") else {},
                        "result": task.result,
                        "observation": task.observation,
                    }
                    for tid, task in graph.tasks.items()
                },
                "global_status": graph.global_status,
                "replan_reason": graph.replan_reason,
                "replan_count": getattr(graph, "replan_count", 0),
            },
            "intent": {
                "demands": [
                    {"intent": d.intent_type, "entities": d.entities, "confidence": getattr(d, "confidence", 0.0), "source": getattr(d, "source", "unknown")}
                    for d in intent_result.demands
                ],
                "needs_clarification": intent_result.needs_clarification,
                "clarification_question": intent_result.clarification_question,
                "missing_entities": intent_result.missing_entities,
                "resolved_entities": intent_result.resolved_entities,
            },
            "rewrite": {
                "rewritten_query": rewrite_result.rewritten_query,
                "search_keywords": rewrite_result.search_keywords,
                "is_follow_up": rewrite_result.is_follow_up,
                "follow_up_type": rewrite_result.follow_up_type,
                "resolved_references": rewrite_result.resolved_references,
            },
            "evidence_cache": session.evidence_cache,
            "evidence_cache_query": session.evidence_cache_query,
            "session_history": [
                {
                    "turn_id": t.turn_id,
                    "user_message": t.user_message,
                    "assistant_reply": t.assistant_reply,
                    "intent": t.intent,
                    "rewritten_query": t.rewritten_query,
                }
                for t in session.working_memory.turns
            ],
        }

    # 基础响应（前端必须字段）
    response = {
        "session_id": session_id,
        "intent": intent_result.demands[0].intent_type if intent_result.demands else "chat",
        "is_clarification": False,
        "reply": reply.model_dump(),
    }

    # eval / 测试阶段才展开的中间态
    if request.eval_context:
        response.update({
            "route_meta": {
                "layer": "llm",
                "intent": intent_result.demands[0].intent_type if intent_result.demands else "chat",
                "confidence": intent_result.demands[0].confidence if intent_result.demands else 0.0,
                "demands": [{"intent": d.intent_type, "entities": d.entities} for d in intent_result.demands],
                "plan_tasks": len(graph.tasks),
                "plan_errors": errors,
            },
            "agent_mode": "llm",
            "llm_agent": {
                "rewritten_query": rewrite_result.rewritten_query,
                "search_keywords": rewrite_result.search_keywords,
                "is_follow_up": rewrite_result.is_follow_up,
                "follow_up_type": rewrite_result.follow_up_type,
                "global_slots": intent_result.resolved_entities,
                "tool_outputs": [{"tool": o["tool"], "task_id": o["task_id"]} for o in tool_outputs],
            },
            "agent": {
                "rewritten_query": rewrite_result.rewritten_query,
                "tools": tool_summary,
                "kb_chunks_count": len(tool_outputs),
                "system_prompt_preview": system_prompt[:200] if system_prompt else "",
            },
            "memory": memory_state,
        })
        if debug_info:
            response["debug_info"] = debug_info
    return response


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
    multi_result: "MultiIntentResult",
    resume_text: str,
):
    """
    v2 路线的 SSE 流式处理生成器。
    在关键步骤推送进度事件，最终聚合使用 LLM 流式生成。
    """
    import time

    # ── 话题切换检测 ──
    if session.working_memory.turns:
        try:
            mm = MemoryManager(llm_client=LLMClient.from_config("memory"))
            is_shift = await mm.detect_topic_shift(rewrite_result.rewritten_query or request.message, session)
            if is_shift:
                logger.info(f"[Chat-v2-Stream] 检测到话题切换，已清除 evidence_cache | session={session_id}")
        except Exception as e:
            logger.warning(f"[Chat-v2-Stream] 话题切换检测失败: {e}")

    # ── ① 新体系：多意图识别 ──
    yield _sse("status", {"step": "intent", "message": "正在识别您的意图..."})

    # 澄清状态机恢复
    pc = session.pending_clarification
    current_turn_id = len(session.working_memory.turns) + 1
    # 触发条件：follow_up_type=clarify，或有pending_clarification且输入很短（补充缺失槽位）
    is_clarify_follow_up = (
        rewrite_result.follow_up_type == "clarify"
        or (
            pc
            and not pc.is_expired(current_turn_id, max_gap=2)
            and len(request.message) < 20
        )
    )
    if (
        pc
        and not pc.is_expired(current_turn_id, max_gap=2)
        and is_clarify_follow_up
    ):
        logger.info(f"[Chat-v2-Stream] 澄清状态恢复 | intent={pc.pending_intent}")
        merged_slots = dict(pc.resolved_slots)
        merged_slots.update(rewrite_result.resolved_references or {})
        if rewrite_result.rewritten_query and rewrite_result.rewritten_query != request.message:
            merged_slots["search_keywords"] = rewrite_result.rewritten_query

        # 从用户原始输入中直接提取实体（公司名/岗位名），补充缺失槽位
        if pc.missing_slots:
            from app.core.llm_intent import _load_kb_entities
            kb_companies, kb_positions = _load_kb_entities()
            msg = request.message or ""
            # 按长度降序匹配，优先命中更长的实体（如 "字节跳动" 优先于 "字节"）
            if "company" in pc.missing_slots and "company" not in merged_slots:
                for c in sorted(kb_companies, key=len, reverse=True):
                    if c in msg:
                        merged_slots["company"] = c
                        break
            if "position" in pc.missing_slots and "position" not in merged_slots:
                for p in sorted(kb_positions, key=len, reverse=True):
                    if p in msg:
                        merged_slots["position"] = p
                        break

        from app.core.llm_intent import MultiIntentResult, IntentCandidate, LLMIntentType, _create_rule_registry
        from app.core.new_arch_adapter import map_intent_name_to_new
        new_intent_name = map_intent_name_to_new(pc.pending_intent)
        intent_type = LLMIntentType(new_intent_name) if new_intent_name else LLMIntentType.CHAT

        # 当 pending_intent=chat（意图模糊）时，基于原始 query 重新推断真实意图
        if intent_type == LLMIntentType.CHAT and request.message:
            if pc and not pc.is_expired(current_turn_id, max_gap=2):
                intent_router = LLMIntentRouter(chat_llm=llm_for_intent)
                re_multi_result = await intent_router.route_multi(
                    raw_message=request.message,
                    session=session,
                    attachments=request.attachments or [],
                )
                if re_multi_result.candidates:
                    intent_type = re_multi_result.primary_intent or re_multi_result.candidates[0].intent_type
                    for cand in re_multi_result.candidates:
                        merged_slots.update(cand.slots or {})
                    logger.info(f"[Chat-v2-Stream] 澄清后重新识别意图 | inferred={intent_type.value} | candidates={[c.intent_type.value for c in re_multi_result.candidates]}")
            else:
                # 无 pending_clarification 时：用规则匹配简单推断
                registry = _create_rule_registry()
                rule_matches = registry.classify_all(request.message, [])
                non_miss = [r for r in rule_matches if r.intent is not None]
                if non_miss:
                    best = non_miss[0]
                    intent_type = best.intent
                    merged_slots.update(best.metadata or {})
                    logger.info(f"[Chat-v2-Stream] 澄清推断意图 | inferred={intent_type.value} | rule={best.rule_name}")
        restored = IntentCandidate(
            intent_type=intent_type,
            confidence=0.85,
            reason=f"澄清恢复: 缺失槽位 {pc.missing_slots}",
            slots=merged_slots,
            slot_sources={k: "clarification_recovery" for k in merged_slots.keys()},
            missing_slots=[s for s in pc.missing_slots if s not in merged_slots],
            needs_clarification=False,
            source="clarification_recovery",
            rule_agreement=True,
        )
        # 判断是否需要再次澄清：VERIFY 有 company 或 position 之一即可；ASSESS 必须 resume
        needs_clarify = False
        clarify_reason = None
        if restored.missing_slots:
            needs_clarify = True
            if intent_type == LLMIntentType.VERIFY:
                if merged_slots.get("company") or merged_slots.get("position"):
                    needs_clarify = False
            elif intent_type == LLMIntentType.ASSESS:
                if merged_slots.get("resume_available"):
                    needs_clarify = False
            if needs_clarify:
                clarify_reason = f"还需要补充：{', '.join(restored.missing_slots)}"
        multi_result = MultiIntentResult(
            candidates=[restored],
            primary_intent=intent_type,
            needs_clarification=needs_clarify,
            clarification_reason=clarify_reason,
            global_slots=merged_slots,
            execution_topology=[[intent_type]],
        )
    else:
        # 非 clarify 场景：直接使用传入的 multi_result
        if pc and pc.is_expired(current_turn_id, max_gap=2):
            session.pending_clarification = None

    intent_result = multi_intent_result_to_intent_result(multi_result)

    # 将意图识别解析的槽位同步到 session.global_slots，供 ReActExecutor 动态解析占位符使用
    if not hasattr(session, "global_slots"):
        session.global_slots = {}
    if multi_result.global_slots:
        for k, v in multi_result.global_slots.items():
            if v is not None:
                session.global_slots[k] = v
    # 补充 search_keywords 和 rewritten_query（Planner 占位符需要）
    if rewrite_result.search_keywords:
        session.global_slots["search_keywords"] = rewrite_result.search_keywords
    if rewrite_result.rewritten_query:
        session.global_slots["query"] = rewrite_result.rewritten_query
        session.global_slots["rewritten_query"] = rewrite_result.rewritten_query

    # 澄清场景
    if intent_result.needs_clarification:
        reply_text = intent_result.clarification_question or "抱歉，我没有完全理解您的意思，能再详细说明一下吗？"
        yield _sse("clarification", {"question": reply_text})
        # 保存澄清状态
        turn_id = len(session.working_memory.turns) + 1
        primary = intent_result.demands[0].intent_type if intent_result.demands else "chat"
        session.pending_clarification = PendingClarification(
            pending_intent=primary,
            missing_slots=intent_result.missing_entities or [],
            clarification_question=reply_text,
            created_turn_id=turn_id,
            resolved_slots=dict(intent_result.resolved_entities or {}),
        )
        yield _sse("done", {
            "session_id": session_id,
            "intent": primary,
            "is_clarification": True,
            "reply": {"text": reply_text},
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

    new_planner = TaskGraphPlanner()
    new_graph = await new_planner.create_graph(
        multi_result=multi_result,
        session=session,
        resume_text=resume_text,
        rewrite_result=rewrite_result,
        history_cache=[],
    )
    graph = convert_task_graph(new_graph)

    # ── ③ ReAct 执行 ──
    yield _sse("status", {"step": "execute", "message": "正在执行分析..."})
    executor = ReActExecutor()
    graph = await executor.execute(graph, session)

    # Update evidence cache
    for task in graph.tasks.values():
        if task.status == "success" and task.result:
            chunks = task.result.get("chunks", []) if isinstance(task.result, dict) else []
            if chunks:
                session.evidence_cache = chunks[:settings.EVIDENCE_CACHE_MAX_SIZE]
                session.evidence_cache_query = rewrite_result.rewritten_query
                break

    # ── ④ 收集工具输出 ──
    tool_outputs = []
    tool_summary = []
    for task in graph.tasks.values():
        if task.tool_name and task.status in ("success", "failed"):
            result_preview = ""
            if task.result:
                if isinstance(task.result, dict):
                    result_preview = str(task.result.get("data", task.result))[:200]
                else:
                    result_preview = str(task.result)[:200]
            elif task.observation:
                result_preview = str(task.observation)[:200]
            tool_summary.append({
                "tool": task.tool_name,
                "status": "✅" if task.status == "success" else "❌",
                "params": task.resolved_params if hasattr(task, "resolved_params") else task.parameters,
                "result_preview": result_preview,
            })
        if task.status == "success" and task.result:
            result = task.result
            # 精简 kb_retrieve / global_rank / external_search 结果，避免 tool_outputs 过长被截断
            if task.tool_name == "kb_retrieve" and isinstance(result, dict):
                result = dict(result)
                chunks = result.get("chunks", [])
                result["chunks"] = [
                    {
                        "chunk_id": c.get("chunk_id", ""),
                        "content": c.get("content", "")[:300],
                        "metadata": {
                            "company": c.get("metadata", {}).get("company", ""),
                            "position": c.get("metadata", {}).get("position", ""),
                        },
                    }
                    for c in chunks[:6]
                ]
            elif task.tool_name == "global_rank" and isinstance(result, dict):
                result = dict(result)
                rankings = result.get("rankings", [])
                result["rankings"] = [
                    {
                        "company": r.get("company", ""),
                        "position": r.get("position", ""),
                        "score": r.get("score", 0),
                        "reason": r.get("reason", "")[:200],
                    }
                    for r in rankings[:5]
                ]
            elif task.tool_name == "external_search" and isinstance(result, dict):
                result = dict(result)
                chunks = result.get("chunks", [])
                result["chunks"] = [
                    {
                        "title": c.get("metadata", {}).get("title", "") if isinstance(c, dict) else "",
                        "content": c.get("content", "")[:300] if isinstance(c, dict) else "",
                        "source_name": c.get("metadata", {}).get("source_name", "") if isinstance(c, dict) else "",
                        "url": c.get("metadata", {}).get("url", "") if isinstance(c, dict) else "",
                    }
                    for c in chunks[:3]
                ]
            tool_outputs.append({
                "task_id": task.task_id,
                "tool": task.tool_name,
                "description": task.description,
                "result": result,
            })

    system_prompt = (
        "你是一位专业的求职顾问。请基于以下工具执行结果，给用户一个清晰、有帮助的回复。\n"
        "要求：\n"
        "1. 先直接回答用户的问题（是/否/有/没有），不要绕弯子，不要重复用户的原话\n"
        "2. 如果有匹配分析结果，给出具体的分数和建议\n"
        "3. 如果有检索结果，基于事实回答，不要编造\n"
        "4. 引用信息时必须标注来源，格式如：[来源：本地知识库-字节跳动] 或 [来源：外部搜索-新华网]\n"
        "5. 当本地知识库与外部搜索结果冲突时，请并列呈现两种说法并标注各自来源，不要替用户做判断\n"
        "6. 语气友好、专业、结构化"
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

    # 清理过期的澄清状态
    if session.pending_clarification:
        current_turn_id = len(session.working_memory.turns)
        if session.pending_clarification.is_expired(current_turn_id, max_gap=2):
            logger.info(
                f"[Chat-v2-stream] 澄清状态过期清理 | intent={session.pending_clarification.pending_intent}"
            )
            session.pending_clarification = None

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
        "agent": {
            "rewritten_query": rewrite_result.rewritten_query,
            "tools": tool_summary,
            "kb_chunks_count": len(tool_outputs),
            "system_prompt_preview": system_prompt[:200] if system_prompt else "",
        },
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

    # 同步简历可用状态到 session.global_slots，供意图识别层 _check_clarification_need 使用
    if not hasattr(session, "global_slots"):
        session.global_slots = {}
    has_resume = resume_text != "（用户尚未上传简历）"
    session.global_slots["resume_available"] = has_resume
    if has_resume:
        session.global_slots["resume_text"] = resume_text

    # 评测模式：注入模拟多轮上下文
    eval_ctx = request.eval_context or {}
    injected_slots = eval_ctx.get("injected_history_slots")
    if injected_slots and isinstance(injected_slots, dict):
        if session.long_term is None:
            from app.core.memory import LongTermMemory
            session.long_term = LongTermMemory()
        session.long_term.entities.update(injected_slots)
    injected_cache = eval_ctx.get("injected_evidence_cache")
    if injected_cache and isinstance(injected_cache, list):
        session.evidence_cache = list(injected_cache)
        session.evidence_cache_query = eval_ctx.get("injected_evidence_query", "")

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

    # ════════════════════════════════════════
    # Step 0: Query 改写（两条路线共用）
    # ════════════════════════════════════════
    rewriter = QueryRewriter()
    rewrite_result = await rewriter.rewrite(raw_query=message_with_ocr, session=session)
    logger.info(
        f"[ChatStream] Query改写 | original='{request.message[:30]}...' "
        f"| rewritten='{rewrite_result.rewritten_query[:40]}...' "
        f"| follow_up={rewrite_result.is_follow_up}/{rewrite_result.follow_up_type} "
        f"| source_pref={rewrite_result.source_preference}"
    )

    # Step 1: 意图识别（基于改写后的 query）
    from app.core.llm_intent import LLMIntentRouter
    llm_for_intent = LLMClient.from_config("chat")
    intent_router = LLMIntentRouter(chat_llm=llm_for_intent)
    multi_result = await intent_router.route_multi(
        rewrite_result=rewrite_result,
        session=session,
        attachments=request.attachments or [],
    )
    logger.info(
        f"[ChatStream] 意图识别 | primary={multi_result.primary_intent.value if multi_result.primary_intent else 'None'} "
        f"| candidates={[c.intent_type.value for c in multi_result.candidates]}"
    )

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
            multi_result=multi_result,
            resume_text=resume_text,
        ):
            yield event

    return StreamingResponse(event_generator(), media_type="text/event-stream")
