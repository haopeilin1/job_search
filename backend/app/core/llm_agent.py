"""
LLM Agent 协调器 —— 整合 LLMIntentRecognizer → Clarification → LLMPlanner → ReActExecutor

这是 LLM Agent 路线的核心入口，与规则路线的 EnhancedAgentOrchestrator 平行。

职责：
1. 接收用户消息，调用 LLMIntentRecognizer 进行语义级意图识别
2. 根据置信度决定：澄清 or 继续
3. 调用 LLMPlanner 生成任务计划（CoT 拆解）
4. 调用 ReActExecutor 循环执行（执行+观察+调整）
5. 组装 AgentContext，供回复生成使用
6. 记忆轮转（复用 MemoryManager）
"""

import logging
import time
from typing import Optional, Any, Tuple
from dataclasses import dataclass, field

from app.core.llm_client import LLMClient
from app.core.memory import SessionMemory, DialogueTurn, MemoryManager
from app.core.intent import IntentResult
from app.core.clarification import ClarificationEngine, ClarificationResult
from app.core.llm_planner import LLMPlanner, TaskPlan
from app.core.react_executor import ReActExecutor
from app.core.tools import ToolCall, ToolResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# 1. LLM Agent 输出数据模型
# ═══════════════════════════════════════════════════════

@dataclass
class LLMAgentContext:
    """LLM Agent 执行后的上下文"""
    original_message: str
    rewritten_query: str
    intent: str
    intent_confidence: float
    intent_reason: str

    # 澄清状态
    is_clarification: bool = False
    clarification_question: str = ""
    clarification_options: list = field(default_factory=list)

    # 执行结果
    task_plan: Optional[TaskPlan] = None
    selected_tools: list = field(default_factory=list)
    tool_results: list = field(default_factory=list)
    kb_chunks: list = field(default_factory=list)
    resume_text: str = ""
    jd_text: str = ""

    # Prompt
    system_prompt: str = ""
    user_prompt: str = ""


# ═══════════════════════════════════════════════════════
# 2. LLM Agent 协调器
# ═══════════════════════════════════════════════════════

class LLMAgentOrchestrator:
    """
    LLM Agent 协调器（LLM 路线核心入口）。

    完整链路：
    1. LLMIntentRecognizer.recognize() → 意图识别
    2. 置信度判断：
       - < 0.7 → ClarificationEngine.analyze() → 返回澄清问题
       - >= 0.7 → 继续
    3. LLMPlanner.create_plan() → 生成任务计划
    4. ReActExecutor.execute() → 循环执行+调整
    5. 组装 LLMAgentContext → 构造 prompt → 回复生成
    6. MemoryManager.rotate_memory() → 记忆轮转
    """

    def __init__(self):
        self.llm = LLMClient.from_config("planner")
        self.clarification = ClarificationEngine(llm_client=self.llm)
        self.planner = LLMPlanner(llm_client=self.llm)
        self.executor = ReActExecutor(planner=self.planner)
        self.memory_manager = MemoryManager(llm_client=self.llm)

    async def run(
        self,
        intent_result: IntentResult,
        message: str,
        session: Optional[SessionMemory] = None,
        resume_text: str = "",
        attachments: list = None,
        rewritten_query: str = "",
        search_keywords: str = "",
        is_follow_up: bool = False,
        follow_up_type: str = "",
    ) -> Tuple[LLMAgentContext, SessionMemory]:
        """
        LLM Agent 完整执行入口。

        Args:
            intent_result: 意图识别结果
            message: 用户原始输入
            session: 会话记忆
            resume_text: 简历文本
            attachments: 附件列表
            rewritten_query: Query 改写后的查询（已做指代消解、口语降噪）
            search_keywords: 检索关键词
            is_follow_up: 是否追问
            follow_up_type: 追问类型

        Returns:
            (LLMAgentContext, SessionMemory)
        """
        attachments = attachments or []

        # 1. 初始化 session
        if session is None:
            session = SessionMemory(session_id=f"s_{int(time.time() * 1000)}")

        logger.info(f"[LLMAgent] 开始执行 | intent={intent_result.intent.value} | confidence={intent_result.confidence:.2f} | message='{message[:40]}...' | rewritten='{rewritten_query[:40] if rewritten_query else '(无)'}...'")

        # 2. 置信度判断 → 澄清 or 继续
        if intent_result.confidence < 0.7:
            logger.info(f"[LLMAgent] 置信度不足({intent_result.confidence:.2f})，进入澄清流程")
            clar_result = await self.clarification.analyze(intent_result, message, session)

            ctx = LLMAgentContext(
                original_message=message,
                rewritten_query=rewritten_query or message,
                intent=intent_result.intent.value,
                intent_confidence=intent_result.confidence,
                intent_reason=intent_result.reason or "",
                is_clarification=True,
                clarification_question=clar_result.question,
                clarification_options=clar_result.suggested_options,
                resume_text=resume_text,
            )
            # 构造澄清用的 prompt
            ctx.system_prompt, ctx.user_prompt = self._build_clarification_prompt(ctx)

            # 记录对话轮次（但 assistant_reply 待生成）
            turn = DialogueTurn(
                turn_id=len(session.working_memory.turns) + 1,
                user_message=message,
                assistant_reply="",
                intent=intent_result.intent,
                rewritten_query=rewritten_query or message,
                evidence_score=intent_result.confidence,
            )
            session.working_memory.append(turn)
            await self.memory_manager.rotate_memory(session, turn)

            return ctx, session

        # 4. LLM 规划（CoT 任务拆解，使用改写后的 query）
        task_plan = await self.planner.create_plan(intent_result, session, resume_text, rewritten_query=rewritten_query or message)
        logger.info(f"[LLMAgent] 计划生成 | tasks={len(task_plan.tasks)} | thought={task_plan.thought[:80]}...")

        # 5. ReAct 执行
        completed_plan = await self.executor.execute(task_plan, session)
        logger.info(f"[LLMAgent] ReAct 执行完成")

        # 6. 提取执行结果
        kb_chunks = []
        jd_text = ""
        tool_calls = []
        tool_results = []

        for task in completed_plan.tasks:
            if task.tool and task.result:
                tool_calls.append(ToolCall(name=task.tool, parameters=task.parameters))
                tool_results.append(task.result)
                if task.tool == "kb_retrieve" and task.result.success:
                    chunks = task.result.data.get("chunks", []) if task.result.data else []
                    kb_chunks.extend(chunks)
                if task.tool == "match_analyze" and task.result.success:
                    jd_text = task.result.data.get("jd_text", "") if task.result.data else ""

        # 7. 组装 AgentContext
        # Query 改写结果从 task_plan.context 中获取
        rewritten_query = task_plan.context.get("rewritten_query", message) if task_plan.context else message

        ctx = LLMAgentContext(
            original_message=message,
            rewritten_query=rewritten_query,
            intent=intent_result.intent.value,
            intent_confidence=intent_result.confidence,
            intent_reason=intent_result.reason or "",
            is_clarification=False,
            task_plan=completed_plan,
            selected_tools=tool_calls,
            tool_results=tool_results,
            kb_chunks=kb_chunks,
            resume_text=resume_text,
            jd_text=jd_text,
        )

        # 8. 构造 LLM prompt
        ctx.system_prompt, ctx.user_prompt = self._build_response_prompt(ctx, session, intent_result)

        # 9. 记录对话轮次
        turn = DialogueTurn(
            turn_id=len(session.working_memory.turns) + 1,
            user_message=message,
            assistant_reply="",  # 待 LLM 生成后回填
            intent=intent_result.intent.value,
            rewritten_query=rewritten_query,
            tool_calls=tool_calls,
            tool_results=tool_results,
            retrieved_chunks=kb_chunks,
            evidence_score=intent_result.confidence,
        )
        session.working_memory.append(turn)
        await self.memory_manager.rotate_memory(session, turn)

        logger.info(f"[LLMAgent] 执行完成 | intent={ctx.intent} | tools={len(tool_calls)} | chunks={len(kb_chunks)}")
        return ctx, session

    # ──────────────────────────── Prompt 构造 ────────────────────────────

    def _build_clarification_prompt(self, ctx: LLMAgentContext) -> Tuple[str, str]:
        """构造澄清场景的 prompt"""
        system = """你是「求职雷达」AI 助手小橘 🍊，一位专业的求职顾问。语气友好、专业、有温度。

当前用户意图不够明确，你需要礼貌地引导用户补充信息。"""

        options_text = ""
        if ctx.clarification_options:
            options_text = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(ctx.clarification_options))

        user = f"""【用户问题】{ctx.original_message}

【系统推测】
- 可能意图：{ctx.intent}
- 置信度：{ctx.intent_confidence:.2f}
- 需要澄清的原因：{ctx.intent_reason}

【澄清问题】
{ctx.clarification_question}

{options_text if options_text else ""}

请用友好、简洁的语气向用户提出澄清问题，可以给出选项引导用户选择。"""

        return system, user

    def _build_response_prompt(
        self,
        ctx: LLMAgentContext,
        session: SessionMemory,
        intent_result: IntentResult,
    ) -> Tuple[str, str]:
        """构造正常回复场景的 system/user prompt"""

        # System prompt：根据意图设定角色
        base_prompts = {
            "match_single": "你是一位资深猎头顾问，擅长分析简历与岗位描述的匹配度。请基于下方的任务执行结果，给出专业、客观、有建设性的分析。",
            "global_match": "你是一位资深猎头顾问。请基于下方的任务执行结果，按匹配度从高到低排序，给出投递策略建议。",
            "rag_qa": "你是一位求职信息顾问。请基于下方的知识库检索结果，准确回答用户问题。如果检索结果中没有相关信息，请明确告知用户。",
            "general": "你是「求职雷达」AI 助手小橘🍊，一位专业的求职顾问。语气友好、专业、有温度。",
        }
        system = base_prompts.get(ctx.intent, base_prompts["general"])

        # 注入工具白名单约束（与规则路线一致）
        tool_constraint = self._get_tool_constraint_prompt(ctx.intent)
        system += "\n\n" + tool_constraint

        # 注入长期记忆偏好
        if session.long_term and session.long_term.preferences.get("行业"):
            system += f"\n用户关注行业：{session.long_term.preferences['行业']}，请优先推荐相关岗位。"

        # User prompt：整合所有信息（三层记忆：工作记忆3轮 + 压缩记忆4-10轮 + 长期记忆）
        parts = []

        # ── 1. 工作内存：最近3轮完整对话 ──
        if len(session.working_memory.turns) > 1:
            parts.append("【最近对话历史】\n" + session.working_memory.get_recent_context(3, exclude_last=True))

        # ── 2. 压缩记忆：4-10轮摘要 ──
        if session.compressed_memories:
            recent_cm = session.compressed_memories[-10:]
            cm_lines = ["【更早对话摘要】"]
            for cm in recent_cm:
                cm_lines.append(f"轮次 {cm.start_turn}-{cm.end_turn}：{cm.summary[:300]}")
                for fact in cm.key_facts[:5]:
                    cm_lines.append(f"  - {fact}")
            parts.append("\n".join(cm_lines))

        # ── 3. 当前问题 ──
        parts.append(f"【用户问题】\n{ctx.original_message}")
        parts.append(f"【改写后的问题】\n{ctx.rewritten_query}")

        # ── 4. 执行计划摘要 ──
        if ctx.task_plan:
            parts.append(ctx.task_plan.get_summary())

        # ── 5. 工具输出 ──
        for tool_call, tool_result in zip(ctx.selected_tools, ctx.tool_results):
            if tool_result.success and tool_result.data:
                import json
                preview = json.dumps(tool_result.data, ensure_ascii=False, indent=2)[:1500]
                parts.append(f"【工具输出: {tool_call.name}】\n{preview}")
            elif not tool_result.success:
                parts.append(f"【工具输出: {tool_call.name}】\n❌ 执行失败: {tool_result.error}")

        # ── 6. 检索证据 ──
        if ctx.kb_chunks:
            import json
            chunks_text = "\n---\n".join(
                f"[来源: {c.get('metadata', {}).get('company', '未知')} - {c.get('metadata', {}).get('section', '未知')}]\n{c.get('content', '')[:500]}"
                for c in ctx.kb_chunks
            )
            parts.append("【检索证据】\n" + chunks_text)

        # ── 7. 简历 ──
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
        if lt_parts:
            parts.append("【用户画像】\n" + "\n".join(lt_parts))

        user = "\n\n".join(parts)
        return system, user

    def _get_tool_constraint_prompt(self, intent: str) -> str:
        """获取工具约束说明"""
        tool_desc = {
            "kb_retrieve": "知识库检索",
            "match_analyze": "简历匹配分析",
            "interview_questions": "面试题生成",
        }

        if intent in ("match_single", "global_match"):
            allowed = ["kb_retrieve", "match_analyze", "interview_questions"]
        elif intent == "rag_qa":
            allowed = ["kb_retrieve"]
        else:
            return "【工具约束】当前为通用对话，不使用任何工具。"

        lines = ["【工具约束】根据当前意图，可参考以下工具输出："]
        for t in allowed:
            lines.append(f"  - {t}：{tool_desc.get(t, '')}")
        return "\n".join(lines)
