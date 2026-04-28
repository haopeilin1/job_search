"""
多轮澄清引擎 —— 当 LLM 意图识别置信度不足时，生成澄清问题引导用户

职责：
1. 分析意图识别的模糊点
2. 生成针对性的澄清问题
3. 提供可选选项（如有）
4. 收集澄清后的信息，辅助下一轮意图识别
"""

import json
import logging
import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from app.core.llm_client import LLMClient
from app.core.memory import SessionMemory
from app.core.intent import IntentResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# 1. 数据模型
# ═══════════════════════════════════════════════════════

@dataclass
class ClarificationResult:
    """澄清结果"""
    needs_clarification: bool      # 是否需要澄清（由调用方根据 confidence 决定）
    question: str                  # 给用户的澄清问题
    suggested_options: List[str] = field(default_factory=list)  # 建议选项
    reason: str = ""               # 为什么需要澄清
    original_intent_hint: str = "" # 原始意图推测（用于调试）


# ═══════════════════════════════════════════════════════
# 2. 澄清引擎
# ═══════════════════════════════════════════════════════

class ClarificationEngine:
    """
    多轮澄清引擎。

    核心流程：
    1. 接收 IntentResult（含 confidence 和 reason）
    2. 分析模糊点（意图不明？实体缺失？信息不足？）
    3. 调用 LLM 生成澄清问题
    4. 返回 ClarificationResult

    澄清后的信息会被注入到 session 的 clarification_context 中，
    下一轮意图识别时会参考这些上下文。
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client

    async def analyze(
        self,
        intent_result: IntentResult,
        message: str,
        session: SessionMemory,
    ) -> ClarificationResult:
        """
        分析是否需要澄清，并生成澄清问题。

        Args:
            intent_result: LLM 意图识别结果
            message: 用户原始消息
            session: 当前会话记忆

        Returns:
            ClarificationResult
        """
        # 高置信度，无需澄清
        if intent_result.confidence >= 0.7:
            return ClarificationResult(
                needs_clarification=False,
                question="",
                reason=f"confidence={intent_result.confidence:.2f} >= 0.7，无需澄清",
            )

        # 低置信度，生成澄清问题
        if self.llm is None:
            # 无 LLM，使用规则 fallback
            return self._rule_clarification(intent_result, message)

        try:
            return await self._llm_clarification(intent_result, message, session)
        except Exception as e:
            logger.warning(f"[ClarificationEngine] LLM 澄清失败: {e}，fallback 到规则")
            return self._rule_clarification(intent_result, message)

    async def build_clarification_context(
        self,
        session: SessionMemory,
        clarification_result: ClarificationResult,
        user_response: str,
    ) -> Dict[str, Any]:
        """
        将用户的澄清回答整合到上下文中，供下一轮意图识别使用。

        Returns:
            澄清上下文字典，包含用户补充的信息
        """
        # 简单实现：将澄清问题和用户回答存入 session
        if not hasattr(session, "_clarification_history"):
            session._clarification_history = []

        session._clarification_history.append({
            "question": clarification_result.question,
            "options": clarification_result.suggested_options,
            "user_response": user_response,
        })

        return {
            "clarification_history": session._clarification_history,
            "last_response": user_response,
        }

    # ── LLM 澄清 ──

    async def _llm_clarification(
        self,
        intent_result: IntentResult,
        message: str,
        session: SessionMemory,
    ) -> ClarificationResult:
        """调用 LLM 生成澄清问题"""

        system_prompt = """你是一位对话澄清专家。当用户意图不够明确时，你需要生成一个简短的澄清问题，帮助用户表达真实需求。

【原则】
1. 问题要简短（不超过30字），友好自然
2. 提供 2-4 个可选选项，降低用户回答成本
3. 不要重复用户已经说过的话
4. 基于意图识别的模糊点针对性提问

【输出格式】
严格 JSON：
{
  "question": "澄清问题",
  "options": ["选项1", "选项2", "选项3"],
  "reason": "为什么需要澄清",
  "intent_hint": "系统推测的用户可能意图"
}"""

        # 构建上下文
        history = ""
        if session.working_memory.turns:
            history = session.working_memory.get_recent_context(2, exclude_last=True)

        clarification_history = getattr(session, "_clarification_history", [])
        clarification_text = ""
        if clarification_history:
            lines = ["【已进行的澄清】"]
            for c in clarification_history:
                lines.append(f"问：{c['question']}")
                lines.append(f"答：{c['user_response']}")
            clarification_text = "\n".join(lines)

        user_prompt = f"""【当前用户输入】
{message}

【意图识别结果】
- 推测意图：{intent_result.intent}
- 置信度：{intent_result.confidence:.2f}
- 判断理由：{intent_result.reason}
- 改写后 query：{intent_result.rewritten_query}

{clarification_text}

{history if history else ""}

请生成澄清问题，帮助用户明确真实意图。输出 JSON："""

        raw = await self.llm.generate(
            prompt=user_prompt,
            system=system_prompt,
            temperature=0.4,
            max_tokens=400,
        )

        # 解析
        text = raw.strip()
        for marker in ["```json", "```"]:
            if marker in text:
                text = re.sub(rf"^{re.escape(marker)}\s*|\s*{re.escape(marker)}$", "", text, flags=re.MULTILINE).strip()

        data = json.loads(text)

        return ClarificationResult(
            needs_clarification=True,
            question=data.get("question", "能再说得具体一点吗？"),
            suggested_options=data.get("options", []),
            reason=data.get("reason", f"confidence={intent_result.confidence:.2f} 过低"),
            original_intent_hint=data.get("intent_hint", ""),
        )

    # ── 规则澄清（Fallback）──

    def _rule_clarification(
        self,
        intent_result: IntentResult,
        message: str,
    ) -> ClarificationResult:
        """规则兜底澄清"""

        confidence = intent_result.confidence
        intent = intent_result.intent

        # 根据意图和置信度生成不同的澄清问题
        if intent == "general" and confidence < 0.5:
            return ClarificationResult(
                needs_clarification=True,
                question="你想了解哪方面的信息呢？",
                suggested_options=["岗位匹配分析", "查询公司/岗位信息", "面试准备建议", "职业规划咨询"],
                reason="意图非常模糊，无法判断用户需求",
                original_intent_hint="general",
            )

        if intent in ("match_single", "global_match"):
            # 匹配类意图但实体缺失
            if not intent_result.metadata.get("company") and not intent_result.metadata.get("position"):
                return ClarificationResult(
                    needs_clarification=True,
                    question="你想分析哪个公司或岗位的匹配度呢？",
                    suggested_options=["字节跳动", "阿里巴巴", "腾讯", "其他公司"],
                    reason="匹配意图已识别，但缺少目标公司/岗位信息",
                    original_intent_hint=intent,
                )

        if intent == "rag_qa":
            # RAG 意图但属性不明
            attrs = intent_result.metadata.get("attributes", [])
            if not attrs:
                return ClarificationResult(
                    needs_clarification=True,
                    question="你想了解该公司的哪方面信息？",
                    suggested_options=["薪资福利", "技能要求", "工作节奏", "团队规模"],
                    reason="RAG 意图已识别，但缺少具体属性查询目标",
                    original_intent_hint="rag_qa",
                )

        # 默认澄清
        return ClarificationResult(
            needs_clarification=True,
            question="能再详细说说你的需求吗？",
            suggested_options=["我想做岗位匹配", "我想查公司信息", "我想准备面试"],
            reason=f"confidence={confidence:.2f} 偏低，需要更多信息",
            original_intent_hint=intent,
        )
