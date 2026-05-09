"""
多轮对话记忆机制 —— 三层记忆架构

┌─────────────────────────────────────────────────────────────┐
│                    SessionMemory（会话记忆）                   │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │ Working Mem  │  │ Compressed   │  │ Long-Term Mem    │ │
│  │ (最近3轮)     │  │ Mem (4-10轮) │  │ (实体/偏好/简历)  │ │
│  └──────────────┘  └──────────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────┘
"""

import logging
import time
import json
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from app.core.llm_client import LLMClient, TIMEOUT_LIGHT, TIMEOUT_STANDARD

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# 0. 枚举定义
# ═══════════════════════════════════════════════════════

class FollowUpType(Enum):
    """追问类型：决定检索策略"""
    EXPAND = "expand"      # 展开型：在原证据里深挖（如"具体怎么做？"）
    EXTEND = "extend"      # 扩展型：引入新信息（如"2024年政策变化？"）
    FORMAT = "format"      # 格式型：改变输出形式（如"用表格总结"）
    FIRST = "first"        # 首轮对话：无历史


class RetrievalAction(Enum):
    """检索决策动作"""
    FULL = "full"              # 全新检索（首轮或话题切换）
    REUSE = "reuse"            # 复用旧证据（rerank后相关度足够）
    INCREMENTAL = "incremental" # 增量检索（合并新旧证据）
    NO_RETRIEVAL = "no_retrieval" # 无需检索（格式型）


# ═══════════════════════════════════════════════════════
# 1. 三层记忆数据模型
# ═══════════════════════════════════════════════════════

@dataclass
class DialogueTurn:
    """单轮对话的完整记录"""
    turn_id: int
    user_message: str
    assistant_reply: str
    intent: str
    rewritten_query: str
    tool_calls: List[Any] = field(default_factory=list)
    tool_results: List[Any] = field(default_factory=list)
    retrieved_chunks: List[Dict] = field(default_factory=list)  # 检索证据
    evidence_score: float = 0.0  # 旧证据rerank后的最高分
    timestamp: float = field(default_factory=time.time)

    # ── 执行编排历史（TaskGraph 快照）──
    task_graph_snapshot: Optional[Dict[str, Any]] = None  # 本轮的完整 TaskGraph 序列化


@dataclass
class WorkingMemory:
    """工作内存：最近3轮完整对话（直接参与推理）"""
    turns: List[DialogueTurn] = field(default_factory=list)
    max_turns: int = 3

    def append(self, turn: DialogueTurn) -> None:
        """添加新轮次（不自动裁剪，裁剪由 rotate_memory 负责）"""
        self.turns.append(turn)

    def pop_overflow(self) -> Optional[DialogueTurn]:
        """若超出容量，弹出最老的一轮（待压缩）"""
        if len(self.turns) > self.max_turns:
            return self.turns.pop(0)
        return None

    def get_recent_context(self, n: int = 3, exclude_last: bool = False) -> str:
        """
        获取最近n轮的对话上下文文本。
        
        Args:
            n: 取最近n轮
            exclude_last: 为 True 时排除最后一轮（当前轮），只取历史轮次
        """
        if exclude_last and len(self.turns) > 0:
            recent = self.turns[-(n + 1):-1]
        else:
            recent = self.turns[-n:]
        lines = []
        for t in recent:
            lines.append(f"用户：{t.user_message}")
            lines.append(f"助手：{t.assistant_reply}")
        return "\n".join(lines)

    def get_user_messages_context(self, n: int = 3) -> str:
        """仅获取用户消息历史（用于长期记忆提取，排除 assistant 回复）"""
        recent = self.turns[-n:]
        lines = []
        for t in recent:
            lines.append(f"用户：{t.user_message}")
        return "\n".join(lines)


@dataclass
class CompressedMemory:
    """压缩记忆：4-10轮的摘要（节省token，保留关键事实）"""
    memory_id: str
    summary: str                    # LLM生成的对话摘要
    key_facts: List[str]            # 提取的关键事实
    start_turn: int
    end_turn: int
    created_at: float = field(default_factory=time.time)


@dataclass
class LongTermMemory:
    """长期记忆：跨会话持久化的用户画像"""
    user_id: str
    entities: Dict[str, Any] = field(default_factory=dict)  # NER提取：技能、公司、学校
    preferences: Dict[str, Any] = field(default_factory=dict)  # 偏好：行业、薪资、城市
    resume_fingerprint: str = ""    # 简历摘要哈希/ID
    topic_flags: Dict[str, Any] = field(default_factory=dict)  # 话题切换标记
    last_updated: float = field(default_factory=time.time)


@dataclass
class PendingClarification:
    """澄清状态：当系统需要用户补充信息时保存"""
    pending_intent: str = ""           # 如 "verify", "assess"
    missing_slots: List[str] = field(default_factory=list)
    clarification_question: str = ""   # 上轮系统问的问题
    expected_slot_types: List[str] = field(default_factory=list)  # 期望补充的槽位类型
    created_turn_id: int = 0           # 创建时的轮次ID（用于过期判断）
    resolved_slots: Dict[str, Any] = field(default_factory=dict)  # 已解析的槽位（由QueryRewriter提供）

    def is_expired(self, current_turn_id: int, max_gap: int = 2) -> bool:
        """超过 max_gap 轮未解决则过期"""
        return current_turn_id - self.created_turn_id > max_gap


@dataclass
class SessionMemory:
    """完整的会话记忆容器"""
    session_id: str
    working_memory: WorkingMemory = field(default_factory=WorkingMemory)
    compressed_memories: List[CompressedMemory] = field(default_factory=list)
    long_term: Optional[LongTermMemory] = None

    # 检索证据缓存：上一轮检索的chunks，用于多轮复用
    evidence_cache: List[Dict] = field(default_factory=list)
    evidence_cache_query: str = ""  # 缓存对应的查询

    # 话题切换检测：当前话题标识
    current_topic: str = "general"

    # 用户通过附件上传的 JD 文本（OCR 提取），存在时跳过知识库检索
    user_provided_jd: str = ""

    # 澄清状态机：当系统触发澄清时保存，用户回复后恢复
    pending_clarification: Optional[PendingClarification] = None


# ═══════════════════════════════════════════════════════
# 2. 检索决策数据模型
# ═══════════════════════════════════════════════════════

@dataclass
class RetrievalDecision:
    """检索决策结果"""
    action: RetrievalAction
    reason: str = ""
    reused_evidence: List[Dict] = field(default_factory=list)  # REUSE时返回
    incremental_query: str = ""  # INCREMENTAL时的改写后查询
    threshold_score: float = 0.0  # rerank阈值分数


# ═══════════════════════════════════════════════════════
# 3. 记忆管理器
# ═══════════════════════════════════════════════════════

class MemoryManager:
    """
    记忆管理器：
    - 维护工作内存（最近3轮）
    - 触发压缩（4-10轮 → 摘要）
    - 提取长期实体/偏好
    - 检测话题切换
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client
        self.compress_trigger = 3  # 工作内存满3轮后触发压缩

    # ──────────────────────────── 1. 更新工作内存 ────────────────────────────

    async def update_working_memory(self, session: SessionMemory, turn: DialogueTurn) -> Optional[DialogueTurn]:
        """添加新轮次，溢出时返回最老的一轮"""
        overflow = session.working_memory.append(turn)
        return overflow

    # ──────────────────────────── 2. 压缩记忆 ────────────────────────────

    async def compress_turns(self, turns: List[DialogueTurn]) -> Optional[CompressedMemory]:
        """
        将溢出的对话轮次压缩为摘要。
        用轻量LLM做摘要，压缩率目标60%。
        """
        if not turns:
            return None

        dialogue_text = "\n".join([
            f"用户：{t.user_message}\n助手：{t.assistant_reply}\n"
            f"意图：{t.intent} | 检索证据数：{len(t.retrieved_chunks)}"
            for t in turns
        ])

        system_prompt = """你是对话摘要专家。请将以下对话压缩为结构化摘要，要求：
1. 保留所有关键事实、数字、用户决策、重要结论
2. 提取用户显性或隐性偏好（如" prefer A over B"）
3. 删除寒暄、重复确认、语气词
4. 输出格式：摘要 + 关键事实列表（bullet points）
压缩率目标：原文的40%长度（保留60%信息）。"""

        user_prompt = f"待压缩对话：\n{dialogue_text}"

        try:
            if self.llm is None:
                self.llm = LLMClient.from_config("memory")
            summary_raw = await self.llm.generate(
                prompt=user_prompt,
                system=system_prompt,
                temperature=0.3,
                max_tokens=800,
                timeout=TIMEOUT_STANDARD,  # 30s，记忆压缩属于标准调用
            )

            key_facts = [line.strip("- ").strip()
                         for line in summary_raw.split("\n")
                         if line.strip().startswith("-")]

            return CompressedMemory(
                memory_id=f"cm_{turns[0].turn_id}_{turns[-1].turn_id}",
                summary=summary_raw,
                key_facts=key_facts,
                start_turn=turns[0].turn_id,
                end_turn=turns[-1].turn_id,
            )
        except Exception as e:
            logger.warning(f"[MemoryManager] 压缩记忆失败: {e}")
            return None

    # ──────────────────────────── 3. 长期记忆更新 ────────────────────────────

    async def extract_long_term_facts(self, session: SessionMemory) -> Dict[str, Any]:
        """
        从工作内存中提取实体和偏好，更新长期记忆。
        只从**用户消息**中提取，避免把 assistant 回复或检索结果误当作用户信息。
        """
        if not session.working_memory.turns:
            return {}

        # 只取用户消息，排除 assistant 回复和检索证据
        recent = session.working_memory.get_user_messages_context(3)

        prompt = f"""从以下**用户消息**中提取结构化用户画像信息。
注意：只提取用户明确表达或提及的个人信息（技能、公司、岗位、偏好等），
不要提取 assistant 提到的任何公司或岗位信息。

用户消息：
{recent}

请输出JSON（不要markdown）：
{{
  "entities": {{"技能": [], "公司": [], "岗位": [], "学校": [], "项目": []}},
  "preferences": {{"行业": "", "薪资期望": "", "城市": "", "工作模式": ""}},
  "flags": {{"话题切换": false, "新需求": ""}}
}}"""

        try:
            if self.llm is None:
                self.llm = LLMClient.from_config("memory")
            raw = await self.llm.generate(prompt=prompt, temperature=0.2, max_tokens=400, timeout=TIMEOUT_LIGHT)
            raw = raw.strip().strip("```json").strip("```").strip()
            extracted = json.loads(raw)
            return extracted
        except Exception as e:
            logger.warning(f"[MemoryManager] 提取长期记忆失败: {e}")
            return {}

    async def update_long_term(self, session: SessionMemory, persist: bool = True):
        """合并提取到的事实到长期记忆，并可选持久化到数据库"""
        extracted = await self.extract_long_term_facts(session)
        if not session.long_term:
            session.long_term = LongTermMemory(user_id=session.session_id)

        lt = session.long_term
        for k, v in extracted.get("entities", {}).items():
            if k not in lt.entities:
                lt.entities[k] = []
            if isinstance(v, list):
                lt.entities[k] = list(set(lt.entities[k] + v))
            else:
                lt.entities[k] = v

        lt.preferences.update(extracted.get("preferences", {}))
        lt.last_updated = time.time()

        if persist:
            from app.core.db import save_long_term_memory
            save_long_term_memory(lt)

    @staticmethod
    def load_long_term(user_id: str) -> Optional[LongTermMemory]:
        """从数据库加载用户长期记忆"""
        from app.core.db import load_long_term_memory
        return load_long_term_memory(user_id)

    # ──────────────────────────── 4. 话题切换检测 ────────────────────────────

    async def detect_topic_shift(self, current_query: str, session: SessionMemory) -> bool:
        """
        检测是否发生话题切换。若切换，应：
        1. 清空证据缓存（旧证据不再适用）
        2. 触发新的FULL检索
        3. 可选：压缩当前话题记忆
        """
        if not session.working_memory.turns:
            return False

        recent = session.working_memory.get_recent_context(2)
        prompt = f"""判断用户新问题是否与当前话题一致。

当前话题上下文：
{recent}

新问题：{current_query}

请只输出：same 或 shift"""

        try:
            if self.llm is None:
                self.llm = LLMClient.from_config("memory")
            result = await self.llm.generate(prompt=prompt, temperature=0.1, max_tokens=5, timeout=TIMEOUT_LIGHT)
            is_shift = "shift" in result.lower()

            if is_shift:
                session.evidence_cache = []
                session.evidence_cache_query = ""
                session.current_topic = f"topic_{int(time.time())}"
                logger.info(f"[MemoryManager] 检测到话题切换，清空证据缓存")

            return is_shift
        except Exception as e:
            logger.warning(f"[MemoryManager] 话题切换检测失败: {e}")
            return False

    # ──────────────────────────── 5. 主入口：记忆轮转 ────────────────────────────

    async def rotate_memory(self, session: SessionMemory, new_turn: DialogueTurn, persist: bool = True):
        """
        每轮对话结束后调用，执行完整的记忆管理：
        1. 检查工作内存溢出，若有则压缩到压缩记忆
        2. 定期更新长期记忆
        3. 持久化会话元数据
        
        注意：new_turn 已由调用方 append 到 session.working_memory
        """
        # 检查是否有溢出的轮次
        overflow = session.working_memory.pop_overflow()
        if overflow:
            compressed = await self.compress_turns([overflow])
            if compressed:
                session.compressed_memories.append(compressed)
                if len(session.compressed_memories) > 10:
                    session.compressed_memories.pop(0)

        # 从第4轮开始，每轮都更新长期记忆（不再限制为每3轮一次）
        if new_turn.turn_id >= 4:
            await self.update_long_term(session, persist=persist)

        # 持久化会话元数据（含完整对话历史）
        if persist:
            from app.core.db import save_session_meta
            user_id = session.long_term.user_id if session.long_term else session.session_id
            dialogue_history = []
            for t in session.working_memory.turns:
                entry = {
                    "turn_id": t.turn_id,
                    "user_message": t.user_message,
                    "assistant_reply": t.assistant_reply,
                    "intent": t.intent,
                    "rewritten_query": t.rewritten_query,
                    "evidence_score": t.evidence_score,
                }
                if t.task_graph_snapshot:
                    entry["task_graph_snapshot"] = t.task_graph_snapshot
                dialogue_history.append(entry)
            save_session_meta(
                session_id=session.session_id,
                user_id=user_id,
                current_topic=session.current_topic,
                evidence_cache_query=session.evidence_cache_query,
                dialogue_history=dialogue_history,
            )
