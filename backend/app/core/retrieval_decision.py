"""
检索决策引擎 —— 多轮 RAG 的"大脑"

负责判断：这轮要不要检索？复用旧证据？还是增量检索？
流程：追问类型分类 → 旧证据rerank → 决策输出
"""

import logging
from typing import List, Optional, Dict, Any

from app.core.memory import (
    FollowUpType, RetrievalAction, RetrievalDecision, SessionMemory, LongTermMemory
)
from app.core.llm_client import LLMClient, TIMEOUT_LIGHT
from app.core import reranker

logger = logging.getLogger(__name__)


class RetrievalDecisionEngine:
    """
    检索决策引擎：
    1. 追问类型分类（展开/扩展/格式）
    2. 旧证据rerank（判断相关性）
    3. 输出决策（复用 / 增量 / 跳过）
    """

    def __init__(self, llm_client: Optional[LLMClient] = None, threshold: float = 0.72):
        self.llm = llm_client
        self.threshold = threshold  # 复用旧证据的相似度阈值

    # ──────────────────────────── 1. 追问类型分类 ────────────────────────────

    async def classify_follow_up(self, query: str, session: SessionMemory) -> FollowUpType:
        """
        判断用户追问类型。
        规则 + LLM 混合策略：先用规则快速过滤，再用LLM确认。
        """
        # 规则层：快速识别格式型
        format_keywords = ["表格", "总结", "格式化", "markdown", "用列表", "重新排版"]
        if any(k in query for k in format_keywords) and session.evidence_cache:
            return FollowUpType.FORMAT

        # 规则层：识别话题切换（新实体/新时间/新领域）
        if session.working_memory.turns:
            new_entities = self._extract_new_entities(query, session.long_term)
            if new_entities:
                return FollowUpType.EXTEND

        # LLM层：语义分类
        if not session.working_memory.turns:
            return FollowUpType.FIRST

        recent_context = session.working_memory.get_recent_context(2)
        prompt = f"""请判断用户的新问题是属于哪种追问类型：

上下文：
{recent_context}

新问题：{query}

类型定义：
- expand（展开型）：在原证据基础上深挖细节，如"具体怎么做"、"详细说说"
- extend（扩展型）：引入新信息、新时间、新实体，如"那2024年呢"、"对比B公司"
- format（格式型）：仅改变输出格式，如"用表格总结"、"列个清单"

请只输出一个单词：expand / extend / format"""

        try:
            if self.llm is None:
                self.llm = LLMClient.from_config("planner")
            result = await self.llm.generate(prompt=prompt, temperature=0.1, max_tokens=10, timeout=TIMEOUT_LIGHT)
            result = result.strip().lower()

            if "format" in result:
                return FollowUpType.FORMAT
            elif "extend" in result:
                return FollowUpType.EXTEND
            else:
                return FollowUpType.EXPAND
        except Exception as e:
            logger.warning(f"[RetrievalDecision] 追问分类失败，默认expand: {e}")
            return FollowUpType.EXPAND

    # ──────────────────────────── 2. 旧证据rerank ────────────────────────────

    async def rerank_old_evidence(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """
        用改写后的query对旧证据重新rerank。
        返回按分数排序的chunks，每个chunk附加score字段。
        """
        if not chunks:
            return []

        try:
            # 使用现有 CrossEncoder reranker
            candidates = [{"content": c.get("content", "")} for c in chunks]
            ranked = await reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=len(candidates),
            )

            scored = []
            for orig_idx, score in ranked:
                chunk = dict(chunks[orig_idx])
                chunk["_rerank_score"] = score
                scored.append(chunk)

            # 未rerank到的补0分
            ranked_indices = {idx for idx, _ in ranked}
            for i, c in enumerate(chunks):
                if i not in ranked_indices:
                    chunk = dict(c)
                    chunk["_rerank_score"] = 0.0
                    scored.append(chunk)

            scored.sort(key=lambda x: x["_rerank_score"], reverse=True)
            return scored
        except Exception as e:
            logger.warning(f"[RetrievalDecision] 旧证据rerank失败: {e}")
            # fallback：直接返回原chunks，给默认分数
            return [{**c, "_rerank_score": 0.5} for c in chunks]

    # ──────────────────────────── 3. 主决策流程 ────────────────────────────

    async def decide(self, query: str, session: SessionMemory) -> RetrievalDecision:
        """
        完整决策流程：
        用户追问 → 类型分类 → 旧证据rerank → 决策输出
        """
        follow_type = await self.classify_follow_up(query, session)

        # 首轮对话：全新检索
        if follow_type == FollowUpType.FIRST or not session.evidence_cache:
            return RetrievalDecision(
                action=RetrievalAction.FULL,
                reason="首轮对话或无可复用证据，执行全新检索",
                incremental_query=query,
            )

        # 格式型：无需检索，直接复用已有上下文
        if follow_type == FollowUpType.FORMAT:
            return RetrievalDecision(
                action=RetrievalAction.NO_RETRIEVAL,
                reason="格式型追问，仅需重新排版已有证据",
                reused_evidence=session.evidence_cache[:5],
            )

        # 展开型 & 扩展型：先rerank旧证据
        reranked = await self.rerank_old_evidence(query, session.evidence_cache)
        top_score = reranked[0].get("_rerank_score", 0) if reranked else 0

        # 展开型：rerank分数够高 → 直接复用旧证据
        if follow_type == FollowUpType.EXPAND and top_score >= self.threshold:
            return RetrievalDecision(
                action=RetrievalAction.REUSE,
                reason=f"展开型追问，旧证据rerank分数({top_score:.2f})高于阈值，直接复用",
                reused_evidence=reranked[:5],
                threshold_score=top_score,
            )

        # 扩展型 或 展开型但分数不足 → 增量检索
        kept_old = [c for c in reranked if c.get("_rerank_score", 0) > 0.5][:3]

        return RetrievalDecision(
            action=RetrievalAction.INCREMENTAL,
            reason=f"{'扩展型追问' if follow_type == FollowUpType.EXTEND else '展开型但旧证据不足'}，"
                   f"rerank分数({top_score:.2f})，执行增量检索并合并证据",
            incremental_query=query,
            reused_evidence=kept_old,
        )

    def _extract_new_entities(self, query: str, long_term: Optional[LongTermMemory]) -> List[str]:
        """简易新实体检测（实际可替换为NER工具）"""
        # 占位：可用LLM或规则提取
        return []
