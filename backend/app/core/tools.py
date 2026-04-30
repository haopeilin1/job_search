"""
Agent 工具集合 —— 意图识别后可调用的工具定义

召回策略：固定使用混合召回（70%向量 + 30%BM25，top-15）。
其余召回方式代码保留在 scripts/test_retrieval.py 中供对比实验，但业务层只用混合召回。
"""

import hashlib
import json
import logging
import re
import time
from typing import Optional, Any, List, Dict
from dataclasses import dataclass, field

from app.core.llm_client import LLMClient, TIMEOUT_STANDARD, TIMEOUT_HEAVY, TIMEOUT_LIGHT
from app.core.tool_registry import BaseTool, ToolResult as NewToolResult
from app.core.config import settings

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════
# 0. 工具结果缓存（简单 LRU，控制内存占用）
# ═══════════════════════════════════════════════════════

class _SimpleCache:
    """简单固定容量的 LRU 缓存"""
    def __init__(self, maxsize: int = 128):
        self._data: Dict[str, Any] = {}
        self._order: List[str] = []
        self._maxsize = maxsize

    def get(self, key: str) -> Optional[Any]:
        if key in self._data:
            # move to end (most recently used)
            self._order.remove(key)
            self._order.append(key)
            return self._data[key]
        return None

    def set(self, key: str, value: Any):
        if key in self._data:
            self._order.remove(key)
        elif len(self._order) >= self._maxsize:
            # evict oldest
            oldest = self._order.pop(0)
            del self._data[oldest]
        self._order.append(key)
        self._data[key] = value

# match_analyze 结果缓存（key = hash(resume+jd+company+position)）
_match_analyze_cache = _SimpleCache(maxsize=64)


# ═══════════════════════════════════════════════════════
# 1. 工具数据模型
# ═══════════════════════════════════════════════════════

@dataclass
class ToolCall:
    """一次工具调用描述"""
    name: str
    parameters: dict = field(default_factory=dict)
    result: Any = None


@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    data: Any = None
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════
# 2. 工具定义（名称 + 描述 + 参数 schema）
# ═══════════════════════════════════════════════════════

TOOL_DEFINITIONS = {
    "kb_retrieve": {
        "name": "kb_retrieve",
        "description": "从知识库中检索与查询相关的 JD chunks（混合召回：70%向量 + 30%BM25）。",
        "parameters": {
            "query": {"type": "string", "description": "检索查询文本", "required": True},
            "company": {"type": "string", "description": "可选：按公司名过滤", "required": False},
            "position": {"type": "string", "description": "可选：按岗位名过滤", "required": False},
            "top_k": {"type": "integer", "description": "返回 top_k 个相关 chunk", "default": settings.RETRIEVAL_TOP_K},
        },
    },
    "match_analyze": {
        "name": "match_analyze",
        "description": "分析简历与 JD 的匹配度。"
                       "输入简历文本和 JD 文本，输出："
                       "(1) 匹配分数 0-100；"
                       "(2) 匹配等级（高度契合/基本匹配/略有差距/不匹配）；"
                       "(3) 核心优势（简历中与 JD 对齐的 2-3 点）；"
                       "(4) 潜在短板（相对 JD 不足的 1-2 点）；"
                       "(5) 面试准备建议。",
        "parameters": {
            "resume_text": {"type": "string", "description": "用户简历文本", "required": True},
            "jd_text": {"type": "string", "description": "岗位描述文本", "required": True},
            "company": {"type": "string", "description": "可选：公司名称", "required": False},
            "position": {"type": "string", "description": "可选：岗位名称", "required": False},
        },
    },
    "interview_questions": {
        "name": "interview_questions",
        "description": "根据简历与 JD 的匹配结果，生成针对性面试题。",
        "parameters": {
            "match_result": {"type": "object", "description": "match_analyze 的输出结果", "required": True},
            "company": {"type": "string", "description": "公司名称", "required": False},
            "position": {"type": "string", "description": "岗位名称", "required": False},
        },
    },
}


# ═══════════════════════════════════════════════════════
# 3. BM25 索引（全局单例，懒加载）
# ═══════════════════════════════════════════════════════

_bm25_index = None
_bm25_ids: List[str] = []
_bm25_docs: List[str] = []
_bm25_metadatas: List[dict] = []


def _get_persist_dir() -> str:
    from pathlib import Path
    return str(Path(__file__).resolve().parent.parent.parent / "data" / "chroma_db")


def _build_bm25_index():
    """从 ChromaDB 加载全部文档，构建 BM25 索引（只执行一次）"""
    global _bm25_index, _bm25_ids, _bm25_docs, _bm25_metadatas
    if _bm25_index is not None:
        return

    import jieba
    from rank_bm25 import BM25Okapi
    import chromadb

    persist_dir = _get_persist_dir()
    client = chromadb.PersistentClient(path=persist_dir)
    coll = client.get_collection("jd_knowledge")

    results = coll.get(include=["documents", "metadatas"])
    _bm25_ids = results["ids"]
    _bm25_docs = results["documents"]
    _bm25_metadatas = results["metadatas"]

    # jieba 分词
    tokenized_docs = []
    for doc in _bm25_docs:
        if doc:
            tokens = list(jieba.cut_for_search(doc))
            tokenized_docs.append(tokens)
        else:
            tokenized_docs.append([])

    _bm25_index = BM25Okapi(tokenized_docs)
    logger.info(f"[BM25] 索引构建完成 | docs={len(_bm25_ids)}")


# ═══════════════════════════════════════════════════════
# 4. 混合召回实现（向量 + BM25，两路独立召回后融合）
# ═══════════════════════════════════════════════════════

async def _kb_retrieve_stub(query: str, company: Optional[str] = None,
                            position: Optional[str] = None, top_k: int = None) -> ToolResult:
    """
    知识库混合召回工具（含 CrossEncoder 重排序）：
      1) 向量路独立召回 top-20
      2) BM25 路独立召回 top-20
      3) 合并去重（chunk_id 为 key）
      4) 两路分数分别 min-max 归一化到 [0,1]
      5) 加权混合：hybrid = 0.70×vec_norm + 0.30×bm25_norm
      6) 按 hybrid_score 降序，取 top-20 作为重排序候选池
      7) CrossEncoder 重排序，输出 top-10
    """
    logger.info(f"[Tool:kb_retrieve] query='{query[:40]}...' company={company} position={position}")

    # 防御：Planner 可能将 top_k 生成为字符串
    if top_k is not None:
        try:
            top_k = int(top_k)
        except (ValueError, TypeError):
            top_k = None

    try:
        import numpy as np
        import jieba
        from app.core.vector_store import VectorStore
        from app.core.embedding import EmbeddingClient
        from app.core.config import settings

        # ── 1. 向量检索 top-20 ──
        vs = VectorStore()
        vs.embedding_client = EmbeddingClient.from_config()
        filters = {}
        if company:
            filters["company"] = company
        if position:
            filters["position"] = position

        vec_results = await vs.query(
            query,
            filters=filters if filters else None,
            top_k=settings.RETRIEVAL_VEC_TOP_K,
        )

        # ── 2. BM25 检索 top-20 ──
        _build_bm25_index()
        query_tokens = list(jieba.cut_for_search(query))
        bm25_scores = _bm25_index.get_scores(query_tokens)  # numpy array, len = 总文档数

        # 取 BM25 top-20（分数 > 0 的才要）
        # 防御：确保 bm25_scores 是 numpy array
        if not isinstance(bm25_scores, np.ndarray):
            bm25_scores = np.array(bm25_scores)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:settings.RETRIEVAL_BM25_TOP_K]
        bm25_top_indices = np.asarray(bm25_top_indices, dtype=int)
        bm25_results = []
        for idx in bm25_top_indices:
            if bm25_scores[idx] > 0:
                bm25_results.append({
                    "chunk_id": _bm25_ids[idx],
                    "content": _bm25_docs[idx],
                    "metadata": _bm25_metadatas[idx],
                    "bm25_score": float(bm25_scores[idx]),
                })

        # ── 3. 合并去重（chunk_id 为 key）──
        pool: dict[str, dict] = {}

        # 放入向量结果
        for r in vec_results:
            cid = r["chunk_id"]
            pool[cid] = {
                "chunk_id": cid,
                "content": r["content"],
                "metadata": r["metadata"],
                "distance": r["distance"],      # 原始向量 distance
                "bm25_score": 0.0,              # 默认 BM25 分数为 0
            }

        # 放入 BM25 结果（去重）
        for r in bm25_results:
            cid = r["chunk_id"]
            if cid in pool:
                pool[cid]["bm25_score"] = r["bm25_score"]
            else:
                pool[cid] = {
                    "chunk_id": cid,
                    "content": r["content"],
                    "metadata": r["metadata"],
                    "distance": None,           # 向量未召回此项
                    "bm25_score": r["bm25_score"],
                }

        items = list(pool.values())
        total_pool = len(items)

        # ── 4. 向量分数归一化（min-max → [0,1]）──
        vec_items = [it for it in items if it["distance"] is not None]
        if vec_items:
            distances = [it["distance"] for it in vec_items]
            d_min, d_max = min(distances), max(distances)
            d_range = d_max - d_min if d_max != d_min else 1e-9
            for it in vec_items:
                # distance 越小越相似 → 翻转
                it["vec_norm"] = 1.0 - (it["distance"] - d_min) / d_range
        for it in items:
            if "vec_norm" not in it:
                it["vec_norm"] = 0.0

        # ── 5. BM25 分数归一化（min-max → [0,1]）──
        bm25_items = [it for it in items if it["bm25_score"] > 0]
        if bm25_items:
            b_min = min(it["bm25_score"] for it in bm25_items)
            b_max = max(it["bm25_score"] for it in bm25_items)
            b_range = b_max - b_min if b_max != b_min else 1e-9
            for it in bm25_items:
                it["bm25_norm"] = (it["bm25_score"] - b_min) / b_range
        for it in items:
            if "bm25_norm" not in it:
                it["bm25_norm"] = 0.0

        # ── 6. 加权混合 ──
        from app.core.config import settings
        VEC_W, BM25_W = settings.RETRIEVAL_VEC_WEIGHT, settings.RETRIEVAL_BM25_WEIGHT
        for it in items:
            it["hybrid_score"] = VEC_W * it["vec_norm"] + BM25_W * it["bm25_norm"]

        # ── 6.5 内容级去重：相同 content 只保留 hybrid 分数最高的一条 ──
        seen_content = {}
        for it in items:
            content = it["content"].strip()
            if content not in seen_content or seen_content[content]["hybrid_score"] < it["hybrid_score"]:
                seen_content[content] = it
        items = list(seen_content.values())
        total_pool = len(items)  # 更新去重后的 pool 大小

        # ── 7. 排序取 top-k（作为重排序候选池）──
        items.sort(key=lambda x: x["hybrid_score"], reverse=True)
        final_top_k = top_k if top_k is not None else settings.RETRIEVAL_TOP_K
        top_results = items[:final_top_k]

        # ── 8. CrossEncoder 重排序 ──
        reranked_results = top_results
        rerank_info = {"enabled": False}
        if settings.RERANKER_ENABLED:
            try:
                from app.core import reranker
                ranked = await reranker.rerank(
                    query=query,
                    candidates=top_results,
                    top_k=settings.RERANKER_TOP_K,
                    batch_size=settings.RERANKER_BATCH_SIZE,
                    max_length=settings.RERANKER_MAX_LENGTH,
                )
                reranked_results = []
                for orig_idx, score in ranked:
                    item = top_results[orig_idx].copy()
                    item["rerank_score"] = round(score, 4)
                    reranked_results.append(item)
                rerank_info = {
                    "enabled": True,
                    "candidates": len(top_results),
                    "output": len(reranked_results),
                    "best_rerank_score": round(ranked[0][1], 4) if ranked else 0,
                }
                logger.info(
                    f"[Tool:kb_retrieve] 重排序完成 | candidates={len(top_results)} -> top={len(reranked_results)} | "
                    f"best_rerank_score={ranked[0][1]:.4f}"
                )
            except Exception as e:
                logger.warning(f"[Tool:kb_retrieve] 重排序失败，fallback 到混合分数: {e}")
                rerank_info = {"enabled": True, "error": str(e), "fallback": True}

        vec_only = sum(1 for it in items if it["distance"] is not None and it["bm25_score"] == 0)
        bm25_only = sum(1 for it in items if it["distance"] is None and it["bm25_score"] > 0)
        both = sum(1 for it in items if it["distance"] is not None and it["bm25_score"] > 0)

        best_hybrid_str = f"{top_results[0]['hybrid_score']:.4f}" if top_results else "N/A"
        logger.info(
            f"[Tool:kb_retrieve] 混合召回完成 | "
            f"pool={total_pool}(vec_only={vec_only}, bm25_only={bm25_only}, both={both}) -> top={len(reranked_results)} | "
            f"best_hybrid={best_hybrid_str}"
        )

        return ToolResult(
            success=True,
            data={
                "chunks": reranked_results,
                "query": query,
                "filters": filters,
                "strategy": "hybrid(70%vec+30%bm25)+reranker",
                "pool_size": total_pool,
                "vec_only": vec_only,
                "bm25_only": bm25_only,
                "both": both,
                "rerank": rerank_info,
            },
        )

    except Exception as e:
        logger.error(f"[Tool:kb_retrieve] 混合召回失败: {e}")
        return ToolResult(success=False, error=str(e))


async def _match_analyze(resume_text: str, jd_text: str,
                             company: Optional[str] = None,
                             position: Optional[str] = None) -> ToolResult:
    """
    简历与 JD 匹配度分析工具（LLM 驱动）。

    调用 LLM 对简历和 JD 进行深度匹配分析，输出结构化 JSON。
    """
    logger.info(f"[Tool:match_analyze] company={company} position={position}")

    if not resume_text or not resume_text.strip():
        return ToolResult(success=False, error="resume_text 为空，无法分析匹配度")
    if not jd_text or not jd_text.strip():
        return ToolResult(success=False, error="jd_text 为空，无法分析匹配度")

    # ── 缓存检查 ──
    cache_key = hashlib.md5(
        f"{resume_text[:500]}|{jd_text[:500]}|{company or ''}|{position or ''}".encode("utf-8")
    ).hexdigest()
    cached = _match_analyze_cache.get(cache_key)
    if cached is not None:
        logger.info(f"[Tool:match_analyze] 缓存命中 | score={cached.get('score', 0)}")
        return ToolResult(success=True, data=cached)

    system_prompt = """# 角色定义
你是一位资深HR专家和招聘顾问，拥有10年以上的简历评估和人才匹配经验。你擅长从多维度分析候选人与职位的匹配程度，能够准确识别候选人的优势与短板。

# 任务目标
你的任务是对比分析候选人简历与职位描述（JD），给出全面的匹配度评估，列出候选人已覆盖的能力项，并识别候选人缺乏的关键经验和能力。

# 工作流上下文
- **Input**：候选人简历的文本内容、职位描述（JD）的文本内容
- **Process**：
  1. 深度分析JD中的核心要求（技能、经验、教育背景等）
  2. 逐项对比简历中的实际经历和能力
  3. 计算综合匹配度（0-100分），考虑以下维度：
     - 技术技能匹配度（权重30%）
     - 项目经验匹配度（权重30%）
     - 教育背景匹配度（权重10%）
     - 工作年限匹配度（权重20%）
     - 其他软技能匹配度（权重10%）
  4. 提炼出JD要求的核心能力清单
  5. 列出候选人已覆盖的能力项
  6. 识别候选人明显缺乏或不满足要求的关键经验和能力
- **Output**：结构化JSON对象，包含匹配度打分、缺乏经验描述和核心能力清单

# 约束与规则
- 匹配度评分必须客观、公正，避免主观臆断
- 缺乏经验要具体明确，避免模糊描述
- 核心能力要从JD中提炼，要包含技能、经验、素质等多方面
- 输出必须是纯JSON格式，不能包含任何Markdown标记或额外说明文字
- 如果简历信息不完整，在缺失维度给予较低评分，并说明原因

# 输出格式
仅返回如下格式的JSON对象：
{
  "match_score": 0到100之间的数字,
  "covered": ["技能1", "技能2"],
  "missing_experience": "候选人缺乏的关键经验和能力的详细描述",
  "core_skills": "从JD中提炼的核心能力要求清单"
}"""

    user_prompt = f"""【候选人简历】
{resume_text[:2000]}

【职位描述（JD）】
{jd_text[:2000]}

请进行匹配分析，严格按要求的 JSON 格式输出："""

    try:
        from app.core.llm_client import LLMClient
        # match_analyze 默认用 turbo（更快更便宜），质量对求职场景足够
        llm = LLMClient.from_config("memory")
        t0 = time.time()
        raw = await llm.generate(
            prompt=user_prompt,
            system=system_prompt,
            temperature=0.3,
            max_tokens=1500,
            timeout=20.0,  # turbo 通常 2-5s 完成，20s 足够容错
        )
        main_latency = time.time() - t0
        logger.info(f"[Tool:match_analyze] 主调用耗时 {main_latency:.2f}s")

        # 解析 JSON
        import json
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```json\s*|\s*```$", "", text).strip()

        data = json.loads(text)

        match_score = int(data.get("match_score", 0))
        covered = data.get("covered", []) or []
        missing = data.get("missing_experience", "")
        core_skills = data.get("core_skills", "")

        # 映射为系统内部格式 + 保留原始字段
        result_data = {
            # 兼容旧格式
            "score": match_score,
            "label": _score_to_label(match_score),
            "advantages": covered,
            "weaknesses": [missing] if missing else [],
            "suggestions": [f"核心能力要求: {core_skills}"] if core_skills else [],
            # 原始分析字段
            "match_score": match_score,
            "covered": covered,
            "missing_experience": missing,
            "core_skills": core_skills,
            "company": company,
            "position": position,
        }

        logger.info(f"[Tool:match_analyze] 分析完成 | score={match_score} | label={result_data['label']}")
        _match_analyze_cache.set(cache_key, result_data)
        return ToolResult(success=True, data=result_data)

    except Exception as e:
        logger.warning(f"[Tool:match_analyze] turbo 调用失败: {e}，返回静态降级")
        # turbo 已失败，直接返回静态降级（不再尝试 core，避免额外延迟）
        return ToolResult(
            success=True,
            data={
                "score": 0,
                "label": "无法评估",
                "advantages": [],
                "weaknesses": [f"分析失败: {e}"],
                "suggestions": ["请稍后重试或手动检查简历/JD内容"],
                "match_score": 0,
                "covered": [],
                "missing_experience": "分析失败",
                "core_skills": "",
                "company": company,
                "position": position,
            },
        )


def _score_to_label(score: int) -> str:
    """根据匹配分数返回标签"""
    if score >= 85:
        return "高度契合"
    if score >= 70:
        return "基本匹配"
    if score >= 50:
        return "略有差距"
    return "不匹配"


async def _evidence_relevance_check(query: str, evidence_chunks: List[Dict]) -> ToolResult:
    """
    检查缓存证据与用户查询的相关性。

    取 top EVIDENCE_CACHE_MAX_SIZE 条 chunk，调用轻量 LLM 判断相关性。
    """
    logger.info(f"[Tool:evidence_relevance_check] query='{query[:40]}...' chunks={len(evidence_chunks)}")

    if not evidence_chunks:
        return ToolResult(
            success=True,
            data={"relevant": False, "confidence": 0.0, "reason": "无缓存证据"},
        )

    max_chunks = getattr(settings, "EVIDENCE_CACHE_MAX_SIZE", 5)
    chunks = evidence_chunks[:max_chunks]

    # 构建 chunk 摘要
    summaries = []
    for i, chunk in enumerate(chunks):
        content = chunk.get("content", "") or chunk.get("text", "")
        summaries.append(f"[{i+1}] {content[:200]}")

    system_prompt = """你是一个证据相关性判断助手。请判断给定的证据片段是否与用户查询相关。

输出严格JSON格式：
{
  "relevant": true/false,
  "confidence": 0.0-1.0,
  "reason": "判断理由"
}

规则：
- 如果证据能直接回答或高度相关查询，relevant=true
- 如果证据与查询主题无关，relevant=false
- confidence 表示你的确信程度
- 必须返回纯JSON，不要markdown代码块"""

    summaries_text = "\n".join(summaries)
    user_prompt = f"""【用户查询】
{query}

【证据片段】
{summaries_text}

请判断相关性，输出JSON："""

    try:
        llm = LLMClient.from_config("memory")
        raw = await llm.generate(
            prompt=user_prompt,
            system=system_prompt,
            timeout=TIMEOUT_LIGHT,
            max_tokens=100,
            temperature=0.1,
        )

        import json
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```json\s*|\s*```$", "", text).strip()

        data = json.loads(text)

        result_data = {
            "relevant": bool(data.get("relevant", False)),
            "confidence": float(data.get("confidence", 0.0)),
            "reason": str(data.get("reason", "")),
            "chunks_checked": len(chunks),
        }

        # 与阈值比较
        threshold = getattr(settings, "EVIDENCE_CACHE_RELEVANCE_THRESHOLD", 0.6)
        if result_data["confidence"] < threshold:
            result_data["relevant"] = False
            if not result_data["reason"]:
                result_data["reason"] = f"置信度({result_data['confidence']:.2f})低于阈值({threshold})"

        logger.info(
            f"[Tool:evidence_relevance_check] 完成 | relevant={result_data['relevant']} | "
            f"confidence={result_data['confidence']:.2f}"
        )
        return ToolResult(success=True, data=result_data)

    except Exception as e:
        logger.error(f"[Tool:evidence_relevance_check] 执行失败: {e}")
        # 保守 fallback：假设不相关
        return ToolResult(
            success=False,
            error=str(e),
            data={"relevant": False, "confidence": 0.0, "reason": f"判断失败: {e}"},
        )


async def _interview_questions(match_result: dict,
                                 company: Optional[str] = None,
                                 position: Optional[str] = None) -> ToolResult:
    """
    面试题生成工具（LLM 驱动）。

    根据匹配分析结果（候选人短板 + JD 核心能力），生成 5 道有深度的模拟面试题。
    """
    logger.info(f"[Tool:interview_questions] company={company} position={position}")

    if not match_result or not isinstance(match_result, dict):
        return ToolResult(success=False, error="match_result 为空或格式错误")

    missing = match_result.get("missing_experience", "")
    core_skills = match_result.get("core_skills", "")
    covered = match_result.get("covered", [])
    score = match_result.get("match_score", 0)

    system_prompt = """# 角色定义
你是一位资深技术面试官，拥有丰富的面试经验和深度的技术理解力。你擅长设计刁钻但有深度、能够真实考察候选人能力的面试题目。

# 任务目标
你的任务是根据候选人缺乏的经验和JD要求的核心能力，设计5道具有挑战性的模拟面试题，每道题都包含1-2个可能的追问方向。

# 工作流上下文
- **Input**：候选人缺乏的经验描述、JD要求的核心能力清单
- **Process**：
  1. 深入理解候选人的短板和职位的核心要求
  2. 针对每个关键短板，设计1-2道深入追问的问题
  3. 问题设计原则：
     - 避免简单的概念性、记忆性问题
     - 侧重考察实际应用能力、问题解决能力、系统设计能力
     - 可以包含场景题、设计题、代码分析题、架构题、技术理解等多种形式
     - 问题要有一定的深度和难度，能够区分优秀候选人和普通候选人
  4. 每道面试题必须包含：
     - **主体问题**：完整的面试题内容
     - **追问方向**：1-2个面试官可以进一步追问的方向（用「追问方向：」标识）
  5. 确保生成5道高质量面试题，覆盖不同的能力维度
- **Output**：结构化JSON对象，包含5道面试题的字符串数组

# 约束与规则
- 问题要具体、明确，避免过于抽象
- 问题要具有实际意义，能够真实反映候选人在工作中的表现
- 每道问题都要有明确的考察点
- 每道问题必须包含1-2个追问方向，用「追问方向：」清晰标识
- 追问方向要具有层次性，可以继续深挖候选人的思考和表达能力
- 输出必须是纯JSON格式，不能包含任何Markdown标记或额外说明文字
- 问题长度适中，既要保证深度，又要便于面试官提问
- interview_questions必须是字符串数组，每个元素是一道完整的面试题（包含题目和追问）

# 输出格式
仅返回如下格式的JSON对象：
{
  "interview_questions": [
    "面试题主体内容。追问方向：1）追问方向一；2）追问方向二",
    "第二道面试题主体内容。追问方向：追问方向一",
    "第三道面试题主体内容。追问方向：1）追问方向一；2）追问方向二",
    "第四道面试题主体内容。追问方向：追问方向一",
    "第五道面试题主体内容。追问方向：1）追问方向一；2）追问方向二"
  ]
}"""

    user_prompt = f"""【候选人匹配分析结果】
- 匹配分数：{score}/100
- 已覆盖能力：{', '.join(covered) if covered else '无'}
- 缺乏的经验：{missing or '（未明确）'}
- JD 核心能力要求：{core_skills or '（未明确）'}
{ f"- 目标公司：{company}" if company else "" }
{ f"- 目标岗位：{position}" if position else "" }

请基于以上信息生成 5 道针对性面试题，严格按 JSON 格式输出："""

    try:
        from app.core.llm_client import LLMClient
        llm = LLMClient.from_config("core")
        t0 = time.time()
        raw = await llm.generate(
            prompt=user_prompt,
            system=system_prompt,
            temperature=0.4,
            max_tokens=2000,
            timeout=30.0,  # 30s，面试题生成标准调用
        )
        main_latency = time.time() - t0
        logger.info(f"[Tool:interview_questions] 主调用耗时 {main_latency:.2f}s")

        import json
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```json\s*|\s*```$", "", text).strip()

        data = json.loads(text)
        questions = data.get("interview_questions", []) or []

        # 验证必须为 5 道
        if len(questions) < 5:
            logger.warning(f"[Tool:interview_questions] LLM 只返回了 {len(questions)} 道题，补充到 5 道")
            while len(questions) < 5:
                questions.append(f"[{len(questions)+1}] 请候选人结合自身经历，谈谈对 {core_skills or '该岗位核心技能'} 的理解与实践。追问方向：1）在实际项目中遇到过什么挑战；2）如果重来一次会怎么改进。")

        # 提取关注维度（从缺失能力和核心能力中提炼）
        focus_areas = []
        if missing:
            focus_areas.append(f"短板深挖：{missing[:100]}")
        if core_skills:
            focus_areas.append(f"核心能力：{core_skills[:100]}")
        if not focus_areas:
            focus_areas = ["综合技术能力考察"]

        result_data = {
            "questions": questions[:5],
            "focus_areas": focus_areas,
            "question_count": len(questions[:5]),
            "company": company,
            "position": position,
        }

        logger.info(f"[Tool:interview_questions] 生成完成 | {result_data['question_count']} 道题 | focus={focus_areas}")
        return ToolResult(success=True, data=result_data)

    except Exception as e:
        logger.warning(f"[Tool:interview_questions] 主调用(core/qwen-plus)失败: {e}")
        # fallback-1：降级到 turbo（qwen-turbo），更快更便宜
        try:
            fallback_prompt = f"""生成5道面试题(JSON)：{{"interview_questions":["题1","题2","题3","题4","题5"]}}
已覆盖：{', '.join(covered) if covered else '无'}
短板：{missing or '未明确'}
核心能力：{core_skills or '未明确'}"""
            t1 = time.time()
            turbo_llm = LLMClient.from_config("memory")
            raw = await turbo_llm.generate(
                prompt=fallback_prompt,
                system="技术面试官，快速生成5道面试题，每题包含追问方向。",
                temperature=0.4,
                max_tokens=1000,
                timeout=10.0,
            )
            fallback_latency = time.time() - t1
            logger.info(f"[Tool:interview_questions] turbo fallback 耗时 {fallback_latency:.2f}s")
            text = raw.strip()
            if text.startswith("```"):
                text = re.sub(r"^```json\s*|\s*```$", "", text).strip()
            questions = []
            try:
                data = json.loads(text)
                questions = data.get("interview_questions", []) or []
            except Exception:
                lines = [l.strip() for l in text.split("\n") if l.strip() and len(l.strip()) > 10]
                questions = lines[:5]
            while len(questions) < 5:
                questions.append(f"[{len(questions)+1}] 请候选人结合自身经历，谈谈对 {core_skills or '该岗位核心技能'} 的理解与实践。追问方向：1）在实际项目中遇到过什么挑战；2）如果重来一次会怎么改进。")
            focus_areas = []
            if missing:
                focus_areas.append(f"短板深挖：{missing[:100]}")
            if core_skills:
                focus_areas.append(f"核心能力：{core_skills[:100]}")
            if not focus_areas:
                focus_areas = ["综合技术能力考察"]
            result_data = {
                "questions": questions[:5],
                "focus_areas": focus_areas,
                "question_count": len(questions[:5]),
                "company": company,
                "position": position,
            }
            logger.info(f"[Tool:interview_questions] turbo fallback 成功 | {result_data['question_count']} 道题")
            return ToolResult(success=True, data=result_data)
        except Exception as fallback_err:
            logger.error(f"[Tool:interview_questions] turbo fallback 也失败: {fallback_err}")
            # 最终降级：返回通用模板题
            return ToolResult(
                success=True,
                data={
                    "questions": [
                        "[Fallback] 请介绍你在相关领域的项目经验。追问方向：1）技术选型理由；2）遇到的最大挑战。",
                        "[Fallback] 请谈谈你对该岗位核心技术的理解。追问方向：1）实际应用场景；2）优缺点分析。",
                        "[Fallback] 描述一次你解决复杂问题的经历。追问方向：1）思路过程；2）最终效果。",
                        "[Fallback] 如果你来设计这个系统，你会怎么做？追问方向：1）架构选型；2）扩展性考虑。",
                        "[Fallback] 请分享一次团队协作的经验。追问方向：1）角色分工；2）冲突解决。",
                    ],
                    "focus_areas": ["通用技术能力", "项目经验"],
                    "question_count": 5,
                    "company": company,
                    "position": position,
                },
            )


# 工具名称 -> handler 映射
TOOL_HANDLERS = {
    "kb_retrieve": _kb_retrieve_stub,
    "match_analyze": _match_analyze,
    "interview_questions": _interview_questions,
    "evidence_relevance_check": _evidence_relevance_check,
}


# ═══════════════════════════════════════════════════════
# 5. 工具执行入口
# ═══════════════════════════════════════════════════════

async def execute_tool(call: ToolCall) -> ToolResult:
    """根据 ToolCall 执行对应的工具 handler"""
    handler = TOOL_HANDLERS.get(call.name)
    if not handler:
        return ToolResult(success=False, error=f"未知工具: {call.name}")

    try:
        result = await handler(**call.parameters)
        call.result = result
        return result
    except Exception as e:
        logger.error(f"[Tool:{call.name}] 执行失败: {e}")
        return ToolResult(success=False, error=str(e))


# ═══════════════════════════════════════════════════════
# 6. BaseTool 子类封装（供 LLM 路线使用）
# ═══════════════════════════════════════════════════════

class KBRetrieveTool(BaseTool):
    """知识库混合检索工具"""

    @property
    def name(self) -> str:
        return "kb_retrieve"

    @property
    def input_schema(self) -> dict:
        from app.core.tool_registry import KB_RETRIEVE_INPUT_SCHEMA
        return KB_RETRIEVE_INPUT_SCHEMA

    @property
    def output_schema(self) -> dict:
        from app.core.tool_registry import KB_RETRIEVE_OUTPUT_SCHEMA
        return KB_RETRIEVE_OUTPUT_SCHEMA

    @property
    def cost_level(self) -> str:
        return "medium"

    @property
    def avg_latency_ms(self) -> int:
        return 800

    async def execute(self, params: dict) -> NewToolResult:
        result = await _kb_retrieve_stub(
            query=params.get("query", ""),
            company=params.get("company"),
            position=params.get("position"),
            top_k=params.get("top_k", 15),
        )
        return NewToolResult(
            success=result.success,
            data=result.data or {},
            error=result.error,
        )


class MatchAnalyzeTool(BaseTool):
    """单JD匹配分析工具"""

    @property
    def name(self) -> str:
        return "match_analyze"

    @property
    def input_schema(self) -> dict:
        from app.core.tool_registry import MATCH_ANALYZE_INPUT_SCHEMA
        return MATCH_ANALYZE_INPUT_SCHEMA

    @property
    def output_schema(self) -> dict:
        from app.core.tool_registry import MATCH_ANALYZE_OUTPUT_SCHEMA
        return MATCH_ANALYZE_OUTPUT_SCHEMA

    @property
    def cost_level(self) -> str:
        return "high"

    @property
    def avg_latency_ms(self) -> int:
        return 1500

    async def execute(self, params: dict) -> NewToolResult:
        jd_text = params.get("jd_text", "")
        # 防御：如果 jd_text 是 chunks 列表，按 company/position 筛选后拼接
        if isinstance(jd_text, list):
            chunks = jd_text
            company = params.get("company")
            position = params.get("position")
            if company or position:
                filtered = []
                for c in chunks:
                    if isinstance(c, dict):
                        meta = c.get("metadata", {})
                        c_company = meta.get("company", "")
                        c_position = meta.get("position", "")
                        if (not company or company in c_company) and (not position or position in c_position):
                            filtered.append(c)
                if filtered:
                    logger.info(f"[MatchAnalyzeTool] 按 company={company} position={position} 筛选，从 {len(chunks)} 个 chunk 中命中 {len(filtered)} 个")
                    chunks = filtered
            jd_text = "\n\n".join(
                c.get("content", "") if isinstance(c, dict) else str(c)
                for c in chunks
            )
            logger.info(f"[MatchAnalyzeTool] jd_text 为列表，已拼接为 {len(jd_text)} 字符")
        elif isinstance(jd_text, dict):
            jd_text = jd_text.get("content", "") or json.dumps(jd_text, ensure_ascii=False)

        result = await _match_analyze(
            resume_text=params.get("resume_text", ""),
            jd_text=jd_text,
            company=params.get("company"),
            position=params.get("position"),
        )
        return NewToolResult(
            success=result.success,
            data=result.data or {},
            error=result.error,
        )


class InterviewGenTool(BaseTool):
    """面试题生成工具"""

    @property
    def name(self) -> str:
        return "interview_gen"

    @property
    def input_schema(self) -> dict:
        from app.core.tool_registry import INTERVIEW_GEN_INPUT_SCHEMA
        return INTERVIEW_GEN_INPUT_SCHEMA

    @property
    def output_schema(self) -> dict:
        from app.core.tool_registry import INTERVIEW_GEN_OUTPUT_SCHEMA
        return INTERVIEW_GEN_OUTPUT_SCHEMA

    @property
    def cost_level(self) -> str:
        return "medium"

    @property
    def avg_latency_ms(self) -> int:
        return 1000

    async def execute(self, params: dict) -> NewToolResult:
        match_result = params.get("match_result", {})
        # 防御：match_result 可能被解析为字符串
        if isinstance(match_result, str):
            logger.warning(f"[InterviewGenTool] match_result 为字符串，尝试解析: {match_result[:200]}")
            try:
                match_result = json.loads(match_result)
            except Exception:
                match_result = {}
        if not isinstance(match_result, dict):
            match_result = {}
        result = await _interview_questions(
            match_result=match_result,
            company=match_result.get("company"),
            position=match_result.get("position"),
        )
        return NewToolResult(
            success=result.success,
            data=result.data or {},
            error=result.error,
        )


class EvidenceRelevanceCheckTool(BaseTool):
    """证据相关性检查工具"""

    @property
    def name(self) -> str:
        return "evidence_relevance_check"

    @property
    def input_schema(self) -> dict:
        from app.core.tool_registry import EVIDENCE_RELEVANCE_CHECK_INPUT_SCHEMA
        return EVIDENCE_RELEVANCE_CHECK_INPUT_SCHEMA

    @property
    def output_schema(self) -> dict:
        from app.core.tool_registry import EVIDENCE_RELEVANCE_CHECK_OUTPUT_SCHEMA
        return EVIDENCE_RELEVANCE_CHECK_OUTPUT_SCHEMA

    @property
    def cost_level(self) -> str:
        return "low"

    @property
    def avg_latency_ms(self) -> int:
        return 300

    async def execute(self, params: dict) -> NewToolResult:
        result = await _evidence_relevance_check(
            query=params.get("query", ""),
            evidence_chunks=params.get("evidence_chunks", []),
        )
        return NewToolResult(
            success=result.success,
            data=result.data or {},
            error=result.error,
        )


class GlobalRankTool(BaseTool):
    """全局匹配排序工具（简化版：调用 LLM）"""

    @property
    def name(self) -> str:
        return "global_rank"

    @property
    def input_schema(self) -> dict:
        from app.core.tool_registry import GLOBAL_RANK_INPUT_SCHEMA
        return GLOBAL_RANK_INPUT_SCHEMA

    @property
    def output_schema(self) -> dict:
        from app.core.tool_registry import GLOBAL_RANK_OUTPUT_SCHEMA
        return GLOBAL_RANK_OUTPUT_SCHEMA

    @property
    def cost_level(self) -> str:
        return "high"

    @property
    def avg_latency_ms(self) -> int:
        return 2000

    def _aggregate_chunks_by_jd(self, chunks: list) -> list:
        """
        将 chunk 列表按 jd_id 聚合成 JD 对象。
        输入: kb_retrieve 返回的 chunk 列表
        输出: [{jd_id, company, position, sections, structured_summary, hybrid_scores, ...}]
        """
        from collections import defaultdict
        jd_map = defaultdict(lambda: {
            "jd_id": "",
            "company": "未知",
            "position": "未知",
            "sections": defaultdict(list),
            "structured_summary": {},
            "hybrid_scores": [],
            "keywords": [],
        })

        for chunk in chunks:
            meta = chunk.get("metadata", {}) or {}
            jd_id = meta.get("jd_id", chunk.get("chunk_id", ""))
            jd = jd_map[jd_id]
            jd["jd_id"] = jd_id
            jd["company"] = meta.get("company", "未知")
            jd["position"] = meta.get("position", "未知")
            section = meta.get("section", "unknown")
            jd["sections"][section].append(chunk.get("content", ""))
            jd["hybrid_scores"].append(chunk.get("hybrid_score", 0))
            # 收集结构化摘要（优先从 basic_info chunk 的 metadata 读取）
            if section == "basic_info" and not jd["structured_summary"]:
                jd["structured_summary"] = {
                    k: meta.get(k)
                    for k in ["min_years", "max_years", "min_education", "category", "domain"]
                    if meta.get(k) is not None
                }
            # 如果是 keywords section，单独收集
            if section == "keywords":
                jd["keywords"].extend(chunk.get("content", "").split(","))

        # 转换为列表，按平均 hybrid_score 降序
        jd_list = []
        for jd in jd_map.values():
            avg_score = sum(jd["hybrid_scores"]) / len(jd["hybrid_scores"]) if jd["hybrid_scores"] else 0
            jd["avg_hybrid_score"] = avg_score
            jd_list.append(jd)

        jd_list.sort(key=lambda x: x["avg_hybrid_score"], reverse=True)
        return jd_list

    def _extract_resume_summary(self, resume_text: str) -> dict:
        """
        从简历文本中快速提取结构化信息（用于粗筛层）。
        基于正则，零成本，不调用 LLM。
        支持多种格式：技术技能/个人技能/产品工具/技术理解/项目经历中的技能描述。
        """
        import re

        result = {
            "hard_skills": [],
            "soft_skills": [],
            "years_of_experience": 0,
            "education": "",
            "domain": "",
        }
        text_lower = resume_text.lower()

        # ═══════════════════════════════════════════════════════
        # 1. 技能提取：多格式匹配
        # ═══════════════════════════════════════════════════════
        skill_sources = []

        # 1a. 传统 "技术技能：xxx" 格式
        m = re.search(r"技术技能[：:]\s*(.+?)(?:\n|$)", resume_text, re.IGNORECASE)
        if m:
            skill_sources.append(m.group(1))

        # 1b. "个人技能" / "专业技能" / "技能" 区块（匹配多行列表）
        m = re.search(r"(?:个人|专业)?技能[：:]\s*(.+?)(?=\n\n|\n[A-Z]|\n[^-•·]|\Z)", resume_text, re.IGNORECASE | re.DOTALL)
        if m:
            skill_sources.append(m.group(1))

        # 1c. "产品工具：xxx" / "技术理解：xxx" 等子类别
        for label in ["产品工具", "技术理解", "技术栈", "开发工具"]:
            pat = rf"{label}[：:]\s*(.+?)(?:\n|$)"
            m = re.search(pat, resume_text, re.IGNORECASE)
            if m:
                skill_sources.append(m.group(1))

        # 1d. 从区块内的列表项提取（- xxx / • xxx / · xxx）
        for m in re.finditer(r"^\s*[-•·]\s*(.+)$", resume_text, re.MULTILINE):
            item = m.group(1).strip()
            # 去掉 "xxx：" 前缀
            item = re.sub(r"^[^：:]*[：:]\s*", "", item)
            skill_sources.append(item)

        # 统一解析所有来源中的技能词
        all_skills = set()
        for src in skill_sources:
            # 按逗号、顿号、分号、空格分隔
            parts = re.split(r"[,，;；/\\\s]+", src)
            for p in parts:
                p = p.strip().strip("-•·")
                # 去掉前缀动词
                p = re.sub(r"^(?:熟练|掌握|了解|熟悉|使用|具备|涉及|能够|通过|运用)\s*", "", p)
                # 去掉常见后缀动词/虚词
                p = re.sub(r"(?:进行|通过|能够|可以|了解|具备|涉及|使用|掌握|熟练|熟悉|运用|及|等|的|了|与|和|或)\s*$", "", p)
                p = p.strip('，、,;；.。:：')
                # 过滤：长度 2-20，不包含过多中文字符的句子
                if 2 <= len(p) <= 20 and len(p) < 15 or (len(p) >= 15 and not re.search(r"[\u4e00-\u9fa5]{4,}", p)):
                    all_skills.add(p)

        # 1e. 兜底：从全文匹配配置中的技术关键词（确保覆盖）
        for kw in settings.RESUME_TECH_KEYWORDS:
            escaped = re.escape(kw)
            if re.search(rf"(^|[^a-zA-Z0-9\u4e00-\u9fa5]){escaped}([^a-zA-Z0-9\u4e00-\u9fa5]|$)", resume_text, re.IGNORECASE):
                all_skills.add(kw)

        result["hard_skills"] = sorted(all_skills)

        # 1f. 质量检查：如果提取结果噪音太大，标记为低质量
        avg_len = sum(len(s) for s in result["hard_skills"]) / len(result["hard_skills"]) if result["hard_skills"] else 0
        is_low_quality = len(result["hard_skills"]) > 15 or avg_len > 12

        # ═══════════════════════════════════════════════════════
        # 2. 软技能
        # ═══════════════════════════════════════════════════════
        m = re.search(r"(?:软技能|综合素质)[：:]\s*(.+?)(?:\n|$)", resume_text, re.IGNORECASE)
        if m and m.group(1):
            result["soft_skills"] = [s.strip() for s in re.split(r"[,，、]", m.group(1)) if s.strip()]

        # ═══════════════════════════════════════════════════════
        # 3. 工作年限：多种模式匹配
        # ═══════════════════════════════════════════════════════
        # 3a. 直接声明工作年限
        patterns = [
            r"(\d+\.?\d*)\s*年\s*(?:以上|经验|工作)",
            r"工作年限[：:]\s*(\d+\.?\d*)\s*年",
            r"工作经验[：:]\s*(\d+\.?\d*)\s*年",
        ]
        for pat in patterns:
            m = re.search(pat, resume_text, re.IGNORECASE)
            if m:
                result["years_of_experience"] = float(m.group(1))
                break

        # 3b. 从教育背景推断：如果最高学历在读且预计毕业年份在未来 → 在校生(years=0)
        # 匹配 "20XX.XX-20XX.XX（预计）" 或 "20XX-20XX"
        edu_year_match = re.search(r"(\d{4})[\.\-]\d{2}[\s\-~～]*(\d{4})[\.\-]?\d{0,2}\s*[（(]?预计[）)]?", resume_text)
        if edu_year_match:
            grad_year = int(edu_year_match.group(2))
            from datetime import datetime
            if grad_year > datetime.now().year:
                result["years_of_experience"] = 0  # 在校生
                result["is_student"] = True

        # ═══════════════════════════════════════════════════════
        # 4. 学历：匹配最高学历
        # ═══════════════════════════════════════════════════════
        edu_patterns = [("博士", 3), ("硕士", 2), ("本科", 1), ("大专", 0)]
        for edu, level in edu_patterns:
            if edu in resume_text:
                result["education"] = edu
                result["education_level"] = level
                break

        # ═══════════════════════════════════════════════════════
        # 5. 领域推断（简单规则）
        # ═══════════════════════════════════════════════════════
        domain_keywords = {
            "AI": ["人工智能", "AI", "大模型", "机器学习", "深度学习", "NLP", "计算机视觉"],
            "后端": ["Java", "后端", "Spring", "微服务", "高并发", "分布式"],
            "前端": ["前端", "React", "Vue", "JavaScript", "TypeScript", "CSS"],
            "产品": ["产品经理", "产品助理", "产品实习", "PRD", "原型", "需求分析"],
            "设计": ["UI", "UX", "设计", "Figma", "视觉", "交互"],
        }
        for domain, kws in domain_keywords.items():
            if any(kw in resume_text for kw in kws):
                result["domain"] = domain
                break

        # ═══════════════════════════════════════════════════════
        # 6. Fallback：如果正则提取效果差，尝试从 resumes.json 读取结构化数据
        # ═══════════════════════════════════════════════════════
        if len(result["hard_skills"]) < 3 or is_low_quality:
            try:
                import json
                from pathlib import Path
                resume_path = Path(__file__).resolve().parent.parent.parent / "data" / "resumes.json"
                with open(resume_path, encoding="utf-8") as f:
                    resumes = json.load(f)
                for r in resumes:
                    raw = r.get("parsed_schema", {}).get("meta", {}).get("raw_text", "")
                    # 使用内容相似度匹配（避免空白差异）
                    if raw and len(raw) > 100 and resume_text and len(resume_text) > 100:
                        raw_norm = raw.strip().replace(" ", "").replace("\r\n", "").replace("\n", "")
                        text_norm = resume_text.strip().replace(" ", "").replace("\r\n", "").replace("\n", "")
                        if raw_norm == text_norm or (len(raw_norm) > 200 and raw_norm[:200] == text_norm[:200]):
                            tech = r.get("parsed_schema", {}).get("skills", {}).get("technical", [])
                            if tech and len(tech) >= 3:
                                result["hard_skills"] = tech
                            # 补充年限（如果正则未提取到）
                            if result["years_of_experience"] == 0:
                                years = r.get("parsed_schema", {}).get("basic_info", {}).get("years_exp")
                                if years is not None:
                                    result["years_of_experience"] = float(years)
                            break
            except Exception:
                pass

        return result

    def _coarse_filter(self, resume: dict, jds: list, top_k: int = None, min_score_threshold: float = None) -> list:
        top_k = top_k or settings.COARSE_FILTER_TOP_K
        min_score_threshold = min_score_threshold if min_score_threshold is not None else settings.COARSE_FILTER_MIN_SCORE
        """
        粗筛层：基于结构化元数据的快速规则过滤。
        不调用 LLM，纯规则计算，O(n) 复杂度。

        评分逻辑：
        - 技能交集（权重最高）：+3分/个
        - 年限差距：-10分（如果低于 min_years）
        - 学历差距：-5分（如果低于 min_education）
        - 领域匹配：+2分
        - 类别匹配：+1分
        - hybrid_score 基础分：+avg_hybrid_score * 5
        """
        edu_map = {"博士": 3, "硕士": 2, "本科": 1, "大专": 0}
        resume_edu_level = edu_map.get(resume.get("education", ""), 0)
        resume_skills = set(s.lower() for s in resume.get("hard_skills", []))
        resume_years = resume.get("years_of_experience", 0)
        resume_domain = resume.get("domain", "")

        scored = []
        for jd in jds:
            score = 0.0
            jd_meta = jd.get("structured_summary", {}) or {}
            jd_keywords = set(k.strip().lower() for k in jd.get("keywords", []))

            # 1. 技能交集（权重最高）
            if resume_skills and jd_keywords:
                overlap = resume_skills & jd_keywords
                score += len(overlap) * 3.0

            # 2. hybrid_score 基础分（检索相关度）
            score += jd.get("avg_hybrid_score", 0) * 5.0

            # 3. 年限检查（硬性门槛，大幅降权但不直接过滤）
            min_years = jd_meta.get("min_years")
            if min_years is not None and resume_years > 0 and resume_years < min_years:
                gap = min_years - resume_years
                score -= 8.0 + gap * 2.0  # 差距越大降权越多

            # 4. 学历检查
            min_edu = jd_meta.get("min_education")
            if min_edu and resume_edu_level > 0:
                min_edu_level = edu_map.get(min_edu, 0)
                if resume_edu_level < min_edu_level:
                    score -= 5.0
                elif resume_edu_level > min_edu_level:
                    score += 1.0  # 学历高于要求，轻微加分

            # 5. 领域匹配
            jd_domain = jd_meta.get("domain", "")
            if resume_domain and jd_domain and resume_domain == jd_domain:
                score += 2.0

            # 6. 类别匹配
            # （简历侧暂无 category，暂不评分）

            scored.append((jd, score))

        # 排序，取 top_k，但保留得分高于阈值的
        scored.sort(key=lambda x: x[1], reverse=True)
        filtered = [j for j, s in scored[:top_k] if s >= min_score_threshold]

        # 兜底：如果过滤后太少，放宽阈值
        if len(filtered) < 3 and len(scored) >= 3:
            filtered = [j for j, s in scored[:max(top_k, 8)] if s >= min_score_threshold - 5]

        best_score = round(scored[0][1], 1) if scored else 0
        worst_score = round(scored[-1][1], 1) if scored else 0
        logger.info(
            f"[GlobalRankTool] 粗筛完成 | input={len(jds)} -> output={len(filtered)} | "
            f"best_score={best_score} | worst_score={worst_score}"
        )
        return filtered

    def _build_jd_summary(self, jd: dict, max_chars: int = 200) -> str:
        """为单个 JD 构建结构化摘要文本"""
        lines = [f"【{jd['company']} · {jd['position']}】"]
        sections = jd.get("sections", {})

        # 按优先级输出 section
        section_order = [
            ("hard_requirements", "硬性要求"),
            ("responsibilities", "岗位职责"),
            ("soft_requirements", "软性要求"),
            ("keywords", "关键词"),
        ]
        for key, label in section_order:
            if key in sections and sections[key]:
                contents = " ".join(sections[key])
                # 硬性要求优先保留完整内容
                if key == "hard_requirements":
                    lines.append(f"[{label}] {contents}")
                else:
                    lines.append(f"[{label}] {contents[:200]}")

        text = "\n".join(lines)
        return text[:max_chars]

    def _template_rank(self, filtered_jds: list, resume_summary: dict, top_k: int = 5) -> dict:
        """
        模板化排序：当 JD 数量较少时跳过 LLM，基于规则生成推荐理由。
        输出格式与 LLM 精排完全一致，确保下游处理无差异。
        """
        resume_skills = set(s.lower() for s in resume_summary.get("hard_skills", []))
        resume_years = resume_summary.get("years_of_experience", 0)
        resume_edu_level = resume_summary.get("education_level", 0)
        resume_domain = resume_summary.get("domain", "")
        edu_map = {"博士": 3, "硕士": 2, "本科": 1, "大专": 0}

        rankings = []
        for i, jd in enumerate(filtered_jds[:top_k]):
            jd_meta = jd.get("structured_summary", {}) or {}
            jd_keywords = set(k.strip().lower() for k in jd.get("keywords", []))

            # 计算技能交集
            skill_overlap = resume_skills & jd_keywords if resume_skills and jd_keywords else set()

            # 计算年限差距（resume_years=0 视为在校生，任何 min_years>0 都算差距）
            min_years = jd_meta.get("min_years")
            year_gap = 0
            if min_years is not None:
                if resume_years > 0 and resume_years < min_years:
                    year_gap = min_years - resume_years
                elif resume_years == 0 and min_years > 0:
                    year_gap = min_years  # 在校生 vs 有年限要求的社招

            # 计算学历差距
            min_edu = jd_meta.get("min_education")
            edu_gap = False
            if min_edu and resume_edu_level > 0:
                min_edu_level = edu_map.get(min_edu, 0)
                edu_gap = resume_edu_level < min_edu_level

            # 判断方向/领域是否一致
            jd_domain = jd_meta.get("domain", "")
            domain_match = bool(resume_domain and jd_domain and resume_domain == jd_domain)
            domain_mismatch = bool(resume_domain and jd_domain and resume_domain != jd_domain)

            # 匹配分数：基于 hybrid_score + 规则调整
            base_score = min(60 + jd.get("avg_hybrid_score", 0) * 30, 95)
            if skill_overlap:
                base_score = min(base_score + len(skill_overlap) * 3, 95)
            if year_gap > 0:
                base_score = max(base_score - 15, 20)
            if edu_gap:
                base_score = max(base_score - 10, 20)
            if domain_mismatch:
                base_score = max(base_score - 10, 15)

            # 生成推荐理由
            reasons = []
            if skill_overlap:
                reasons.append(f"技能匹配：{', '.join(list(skill_overlap)[:3])}")
            if domain_match:
                reasons.append("领域方向一致")
            if domain_mismatch:
                reasons.append(f"方向差异：简历侧重{resume_domain}，JD方向{jd_domain}")
            if year_gap > 0:
                if resume_years == 0:
                    reasons.append(f"经验门槛：需{min_years}年+经验（当前在校生）")
                else:
                    reasons.append(f"经验要求：需{min_years}年+经验（当前{resume_years}年）")
            if edu_gap:
                reasons.append(f"学历要求：需{min_edu}及以上学历")
            if not reasons:
                reasons.append(f"检索相关度较高（{jd.get('avg_hybrid_score', 0):.2f}）")

            # 确定优先级
            if base_score >= 80:
                priority = "高"
            elif base_score >= 60:
                priority = "中"
            elif base_score >= 40:
                priority = "低"
            else:
                priority = "极低"

            # key_match / key_gap
            key_match = []
            key_gap = []
            if skill_overlap:
                key_match.append(f"技能交集：{', '.join(list(skill_overlap)[:3])}")
            if domain_match:
                key_match.append("领域匹配")
            if not key_match:
                key_match.append("检索召回")
            if domain_mismatch:
                key_gap.append(f"方向不匹配（简历{resume_domain} vs JD{jd_domain}）")
            if year_gap > 0:
                if resume_years == 0:
                    key_gap.append(f"在校生身份不符社招要求（需{min_years}年经验）")
                else:
                    key_gap.append(f"工作年限不足（差{year_gap:.0f}年）")
            if edu_gap:
                key_gap.append(f"学历未达要求（需{min_edu}）")
            if not key_gap:
                key_gap.append("需进一步了解详细匹配情况")

            rankings.append({
                "rank": i + 1,
                "jd_id": jd.get("jd_id", ""),
                "company": jd.get("company", "未知"),
                "position": jd.get("position", "未知"),
                "match_score": round(base_score),
                "recommend_reason": "；".join(reasons),
                "key_match": key_match,
                "key_gap": key_gap,
                "apply_priority": priority,
            })

        # 生成策略建议
        high_count = sum(1 for r in rankings if r["apply_priority"] == "高")
        med_count = sum(1 for r in rankings if r["apply_priority"] == "中")
        if high_count >= 2:
            strategy = f"共识别 {high_count} 个高度匹配岗位，建议优先投递；另有 {med_count} 个中等匹配岗位可作为备选。"
        elif high_count == 1:
            strategy = "有 1 个高度匹配岗位建议优先投递，其余岗位可根据个人偏好选择性尝试。"
        elif med_count > 0:
            strategy = f"暂未发现高度匹配岗位，{med_count} 个中等匹配岗位可作为参考，建议进一步评估后再投递。"
        else:
            strategy = "当前推荐岗位与简历匹配度普遍较低，建议优化简历或调整求职方向。"

        return {
            "rankings": rankings,
            "strategy_advice": strategy,
            "_template_mode": True,
        }

    async def execute(self, params: dict) -> NewToolResult:
        candidate_chunks = params.get("candidate_jds", [])
        resume_text = params.get("resume_text", "")
        top_k = params.get("top_k", 5)

        # 防御：candidate_jds 可能是 JSON 字符串（LLM 输出时序列化）
        if isinstance(candidate_chunks, str):
            try:
                candidate_chunks = json.loads(candidate_chunks)
            except Exception:
                candidate_chunks = []
        if not isinstance(candidate_chunks, list):
            candidate_chunks = []

        if not candidate_chunks:
            return NewToolResult(success=False, error="candidate_jds 为空")

        # ═══════════════════════════════════════════════════════
        # 双层召回：粗筛层 → 精排层（动态决策）
        # ═══════════════════════════════════════════════════════

        # ── 1. 将 chunk 列表聚合成 JD 列表 ──
        aggregated_jds = self._aggregate_chunks_by_jd(candidate_chunks)
        if not aggregated_jds:
            return NewToolResult(success=False, error="chunk 聚合后无有效 JD")

        logger.info(
            f"[GlobalRankTool] 聚合完成 | chunks={len(candidate_chunks)} -> jds={len(aggregated_jds)}"
        )

        # ── 2. 粗筛层：基于结构化元数据快速过滤 ──
        resume_summary = self._extract_resume_summary(resume_text)
        coarse_top_k = max(top_k * settings.COARSE_FILTER_MULTIPLIER, settings.COARSE_FILTER_MIN_POOL)
        filtered_jds = self._coarse_filter(resume_summary, aggregated_jds, top_k=coarse_top_k)

        if not filtered_jds:
            logger.warning("[GlobalRankTool] 粗筛后无匹配JD， Fallback 到按 hybrid_score 排序")
            filtered_jds = aggregated_jds[:coarse_top_k]

        logger.info(
            f"[GlobalRankTool] 粗筛完成 | jds={len(aggregated_jds)} -> filtered={len(filtered_jds)} | "
            f"resume_years={resume_summary.get('years_of_experience')} | "
            f"resume_skills={len(resume_summary.get('hard_skills', []))}"
        )

        # ── 3. 动态决策：JD 数量少时跳过 LLM，使用模板输出 ──
        if len(aggregated_jds) <= settings.GLOBAL_RANK_LLM_THRESHOLD:
            logger.info(
                f"[GlobalRankTool] JD数={len(aggregated_jds)} ≤ 阈值={settings.GLOBAL_RANK_LLM_THRESHOLD}，"
                f"跳过LLM精排，使用模板输出"
            )
            data = self._template_rank(filtered_jds, resume_summary, top_k=top_k)
            data["_coarse_filter_meta"] = {
                "input_jds": len(aggregated_jds),
                "output_jds": len(filtered_jds),
                "filter_ratio": round(1 - len(filtered_jds) / len(aggregated_jds), 2) if aggregated_jds else 0,
            }
            return NewToolResult(success=True, data=data)

        # ── 4. JD 数量较多时，调用 LLM 精排 ──
        jd_summaries = []
        for jd in filtered_jds:
            summary = self._build_jd_summary(jd, max_chars=200)
            jd_summaries.append(summary)

        system_prompt = """你是一位资深HR顾问。请将用户简历与以下多个JD进行对比分析，按匹配度排序并给出投递建议。
输出严格JSON格式：
{
  "rankings": [
    {
      "rank": 1,
      "company": "公司名",
      "position": "岗位名",
      "match_score": 85,
      "recommend_reason": "推荐理由",
      "key_match": ["匹配点1", "匹配点2"],
      "key_gap": ["差距1"],
      "apply_priority": "高"
    }
  ],
  "strategy_advice": "整体投递策略建议"
}"""

        user_prompt = (
            f"【用户简历】\n{resume_text[:1200]}...\n\n"
            f"【候选JD】（共{len(jd_summaries)}个，已粗筛）\n\n"
            + "\n\n---\n\n".join(jd_summaries)
            + "\n\n请输出JSON："
        )

        try:
            llm = LLMClient.from_config("core")
            raw = await llm.generate(
                prompt=user_prompt,
                system=system_prompt,
                temperature=0.3,
                max_tokens=2500,
                timeout=TIMEOUT_HEAVY,
            )
            data = json.loads(raw.strip())
            # 补充 jd_id（LLM 返回的 ranking 可能没有 jd_id）
            for i, ranking in enumerate(data.get("rankings", [])):
                if i < len(filtered_jds):
                    ranking["jd_id"] = filtered_jds[i].get("jd_id", "")
            # 注入粗筛元数据（供埋点使用）
            data["_coarse_filter_meta"] = {
                "input_jds": len(aggregated_jds),
                "output_jds": len(filtered_jds),
                "filter_ratio": round(1 - len(filtered_jds) / len(aggregated_jds), 2) if aggregated_jds else 0,
            }
            return NewToolResult(success=True, data=data)
        except Exception as e:
            logger.warning(f"[GlobalRankTool] LLM 排序失败: {e}")

            # Fallback：模板输出
            data = self._template_rank(filtered_jds, resume_summary, top_k=top_k)
            data["strategy_advice"] = f"LLM排序失败（{str(e)[:60]}），返回基于规则的默认排序。"
            data["_coarse_filter_meta"] = {
                "input_jds": len(aggregated_jds),
                "output_jds": len(filtered_jds),
                "filter_ratio": round(1 - len(filtered_jds) / len(aggregated_jds), 2) if aggregated_jds else 0,
            }
            return NewToolResult(success=True, data=data)


class QASynthesizeTool(BaseTool):
    """问答综合工具（简化版：调用 LLM）"""

    @property
    def name(self) -> str:
        return "qa_synthesize"

    @property
    def input_schema(self) -> dict:
        from app.core.tool_registry import QA_SYNTHESIZE_INPUT_SCHEMA
        return QA_SYNTHESIZE_INPUT_SCHEMA

    @property
    def output_schema(self) -> dict:
        from app.core.tool_registry import QA_SYNTHESIZE_OUTPUT_SCHEMA
        return QA_SYNTHESIZE_OUTPUT_SCHEMA

    @property
    def cost_level(self) -> str:
        return "medium"

    @property
    def avg_latency_ms(self) -> int:
        return 1200

    async def execute(self, params: dict) -> NewToolResult:
        question = params.get("question", "")
        evidence_chunks = params.get("evidence_chunks", [])
        qa_type = params.get("qa_type", "factual")

        if not evidence_chunks:
            return NewToolResult(
                success=True,
                data={
                    "answer": "抱歉，知识库中没有找到相关信息。",
                    "citations": [],
                    "confidence": "low",
                    "insufficient_note": "未检索到相关证据",
                }
            )

        evidence_text = "\n\n".join(
            f"[{i+1}] {c.get('content', '')}" for i, c in enumerate(evidence_chunks[:10])
        )

        system_prompt = f"""你是一位专业的求职顾问。基于以下检索到的证据，回答用户的问题。
问题类型：{qa_type}
- factual：事实性问题
- comparative：对比性问题
- temporal：时间相关问题
- definition：定义/概念问题

【约束】
- 必须引用证据来源（用[1][2]等标注）
- 证据不足时明确说明，不要编造
- 回答简洁、结构化

输出严格JSON：
{{
  "answer": "自然语言回答",
  "citations": [
    {{"index": 1, "source": "公司-岗位-板块", "quote": "引用的原文片段"}}
  ],
  "confidence": "high/medium/low",
  "insufficient_note": null
}}"""

        user_prompt = f"【用户问题】\n{question}\n\n【检索证据】\n{evidence_text}\n\n请输出JSON："

        try:
            llm = LLMClient.from_config("chat")
            raw = await llm.generate(prompt=user_prompt, system=system_prompt, temperature=0.3, max_tokens=1500, timeout=TIMEOUT_STANDARD)
            data = json.loads(raw.strip())
            return NewToolResult(success=True, data=data)
        except Exception as e:
            logger.warning(f"[QASynthesizeTool] LLM 回答生成失败: {e}")
            return NewToolResult(
                success=True,
                data={
                    "answer": f"基于检索到的 {len(evidence_chunks)} 条信息，我暂时无法给出准确回答。",
                    "citations": [],
                    "confidence": "low",
                    "insufficient_note": str(e),
                }
            )


class FileOpsTool(BaseTool):
    """资料管理工具"""

    @property
    def name(self) -> str:
        return "file_ops"

    @property
    def input_schema(self) -> dict:
        from app.core.tool_registry import FILE_OPS_INPUT_SCHEMA
        return FILE_OPS_INPUT_SCHEMA

    @property
    def output_schema(self) -> dict:
        from app.core.tool_registry import FILE_OPS_OUTPUT_SCHEMA
        return FILE_OPS_OUTPUT_SCHEMA

    @property
    def cost_level(self) -> str:
        return "low"

    @property
    def avg_latency_ms(self) -> int:
        return 500

    async def execute(self, params: dict) -> NewToolResult:
        operation = params.get("operation", "")

        if operation == "upload_resume":
            return NewToolResult(
                success=False,
                error="简历上传请前往「我的简历」页面完成，支持 PDF / DOCX / TXT 格式。",
                data={"redirect_tip": "请切换到底部「我的简历」标签上传"},
            )

        if operation in ("upload_jd", "upload_jd_image"):
            file_data = params.get("file_data")
            text_data = params.get("text_data")
            return NewToolResult(
                success=True,
                data={
                    "operation": operation,
                    "file_id": f"file_{hash(str(file_data or text_data)) % 10000}",
                    "extracted_data": {
                        "raw_text": text_data or "（文件已接收，解析中...）",
                        "structured": {},
                    },
                    "message": f"{operation} 操作成功",
                }
            )
        elif operation in ("delete_jd", "update_resume"):
            target_id = params.get("target_id")
            return NewToolResult(
                success=True,
                data={
                    "operation": operation,
                    "file_id": target_id,
                    "message": f"{operation} 操作成功",
                }
            )
        elif operation in ("list_jds", "list_resumes"):
            return NewToolResult(
                success=True,
                data={
                    "operation": operation,
                    "file_id": None,
                    "message": f"{operation} 结果：暂无数据",
                }
            )
        else:
            return NewToolResult(success=False, error=f"未知操作: {operation}")


class GeneralChatTool(BaseTool):
    """通用对话工具（直接调用 chat LLM）"""

    @property
    def name(self) -> str:
        return "general_chat"

    @property
    def input_schema(self) -> dict:
        from app.core.tool_registry import GENERAL_CHAT_INPUT_SCHEMA
        return GENERAL_CHAT_INPUT_SCHEMA

    @property
    def output_schema(self) -> dict:
        from app.core.tool_registry import GENERAL_CHAT_OUTPUT_SCHEMA
        return GENERAL_CHAT_OUTPUT_SCHEMA

    @property
    def cost_level(self) -> str:
        return "low"

    @property
    def avg_latency_ms(self) -> int:
        return 300

    async def execute(self, params: dict) -> NewToolResult:
        user_message = params.get("user_message", "")
        chat_type = params.get("chat_type", "other")
        conversation_history = params.get("conversation_history", "")

        system_prompt = """你是一位专业的求职顾问助手。请友好、简洁地回答用户的问题。
如果是职业规划或行业咨询，给出结构化、可执行的建议。"""

        user_prompt = user_message
        if conversation_history:
            user_prompt = f"【历史对话】\n{conversation_history}\n\n【当前问题】\n{user_message}"

        try:
            llm = LLMClient.from_config("chat")
            response = await llm.generate(prompt=user_prompt, system=system_prompt, temperature=0.7, max_tokens=1000, timeout=TIMEOUT_STANDARD)
            return NewToolResult(
                success=True,
                data={
                    "response": response.strip(),
                    "suggested_topics": [
                        "分析一下我的简历匹配度",
                        "推荐几家适合我的公司",
                        "生成一些面试题",
                    ],
                }
            )
        except Exception as e:
            return NewToolResult(success=False, error=str(e))


# ═══════════════════════════════════════════════════════
# 7. 工厂函数：创建 ToolRegistry 实例
# ═══════════════════════════════════════════════════════

def create_tool_registry():
    """创建并注册所有工具的 ToolRegistry 实例"""
    from app.core.tool_registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(KBRetrieveTool())
    registry.register(MatchAnalyzeTool())
    registry.register(GlobalRankTool())
    registry.register(QASynthesizeTool())
    registry.register(InterviewGenTool())
    registry.register(EvidenceRelevanceCheckTool())
    registry.register(FileOpsTool())
    registry.register(GeneralChatTool())
    return registry
