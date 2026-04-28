"""
知识库召回方式对比测试脚本

四种召回方式：
1. 纯向量召回（top-k=15）
2. 纯关键词召回（top-k=15）
3. 混合召回（70%向量 + 30%关键词，top-15）
4. 元数据预过滤+向量（section=hard_requirements，top-15）

使用方式：
    cd backend
    python scripts/test_retrieval.py
    # 然后输入测试问题
"""

import asyncio
import sys
import os
from pathlib import Path

# 确保 backend 目录在路径中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb
from dotenv import load_dotenv

# 加载 .env 配置
env_file = Path(__file__).resolve().parent.parent / ".env"
if env_file.exists():
    load_dotenv(env_file)

from app.core.embedding import EmbeddingClient
from app.core.vector_store import VectorStore


# ═══════════════════════════════════════════════════════
# 1. 纯向量召回
# ═══════════════════════════════════════════════════════

async def retrieve_vector_only(query: str, top_k: int = 15) -> list:
    """纯向量召回：Embedding → ChromaDB query"""
    vs = VectorStore()
    vs.embedding_client = EmbeddingClient.from_config()
    results = await vs.query(query, top_k=top_k)
    return results


# ═══════════════════════════════════════════════════════
# 2. 纯关键词召回
# ═══════════════════════════════════════════════════════

def retrieve_keyword_only(query: str, top_k: int = 15) -> list:
    """
    纯关键词召回：基于文本包含匹配，计算关键词命中次数。
    中文采用字符级匹配 + 简单分词（无 jieba 时退化为字符匹配）。
    """
    persist_dir = str(Path(__file__).resolve().parent.parent / "data" / "chroma_db")
    client = chromadb.PersistentClient(path=persist_dir)
    coll = client.get_collection("jd_knowledge")

    # 获取全部文档
    all_data = coll.get(include=["documents", "metadatas"])
    docs = all_data.get("documents", [])
    metas = all_data.get("metadatas", [])
    ids = all_data.get("ids", [])

    # 提取查询关键词（中文按字符拆分 + 保留连续英文/数字词）
    keywords = _extract_keywords(query)

    scored = []
    for i, doc in enumerate(docs):
        if not doc:
            continue
        score = sum(1 for kw in keywords if kw in doc)
        if score > 0:
            scored.append({
                "chunk_id": ids[i],
                "content": doc,
                "metadata": metas[i],
                "keyword_score": score,
            })

    # 按关键词分数降序
    scored.sort(key=lambda x: x["keyword_score"], reverse=True)
    return scored[:top_k]


def _extract_keywords(text: str) -> list:
    """提取查询关键词：中文按字符拆分，英文/数字保留连续词。"""
    import re
    keywords = set()
    # 中文：每个字符都是一个潜在关键词（去除常见虚词）
    stop_chars = set("的了呢吗吧啊哦嗯之与及或但而因为所以如果虽然但是")  # 标点+虚词
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff" and ch not in stop_chars:
            keywords.add(ch)
    # 英文/数字：连续单词
    words = re.findall(r"[a-zA-Z0-9_+-]+", text)
    for w in words:
        if len(w) >= 2:
            keywords.add(w.lower())
    return list(keywords)


# ═══════════════════════════════════════════════════════
# 3. 混合召回（70%向量 + 30%关键词）
# ═══════════════════════════════════════════════════════

async def retrieve_hybrid(query: str, top_k: int = 15,
                          vec_weight: float = 0.7,
                          kw_weight: float = 0.3) -> list:
    """
    混合召回：
    1. 先向量召回 top-50（作为候选池）
    2. 计算候选池的关键词分数
    3. 归一化后加权混合，取 top-k
    """
    vs = VectorStore()
    vs.embedding_client = EmbeddingClient.from_config()

    # 1. 向量召回候选池（多召回一些用于混合）
    vec_results = await vs.query(query, top_k=50)
    if not vec_results:
        return []

    # 2. 关键词分数（只给候选池中的文档打分，避免全量扫描）
    keywords = _extract_keywords(query)
    cand_kw_scores = {}
    for r in vec_results:
        doc = r.get("content", "")
        score = sum(1 for kw in keywords if kw in doc)
        cand_kw_scores[r["chunk_id"]] = score

    # 3. 归一化并混合
    distances = [r["distance"] for r in vec_results]
    max_dist = max(distances)
    min_dist = min(distances)
    dist_range = max_dist - min_dist if max_dist != min_dist else 1e-9

    max_kw = max(cand_kw_scores.values()) if cand_kw_scores else 1
    kw_range = max_kw if max_kw > 0 else 1

    hybrid = []
    for r in vec_results:
        # 向量相似度：distance 越小 → score 越高（归一化到 0-1）
        vec_sim = 1 - (r["distance"] - min_dist) / dist_range
        # 关键词分数：归一化到 0-1
        kw_sim = cand_kw_scores.get(r["chunk_id"], 0) / kw_range
        # 混合分数
        hybrid_score = vec_weight * vec_sim + kw_weight * kw_sim

        hybrid.append({
            **r,
            "vec_sim": round(vec_sim, 4),
            "kw_sim": round(kw_sim, 4),
            "hybrid_score": round(hybrid_score, 4),
        })

    # 按混合分数降序，取 top-k
    hybrid.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return hybrid[:top_k]


# ═══════════════════════════════════════════════════════
# 4. 元数据预过滤 + 向量召回
# ═══════════════════════════════════════════════════════

async def retrieve_meta_filter_vector(query: str, top_k: int = 15) -> list:
    """
    元数据预过滤 + 向量召回：
    先 where={"section": "hard_requirements"} 过滤，再在子集中向量召回。
    """
    vs = VectorStore()
    vs.embedding_client = EmbeddingClient.from_config()
    results = await vs.query(
        query,
        filters={"section": "hard_requirements"},
        top_k=top_k,
    )
    return results


# ═══════════════════════════════════════════════════════
# 输出格式化
# ═══════════════════════════════════════════════════════

def fmt_result(idx: int, r: dict, score_key: str, score_val) -> str:
    """格式化单条结果"""
    meta = r.get("metadata", {})
    company = meta.get("company", "?")
    section = meta.get("section", "?")
    content = r.get("content", "")[:80].replace("\n", " ")
    return f"  {idx:2d}. [{company:12s}] {section:18s} | {score_key}={score_val} | {content}..."


# ═══════════════════════════════════════════════════════
# 主测试入口
# ═══════════════════════════════════════════════════════

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="知识库召回方式对比测试")
    parser.add_argument("query", nargs="?", default=None, help="测试查询问题（可选，不提供则交互式输入）")
    args = parser.parse_args()

    print("=" * 70)
    print("知识库召回方式对比测试")
    print("=" * 70)

    if args.query:
        await run_single_query(args.query)
        return

    print("提示：输入测试问题，对比四种召回方式的结果。输入 'q' 退出。\n")
    while True:
        query = input("请输入查询问题: ").strip()
        if query.lower() in ("q", "quit", "exit"):
            break
        if not query:
            continue
        await run_single_query(query)


async def run_single_query(query: str):

        print(f"\n{'─' * 70}")
        print(f"查询: {query}")
        print(f"{'─' * 70}\n")

        await run_single_query(query)


async def run_single_query(query: str):
    # ── 1. 纯向量召回 ──
    print("【1】纯向量召回 (top-15)")
    vec_results = await retrieve_vector_only(query, top_k=15)
    for i, r in enumerate(vec_results):
        print(fmt_result(i + 1, r, "dist", f"{r['distance']:.4f}"))
    print(f"  → 共召回 {len(vec_results)} 条\n")

    # ── 2. 纯关键词召回 ──
    print("【2】纯关键词召回 (top-15)")
    kw_results = retrieve_keyword_only(query, top_k=15)
    for i, r in enumerate(kw_results):
        print(fmt_result(i + 1, r, "kw_score", r["keyword_score"]))
    print(f"  → 共召回 {len(kw_results)} 条\n")

    # ── 3. 混合召回 ──
    print("【3】混合召回 (70%向量 + 30%关键词, top-15)")
    hybrid_results = await retrieve_hybrid(query, top_k=15)
    for i, r in enumerate(hybrid_results):
        score_str = f"hybrid={r['hybrid_score']:.4f}(vec={r['vec_sim']:.4f},kw={r['kw_sim']:.4f})"
        print(fmt_result(i + 1, r, "score", score_str))
    print(f"  → 共召回 {len(hybrid_results)} 条\n")

    # ── 4. 元数据预过滤 + 向量 ──
    print("【4】元数据预过滤+向量 (section=hard_requirements, top-15)")
    meta_results = await retrieve_meta_filter_vector(query, top_k=15)
    for i, r in enumerate(meta_results):
        print(fmt_result(i + 1, r, "dist", f"{r['distance']:.4f}"))
    print(f"  → 共召回 {len(meta_results)} 条\n")

    print("=" * 70)
    print()


if __name__ == "__main__":
    asyncio.run(main())
