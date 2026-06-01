"""
RAG 检索层独立评测脚本

用法：
    cd backend && python eval/run_rag_eval.py
    cd backend && python eval/run_rag_eval.py --strategy hybrid --top_k 10 --rerank
    cd backend && python eval/run_rag_eval.py --output eval/rag_report.json

支持策略：
    - vector: 仅向量检索
    - bm25: 仅 BM25 检索
    - hybrid: 混合召回（默认 70%vec + 30%bm25）
    - hybrid+rerank: 混合召回 + CrossEncoder 重排序
"""

import argparse
import asyncio
import json
import logging
import math
import sys
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple

# 将 backend 加入路径
sys.path.insert(0, "..")

from app.core.vector_store import VectorStore
from app.core.embedding import EmbeddingClient
from app.core.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_rag_dataset(path: str = "eval/rag_test_dataset.jsonl") -> List[Dict]:
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skip malformed line {i}: {e}")
    return cases


def load_all_chunks(path: str = "data/jds.json") -> Dict[str, Dict]:
    """加载所有 chunk 的元数据，建立 chunk_id -> {jd_id, section} 映射"""
    chunk_meta = {}
    with open(path, "r", encoding="utf-8") as f:
        jds = json.load(f)
    for jd in jds:
        jd_id = jd["id"]
        chunk_ids = jd.get("meta", {}).get("chunk_ids", [])
        sections = jd.get("sections", {})
        hard_count = len(sections.get("hard_requirements", []))
        soft_count = len(sections.get("soft_requirements", []))
        
        for idx, cid in enumerate(chunk_ids):
            if idx == 0:
                section = "basic_info"
            elif idx == 1:
                section = "responsibilities"
            elif 2 <= idx < 2 + hard_count:
                section = "hard_requirements"
            elif 2 + hard_count <= idx < 2 + hard_count + soft_count:
                section = "soft_requirements"
            elif idx == len(chunk_ids) - 1:
                section = "keywords"
            else:
                section = "unknown"
            chunk_meta[cid] = {"jd_id": jd_id, "section": section, "company": jd["company"], "position": jd["position"]}
    return chunk_meta


async def retrieve_vector(query: str, top_k: int = 20) -> List[Dict]:
    """仅向量检索"""
    vs = VectorStore()
    vs.embedding_client = EmbeddingClient.from_config()
    results = await vs.query(query, top_k=top_k)
    return [{"chunk_id": r["chunk_id"], "score": 1.0 - r["distance"]} for r in results]


def retrieve_bm25(query: str, top_k: int = 20) -> List[Dict]:
    from pathlib import Path
    """仅 BM25 检索"""
    import jieba
    import numpy as np
    from rank_bm25 import BM25Okapi
    import chromadb

    persist_dir = str(Path(__file__).resolve().parent.parent / "data" / "chroma_db")
    client = chromadb.PersistentClient(path=persist_dir)
    coll = client.get_collection("jd_knowledge")
    results = coll.get(include=["documents", "metadatas"])
    ids = results["ids"]
    docs = results["documents"]

    tokenized_docs = [list(jieba.cut_for_search(d)) if d else [] for d in docs]
    bm25 = BM25Okapi(tokenized_docs)
    tokens = list(jieba.cut_for_search(query))
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [{"chunk_id": ids[i], "score": float(scores[i])} for i in top_indices if scores[i] > 0]


async def retrieve_hybrid(query: str, top_k: int = 20, vec_weight: float = 0.7, bm25_weight: float = 0.3) -> List[Dict]:
    """混合召回（简化版，复用 tools.py 中的逻辑）"""
    # 优先复用业务层的 _kb_retrieve_stub
    from app.core.tools import _kb_retrieve_stub
    result = await _kb_retrieve_stub(query=query, top_k=top_k)
    if result.success and result.data:
        chunks = result.data.get("chunks", [])
        return [{"chunk_id": c["chunk_id"], "score": c.get("hybrid_score", 0)} for c in chunks]
    return []


async def retrieve_hybrid_rerank(query: str, top_k: int = 10) -> List[Dict]:
    """混合召回 + CrossEncoder 重排序"""
    from app.core.tools import _kb_retrieve_stub
    result = await _kb_retrieve_stub(query=query, top_k=top_k)
    if result.success and result.data:
        chunks = result.data.get("chunks", [])
        return [{"chunk_id": c["chunk_id"], "score": c.get("rerank_score", c.get("hybrid_score", 0))} for c in chunks]
    return []


def compute_metrics(case: Dict, retrieved_chunks: List[Dict], chunk_meta: Dict) -> Dict:
    """计算单条 case 的指标"""
    golden_chunks = set(case.get("golden_chunk_ids", []))
    golden_jds = set(case.get("golden_jd_ids", []))
    relevance_scores = case.get("relevance_scores", {})
    
    if not retrieved_chunks:
        return {
            "context_precision": 0.0,
            "context_recall": 0.0,
            "context_f1": 0.0,
            "hit_rate": 1.0 if not golden_jds else 0.0,
            "mrr": 0.0,
            "ndcg@k": 0.0,
            "retrieved_count": 0,
            "golden_count": len(golden_chunks),
        }

    retrieved_ids = [r["chunk_id"] for r in retrieved_chunks]
    retrieved_set = set(retrieved_ids)
    
    # --- Chunk Level Metrics ---
    intersect = retrieved_set & golden_chunks
    precision = len(intersect) / len(retrieved_set) if retrieved_set else 0.0
    recall = len(intersect) / len(golden_chunks) if golden_chunks else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # --- JD Level Metrics ---
    # 提取 retrieved chunks 对应的 JDs
    retrieved_jds = []
    seen_jds = set()
    for cid in retrieved_ids:
        meta = chunk_meta.get(cid, {})
        jd_id = meta.get("jd_id")
        if jd_id is not None and jd_id not in seen_jds:
            seen_jds.add(jd_id)
            retrieved_jds.append(jd_id)

    # Hit Rate
    hit = 1.0 if any(jd in golden_jds for jd in retrieved_jds) else 0.0

    # MRR
    mrr = 0.0
    for rank, jd_id in enumerate(retrieved_jds, 1):
        if jd_id in golden_jds:
            mrr = 1.0 / rank
            break

    # NDCG@k (k = len(retrieved_jds))
    def dcg(scores):
        return sum((2**s - 1) / math.log2(i + 2) for i, s in enumerate(scores))
    
    k = len(retrieved_jds)
    retrieved_scores = [relevance_scores.get(str(jd), 0) for jd in retrieved_jds]
    # ideal scores: 取所有 relevance_score > 0 的 JD 按分数降序排列后取前 k 个
    all_relevant_scores = sorted(
        [s for s in relevance_scores.values() if s > 0],
        reverse=True
    )
    ideal_scores = all_relevant_scores[:k]
    while len(ideal_scores) < k:
        ideal_scores.append(0)
    
    dcg_val = dcg(retrieved_scores)
    idcg_val = dcg(ideal_scores)
    ndcg = dcg_val / idcg_val if idcg_val > 0 else 0.0

    return {
        "context_precision": round(precision, 4),
        "context_recall": round(recall, 4),
        "context_f1": round(f1, 4),
        "hit_rate": hit,
        "mrr": round(mrr, 4),
        "ndcg": round(ndcg, 4),
        "retrieved_count": len(retrieved_ids),
        "golden_count": len(golden_chunks),
        "retrieved_jds": retrieved_jds,
    }


async def run_eval(strategy: str, top_k: int, output_path: str):
    cases = load_rag_dataset()
    chunk_meta = load_all_chunks()
    
    logger.info(f"Loaded {len(cases)} RAG test cases")
    logger.info(f"Strategy: {strategy}, top_k: {top_k}")

    results = []
    total_metrics = defaultdict(float)

    for case in cases:
        query = case.get("rewritten_query") or case["original_query"]
        
        t0 = time.time()
        if strategy == "vector":
            retrieved = await retrieve_vector(query, top_k=top_k)
        elif strategy == "bm25":
            retrieved = retrieve_bm25(query, top_k=top_k)
        elif strategy == "hybrid":
            retrieved = await retrieve_hybrid(query, top_k=top_k)
        elif strategy == "hybrid_rerank":
            retrieved = await retrieve_hybrid_rerank(query, top_k=top_k)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        latency = time.time() - t0

        metrics = compute_metrics(case, retrieved, chunk_meta)
        metrics["latency_ms"] = round(latency * 1000, 2)
        
        for k in ["context_precision", "context_recall", "context_f1", "hit_rate", "mrr", "ndcg"]:
            total_metrics[k] += metrics[k]

        results.append({
            "case_id": case["case_id"],
            "query": query,
            "strategy": strategy,
            **metrics,
        })

    n = len(cases)
    avg_metrics = {k: round(v / n, 4) for k, v in total_metrics.items()}

    report = {
        "summary": {
            "strategy": strategy,
            "top_k": top_k,
            "total_cases": n,
            "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / n, 2),
            **avg_metrics,
        },
        "cases": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"Report saved to {output_path}")
    logger.info("=" * 50)
    for k, v in avg_metrics.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="RAG Retrieval Evaluation")
    parser.add_argument("--strategy", choices=["vector", "bm25", "hybrid", "hybrid_rerank"], default="hybrid")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--output", default="eval/rag_eval_report.json")
    args = parser.parse_args()

    asyncio.run(run_eval(args.strategy, args.top_k, args.output))


if __name__ == "__main__":
    from pathlib import Path
    main()
