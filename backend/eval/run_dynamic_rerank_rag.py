"""
路由 RAG + 动态 Reranker top_k 评测
"""

import asyncio
import json
import sys
sys.path.insert(0, "..")

import numpy as np
import jieba
import chromadb
from rank_bm25 import BM25Okapi

from app.core.vector_store import VectorStore
from app.core.embedding import EmbeddingClient
from app.core import reranker
from eval.run_rag_eval import load_rag_dataset, load_all_chunks, compute_metrics


# 路由配置：{候选池大小, reranker输出大小}
ROUTES = {
    "single_jd":           {"pool": 12, "out": 10},
    "single_jd_not_found": {"pool": 5,  "out": 5},
    "verify":              {"pool": 12, "out": 10},
    "verify_expand":       {"pool": 10, "out": 8},
    "explore":             {"pool": 25, "out": 18},
    "explore+single_jd":   {"pool": 15, "out": 12},
    "explore+verify":      {"pool": 15, "out": 12},
    "skill_explore":       {"pool": 18, "out": 12},
    "multi_jd_compare":    {"pool": 15, "out": 12},
}


async def retrieve_with_dynamic_k(query, pool_k, out_k):
    vs = VectorStore()
    vs.embedding_client = EmbeddingClient.from_config()
    vec_results = await vs.query(query, top_k=pool_k)

    # BM25
    client = chromadb.PersistentClient(path="data/chroma_db")
    coll = client.get_collection("jd_knowledge")
    bm25_data = coll.get(include=["documents", "metadatas"])
    ids = bm25_data["ids"]
    docs = bm25_data["documents"]
    metas = bm25_data["metadatas"]

    tokenized = [list(jieba.cut_for_search(d)) if d else [] for d in docs]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(list(jieba.cut_for_search(query)))
    top_idx = np.argsort(scores)[::-1][:pool_k]

    # 融合
    pool = {}
    for r in vec_results:
        cid = r["chunk_id"]
        pool[cid] = {
            "chunk_id": cid,
            "content": r["content"],
            "metadata": r["metadata"],
            "hybrid_score": 1.0 - r["distance"] if r["distance"] is not None else 0.0,
        }
    for idx in top_idx:
        if scores[idx] > 0:
            cid = ids[idx]
            if cid in pool:
                pool[cid]["hybrid_score"] += float(scores[idx]) * 0.3
            else:
                pool[cid] = {
                    "chunk_id": cid,
                    "content": docs[idx],
                    "metadata": metas[idx],
                    "hybrid_score": float(scores[idx]) * 0.3,
                }

    items = sorted(pool.values(), key=lambda x: x["hybrid_score"], reverse=True)[:pool_k]

    # 动态 reranker
    if out_k < len(items):
        ranked = await reranker.rerank(query=query, candidates=items, top_k=out_k)
        result = []
        for orig_idx, score in ranked:
            item = items[orig_idx].copy()
            item["rerank_score"] = round(score, 4)
            result.append(item)
        items = result
    else:
        for item in items:
            item["rerank_score"] = item["hybrid_score"]

    return items


async def main():
    cases = load_rag_dataset()
    chunk_meta = load_all_chunks()

    results = []
    total = {"context_precision": 0, "context_recall": 0, "context_f1": 0, "hit_rate": 0, "mrr": 0, "ndcg": 0}
    type_metrics = {}

    for case in cases:
        rtype = case.get("retrieval_type", "unknown")
        cfg = ROUTES.get(rtype, {"pool": 20, "out": 10})
        query = case.get("rewritten_query") or case["original_query"]

        import time
        t0 = time.time()
        chunks = await retrieve_with_dynamic_k(query, cfg["pool"], cfg["out"])
        latency = time.time() - t0

        retrieved = [{"chunk_id": c["chunk_id"], "score": c.get("rerank_score", 0)} for c in chunks]
        metrics = compute_metrics(case, retrieved, chunk_meta)
        metrics["latency_ms"] = round(latency * 1000, 2)
        metrics["top_k"] = cfg["out"]

        for k in total:
            total[k] += metrics[k]
        type_metrics.setdefault(rtype, []).append(metrics)
        results.append({"case_id": case["case_id"], **metrics})

    n = len(cases)
    avg = {k: round(v / n, 4) for k, v in total.items()}

    print("=== Routed RAG with dynamic reranker ===")
    print(f"Precision: {avg['context_precision']}")
    print(f"Recall:    {avg['context_recall']}")
    print(f"F1:        {avg['context_f1']}")
    print(f"Hit Rate:  {avg['hit_rate']}")
    print(f"MRR:       {avg['mrr']}")
    print(f"NDCG:      {avg['ndcg']}")
    print(f"Latency:   {sum(r['latency_ms'] for r in results) / n:.1f}ms")

    print()
    print("=== By Type ===")
    for t, items in sorted(type_metrics.items()):
        m = {k: round(sum(i[k] for i in items) / len(items), 4) for k in total}
        print(
            f"{t:<20} n={len(items)}  "
            f"P={m['context_precision']:.3f} R={m['context_recall']:.3f} F1={m['context_f1']:.3f}  "
            f"Hit={m['hit_rate']:.3f} MRR={m['mrr']:.3f}"
        )

    # 保存报告
    with open("eval/rag_report_routed_dynamic.json", "w", encoding="utf-8") as f:
        json.dump({"summary": avg, "by_type": {t: {k: round(sum(i[k] for i in items) / len(items), 4) for k in total} | {"count": len(items)} for t, items in type_metrics.items()}, "cases": results}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
