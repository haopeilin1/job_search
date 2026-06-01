"""
RAG 策略独立消融实验

测试四种检索策略：
1. vector      — 纯向量检索
2. bm25        — 纯 BM25 关键词检索
3. hybrid_73   — 混合检索（向量 70% + BM25 30%，min-max 归一化加权）
4. hybrid_rrf  — RRF 融合（Reciprocal Rank Fusion，k=60）

每组策略测试两组参数：
- pool=15, rerank_out=10（轻量版）
- pool=25, rerank_out=15（深度版）

按 retrieval_type 分组计算指标，输出对比表格。

完全独立于业务层的 _kb_retrieve_stub，直接操作 ChromaDB + BM25 + CrossEncoder。
"""

import asyncio
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
import jieba
import numpy as np
from rank_bm25 import BM25Okapi


# ── 配置 ──────────────────────────────────────────
CHROMA_PERSIST_DIR = str(Path(__file__).resolve().parent.parent / "data" / "chroma_db")
COLLECTION_NAME = "jd_knowledge"
RERANKER_MODEL = "BAAI/bge-reranker-base"
RRF_K = 60

# 参数矩阵: [(候选池大小, reranker输出大小)]
K_CONFIGS = [
    (15, 10),
    (25, 15),
]

# 按场景分组统计
@dataclass
class Metrics:
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    hit: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    latency_ms: float = 0.0


def load_rag_dataset(path: str = "eval/rag_test_dataset.jsonl") -> List[Dict]:
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return cases


def load_all_chunks(path: str = "data/jds.json") -> Dict[str, Dict]:
    """chunk_id -> {jd_id, section, company, position}"""
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
            chunk_meta[cid] = {
                "jd_id": jd_id,
                "section": section,
                "company": jd["company"],
                "position": jd["position"],
            }
    return chunk_meta


class IndependentRetriever:
    """
    完全独立的检索器：不依赖业务层的 _kb_retrieve_stub，
    直接操作 ChromaDB + BM25 + CrossEncoder。
    """

    def __init__(self):
        self._client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self._coll = self._client.get_collection(COLLECTION_NAME)
        self._bm25_index = None
        self._bm25_ids: List[str] = []
        self._bm25_docs: List[str] = []
        self._bm25_metadatas: List[dict] = []
        self._reranker = None
        self._build_bm25()

    def _build_bm25(self):
        results = self._coll.get(include=["documents", "metadatas"])
        self._bm25_ids = results["ids"]
        self._bm25_docs = results["documents"]
        self._bm25_metadatas = results["metadatas"]
        tokenized = [
            list(jieba.cut_for_search(d)) if d else []
            for d in self._bm25_docs
        ]
        self._bm25_index = BM25Okapi(tokenized)

    def _get_reranker(self):
        if self._reranker is None:
            from app.core import reranker as reranker_mod
            self._reranker = reranker_mod
        return self._reranker

    # ── 1. 纯向量检索 ──────────────────────────
    async def retrieve_vector(self, query: str, top_k: int) -> List[Dict]:
        from app.core.embedding import EmbeddingClient

        embed = EmbeddingClient.from_config()
        embeddings = await embed.embed([query])
        query_emb = embeddings[0]

        results = self._coll.query(
            query_embeddings=[query_emb],
            n_results=top_k,
        )
        output = []
        for i in range(len(results["ids"][0])):
            cid = results["ids"][0][i]
            dist = results["distances"][0][i]
            output.append({
                "chunk_id": cid,
                "score": 1.0 - dist,
                "rank": i + 1,
            })
        return output

    # ── 2. 纯 BM25 检索 ────────────────────────
    def retrieve_bm25(self, query: str, top_k: int) -> List[Dict]:
        tokens = list(jieba.cut_for_search(query))
        scores = self._bm25_index.get_scores(tokens)
        if not isinstance(scores, np.ndarray):
            scores = np.array(scores)
        top_indices = np.argsort(scores)[::-1][:top_k]
        output = []
        rank = 1
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            output.append({
                "chunk_id": self._bm25_ids[idx],
                "score": float(scores[idx]),
                "rank": rank,
            })
            rank += 1
        return output

    # ── 3. 7:3 加权混合 ────────────────────────
    def _merge_weighted(self, vec_results: List[Dict], bm25_results: List[Dict], pool_k: int) -> List[Dict]:
        pool: Dict[str, Dict] = {}

        # 向量分数 min-max 归一化
        vec_scores = [r["score"] for r in vec_results]
        v_min, v_max = min(vec_scores), max(vec_scores) if vec_scores else (0, 1)
        v_range = v_max - v_min if v_max != v_min else 1e-9

        for r in vec_results:
            cid = r["chunk_id"]
            pool[cid] = {
                "chunk_id": cid,
                "vec_norm": (r["score"] - v_min) / v_range,
                "bm25_norm": 0.0,
            }

        # BM25 分数 min-max 归一化
        bm25_scores = [r["score"] for r in bm25_results if r["score"] > 0]
        b_min, b_max = min(bm25_scores), max(bm25_scores) if bm25_scores else (0, 1)
        b_range = b_max - b_min if b_max != b_min else 1e-9

        for r in bm25_results:
            cid = r["chunk_id"]
            if cid in pool:
                pool[cid]["bm25_norm"] = (r["score"] - b_min) / b_range
            else:
                pool[cid] = {
                    "chunk_id": cid,
                    "vec_norm": 0.0,
                    "bm25_norm": (r["score"] - b_min) / b_range,
                }

        for item in pool.values():
            item["hybrid_score"] = 0.7 * item["vec_norm"] + 0.3 * item["bm25_norm"]

        items = sorted(pool.values(), key=lambda x: x["hybrid_score"], reverse=True)[:pool_k]
        return [{"chunk_id": i["chunk_id"], "score": i["hybrid_score"], "rank": idx + 1}
                for idx, i in enumerate(items)]

    # ── 4. RRF 融合 ────────────────────────────
    def _merge_rrf(self, vec_results: List[Dict], bm25_results: List[Dict], pool_k: int) -> List[Dict]:
        scores: Dict[str, float] = {}
        ranks: Dict[str, Dict[str, int]] = {}

        for r in vec_results:
            cid = r["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + r["rank"])
            ranks.setdefault(cid, {})["vec"] = r["rank"]

        for r in bm25_results:
            cid = r["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + r["rank"])
            ranks.setdefault(cid, {})["bm25"] = r["rank"]

        items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:pool_k]
        return [{"chunk_id": cid, "score": score, "rank": idx + 1}
                for idx, (cid, score) in enumerate(items)]

    # ── Reranker ───────────────────────────────
    async def rerank(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        if not candidates or top_k <= 0:
            return candidates

        reranker_mod = self._get_reranker()
        ranked = await reranker_mod.rerank(
            query=query,
            candidates=candidates,
            top_k=top_k,
        )
        result = []
        for orig_idx, score in ranked:
            item = candidates[orig_idx].copy()
            item["rerank_score"] = round(score, 4)
            result.append(item)
        return result

    # ── 统一入口 ───────────────────────────────
    async def retrieve(
        self,
        query: str,
        strategy: str,
        pool_k: int,
        rerank_k: int,
    ) -> List[Dict]:
        t0 = time.time()

        # 两路独立召回
        vec_results = await self.retrieve_vector(query, top_k=pool_k)
        bm25_results = self.retrieve_bm25(query, top_k=pool_k)

        # 融合
        if strategy == "vector":
            candidates = vec_results
        elif strategy == "bm25":
            candidates = bm25_results
        elif strategy == "hybrid_73":
            candidates = self._merge_weighted(vec_results, bm25_results, pool_k)
        elif strategy == "hybrid_rrf":
            candidates = self._merge_rrf(vec_results, bm25_results, pool_k)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # CrossEncoder 重排序
        reranked = await self.rerank(query, candidates, rerank_k)

        latency = (time.time() - t0) * 1000
        return reranked, latency


def compute_metrics(case: Dict, retrieved: List[Dict], chunk_meta: Dict) -> Dict:
    golden_chunks = set(case.get("golden_chunk_ids", []))
    golden_jds = set(case.get("golden_jd_ids", []))
    relevance_scores = case.get("relevance_scores", {})

    if not retrieved:
        return {
            "context_precision": 0.0,
            "context_recall": 0.0,
            "context_f1": 0.0,
            "hit_rate": 1.0 if not golden_jds else 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
            "retrieved_count": 0,
            "golden_count": len(golden_chunks),
        }

    retrieved_ids = [r["chunk_id"] for r in retrieved]
    retrieved_set = set(retrieved_ids)

    # Chunk 级
    intersect = retrieved_set & golden_chunks
    precision = len(intersect) / len(retrieved_set) if retrieved_set else 0.0
    recall = len(intersect) / len(golden_chunks) if golden_chunks else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # JD 级
    retrieved_jds = []
    seen = set()
    for cid in retrieved_ids:
        jd_id = chunk_meta.get(cid, {}).get("jd_id")
        if jd_id is not None and jd_id not in seen:
            seen.add(jd_id)
            retrieved_jds.append(jd_id)

    hit = 1.0 if any(jd in golden_jds for jd in retrieved_jds) else 0.0

    mrr = 0.0
    for rank, jd_id in enumerate(retrieved_jds, 1):
        if jd_id in golden_jds:
            mrr = 1.0 / rank
            break

    # NDCG
    def dcg(scores):
        return sum((2 ** s - 1) / math.log2(i + 2) for i, s in enumerate(scores))

    k = len(retrieved_jds)
    retrieved_scores = [relevance_scores.get(str(jd), 0) for jd in retrieved_jds]
    all_relevant = sorted([s for s in relevance_scores.values() if s > 0], reverse=True)
    ideal_scores = all_relevant[:k]
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
    }


async def main():
    strategies = ["vector", "bm25", "hybrid_73", "hybrid_rrf"]
    cases = load_rag_dataset()
    chunk_meta = load_all_chunks()
    retriever = IndependentRetriever()

    print(f"Loaded {len(cases)} RAG test cases")
    print(f"Loaded {len(chunk_meta)} chunks from ChromaDB")

    all_results = []

    for strategy in strategies:
        for pool_k, rerank_k in K_CONFIGS:
            config_label = f"k{pool_k}-{rerank_k}"
            print(f"\n>>> Running {strategy} {config_label} ...")

            case_results = []
            total = {"context_precision": 0, "context_recall": 0, "context_f1": 0, "hit_rate": 0, "mrr": 0, "ndcg": 0}
            type_metrics = {}

            for case in cases:
                query = case.get("rewritten_query") or case["original_query"]
                rtype = case.get("retrieval_type", "unknown")

                retrieved, latency = await retriever.retrieve(
                    query=query,
                    strategy=strategy,
                    pool_k=pool_k,
                    rerank_k=rerank_k,
                )

                metrics = compute_metrics(case, retrieved, chunk_meta)
                metrics["latency_ms"] = round(latency, 2)
                metrics["strategy"] = strategy
                metrics["config"] = config_label
                metrics["retrieval_type"] = rtype

                for k in ["context_precision", "context_recall", "context_f1", "hit_rate", "mrr", "ndcg"]:
                    total[k] += metrics[k]

                type_metrics.setdefault(rtype, []).append(metrics)
                case_results.append(metrics)

            n = len(cases)
            avg = {k: round(v / n, 4) for k, v in total.items()}
            avg_latency = round(sum(r["latency_ms"] for r in case_results) / n, 2)

            # 按场景汇总
            type_summary = {}
            for rtype, items in type_metrics.items():
                m = {k: round(sum(i[k] for i in items) / len(items), 4) for k in total}
                type_summary[rtype] = {
                    "count": len(items),
                    **m,
                }

            result = {
                "strategy": strategy,
                "config": config_label,
                "pool_k": pool_k,
                "rerank_k": rerank_k,
                "summary": {
                    "total_cases": n,
                    "avg_latency_ms": avg_latency,
                    **avg,
                },
                "by_type": type_summary,
                "cases": case_results,
            }
            all_results.append(result)

            print(f"  Precision: {avg['context_precision']}")
            print(f"  Recall:    {avg['context_recall']}")
            print(f"  F1:        {avg['context_f1']}")
            print(f"  Hit Rate:  {avg['hit_rate']}")
            print(f"  MRR:       {avg['mrr']}")
            print(f"  NDCG:      {avg['ndcg']}")
            print(f"  Latency:   {avg_latency}ms")

    # 保存完整报告
    with open("eval/rag_strategy_ablation_report.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # ── 打印汇总表格 ──────────────────────────────
    print("\n" + "=" * 130)
    header = (
        f"{'策略':<12} {'参数':<8} {'Precision':>10} {'Recall':>10} {'F1':>10} "
        f"{'HitRate':>8} {'MRR':>8} {'NDCG':>8} {'延迟ms':>8}"
    )
    print(header)
    print("=" * 130)
    for r in all_results:
        s = r["summary"]
        line = (
            f"{r['strategy']:<12} {r['config']:<8} "
            f"{s['context_precision']:>10.4f} {s['context_recall']:>10.4f} {s['context_f1']:>10.4f} "
            f"{s['hit_rate']:>8.4f} {s['mrr']:>8.4f} {s['ndcg']:>8.4f} {s['avg_latency_ms']:>8.1f}"
        )
        print(line)
    print("=" * 130)

    # ── 按场景打印详细对比 ────────────────────────
    types = sorted({t for r in all_results for t in r["by_type"]})
    for metric in ["context_precision", "context_recall", "context_f1", "hit_rate", "mrr", "ndcg"]:
        print(f"\n=== {metric.upper()} 按场景对比 ===")
        print(f"{'场景':<20}", end="")
        for r in all_results:
            print(f"{r['strategy'][:6]}_{r['config']:<10}", end="")
        print()
        print("-" * 110)
        for t in types:
            print(f"{t:<20}", end="")
            for r in all_results:
                val = r["by_type"].get(t, {}).get(metric, 0.0)
                print(f"{val:>16.4f}", end="")
            print()

    print("\nReport saved to eval/rag_strategy_ablation_report.json")


if __name__ == "__main__":
    asyncio.run(main())
