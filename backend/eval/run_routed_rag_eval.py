"""
路由 RAG 评测：不同 retrieval_type 使用不同的 top_k
"""

import asyncio
import json
import sys
sys.path.insert(0, "..")

from eval.run_rag_eval import run_eval


# 路由策略配置
ROUTED_CONFIGS = {
    "uniform_k10": {  # 基线：所有场景统一 k=10
        "single_jd": 10,
        "single_jd_not_found": 10,
        "verify": 10,
        "verify_expand": 10,
        "explore": 10,
        "explore+single_jd": 10,
        "explore+verify": 10,
        "skill_explore": 10,
        "multi_jd_compare": 10,
    },
    "routed_v1": {  # 路由策略 v1
        "single_jd": 8,           # 精确查询，少而精
        "single_jd_not_found": 5,  # 空结果检测，快速止损
        "verify": 10,              # 属性查询，够用就行
        "verify_expand": 8,        # 多轮展开，继承缓存
        "explore": 18,             # 宽泛探索，需要广度
        "explore+single_jd": 12,   # 探索+深度，兼顾
        "explore+verify": 12,      # 探索+属性，兼顾
        "skill_explore": 12,       # 技能导向，多候选
        "multi_jd_compare": 12,    # 多 JD 对比
    },
    "routed_v2": {  # 路由策略 v2（更激进）
        "single_jd": 10,
        "single_jd_not_found": 5,
        "verify": 12,
        "verify_expand": 10,
        "explore": 20,
        "explore+single_jd": 15,
        "explore+verify": 15,
        "skill_explore": 15,
        "multi_jd_compare": 15,
    },
}


async def run_routed_eval(config_name: str, config: dict, output_path: str):
    from eval.run_rag_eval import load_rag_dataset, load_all_chunks, compute_metrics
    from app.core.tools import _kb_retrieve_stub
    import time

    cases = load_rag_dataset()
    chunk_meta = load_all_chunks()

    results = []
    total_metrics = {
        "context_precision": 0.0, "context_recall": 0.0, "context_f1": 0.0,
        "hit_rate": 0.0, "mrr": 0.0, "ndcg": 0.0,
    }
    type_metrics = {}

    for case in cases:
        rtype = case.get("retrieval_type", "unknown")
        top_k = config.get(rtype, 10)
        query = case.get("rewritten_query") or case["original_query"]

        t0 = time.time()
        result = await _kb_retrieve_stub(query=query, top_k=top_k)
        latency = time.time() - t0

        if result.success and result.data:
            chunks = result.data.get("chunks", [])
            retrieved = [{"chunk_id": c["chunk_id"], "score": c.get("rerank_score", c.get("hybrid_score", 0))} for c in chunks]
        else:
            retrieved = []

        metrics = compute_metrics(case, retrieved, chunk_meta)
        metrics["latency_ms"] = round(latency * 1000, 2)
        metrics["top_k_used"] = top_k
        metrics["retrieval_type"] = rtype

        for k in total_metrics:
            total_metrics[k] += metrics[k]

        type_metrics.setdefault(rtype, []).append(metrics)

        results.append({
            "case_id": case["case_id"],
            "query": query,
            "top_k_used": top_k,
            **metrics,
        })

    n = len(cases)
    avg_metrics = {k: round(v / n, 4) for k, v in total_metrics.items()}
    avg_latency = round(sum(r["latency_ms"] for r in results) / n, 2)

    # 按场景分组统计
    type_summary = {}
    for rtype, items in type_metrics.items():
        m = {k: round(sum(i[k] for i in items) / len(items), 4) for k in total_metrics}
        type_summary[rtype] = {
            "count": len(items),
            **m,
        }

    report = {
        "config_name": config_name,
        "config": config,
        "summary": {
            "total_cases": n,
            "avg_latency_ms": avg_latency,
            **avg_metrics,
        },
        "by_type": type_summary,
        "cases": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


async def main():
    for config_name, config in ROUTED_CONFIGS.items():
        print(f"\n>>> Running {config_name} ...")
        output = f"eval/rag_report_{config_name}.json"
        report = await run_routed_eval(config_name, config, output)

        print(f"  Precision: {report['summary']['context_precision']}")
        print(f"  Recall:    {report['summary']['context_recall']}")
        print(f"  F1:        {report['summary']['context_f1']}")
        print(f"  Hit Rate:  {report['summary']['hit_rate']}")
        print(f"  MRR:       {report['summary']['mrr']}")
        print(f"  NDCG:      {report['summary']['ndcg']}")
        print(f"  Latency:   {report['summary']['avg_latency_ms']}ms")

    # 汇总对比
    print("\n" + "=" * 100)
    print(f"{'策略':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'HitRate':>8} {'MRR':>8} {'NDCG':>8} {'延迟ms':>8}")
    print("=" * 100)
    for config_name in ROUTED_CONFIGS:
        with open(f"eval/rag_report_{config_name}.json", "r", encoding="utf-8") as f:
            r = json.load(f)["summary"]
        print(f"{config_name:<15} {r['context_precision']:>10.4f} {r['context_recall']:>10.4f} {r['context_f1']:>10.4f} "
              f"{r['hit_rate']:>8.4f} {r['mrr']:>8.4f} {r['ndcg']:>8.4f} {r['avg_latency_ms']:>8.1f}")
    print("=" * 100)


if __name__ == "__main__":
    asyncio.run(main())
