"""
RAG k值消融实验（精简版：只跑 hybrid 和 hybrid_rerank）
"""

import asyncio
import json
import sys
sys.path.insert(0, "..")

from eval.run_rag_eval import run_eval


async def main():
    strategies = ['hybrid', 'hybrid_rerank']
    top_ks = [5, 10, 15, 20, 30]

    results = []
    for strategy in strategies:
        for k in top_ks:
            print(f"\n>>> Running {strategy} top_k={k} ...")
            output = f"eval/rag_report_{strategy}_k{k}.json"
            await run_eval(strategy, k, output)
            with open(output, "r", encoding="utf-8") as f:
                report = json.load(f)
            results.append({
                "strategy": strategy,
                "top_k": k,
                **report["summary"]
            })

    print("\n" + "=" * 100)
    header = f"{'策略':<15} {'k':>3} {'Precision':>10} {'Recall':>10} {'F1':>10} {'HitRate':>8} {'MRR':>8} {'NDCG':>8} {'延迟ms':>10}"
    print(header)
    print("=" * 100)
    for r in results:
        line = (
            f"{r['strategy']:<15} {r['top_k']:>3} "
            f"{r['context_precision']:>10.4f} {r['context_recall']:>10.4f} {r['context_f1']:>10.4f} "
            f"{r['hit_rate']:>8.4f} {r['mrr']:>8.4f} {r['ndcg']:>8.4f} {r['avg_latency_ms']:>10.1f}"
        )
        print(line)
    print("=" * 100)

    with open("eval/rag_k_ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nSaved to eval/rag_k_ablation_summary.json")


if __name__ == "__main__":
    asyncio.run(main())
