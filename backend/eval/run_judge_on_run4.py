#!/usr/bin/env python3
"""
对 run4 的已有结果补跑 v3.5 Judge 评估。
从 test_dataset.jsonl 补全缺失的 eval_context 和 resume_id。

用法:
    cd backend && python eval/run_judge_on_run4.py
输出:
    eval/results/run4/_report_judge_v35.json
"""
import asyncio
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.judge_postprocess import judge_single_case, _build_case_context

RUN_DIR = Path(__file__).resolve().parent / "results" / "run4"
DATASET_FILE = Path(__file__).resolve().parent / "test_dataset.jsonl"
OUTPUT_FILE = RUN_DIR / "_report_judge_v35.json"


def load_test_cases() -> dict:
    """加载测试集，建立 case_id -> case 的映射"""
    cases = {}
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                c = json.loads(line)
                cases[c["session_id"]] = c
    return cases


def load_run4_results() -> list:
    """加载 run4 的所有 case 结果"""
    results = []
    for fpath in sorted(RUN_DIR.glob("eval_*.json")):
        with open(fpath, "r", encoding="utf-8") as f:
            results.append(json.load(f))
    return results


async def main():
    print("=" * 60)
    print("Run4 v3.5 Judge 补跑")
    print("=" * 60)

    test_cases = load_test_cases()
    run_results = load_run4_results()
    print(f"加载测试集: {len(test_cases)} 条")
    print(f"加载 run4 结果: {len(run_results)} 条")

    judge_results = []

    for idx, r in enumerate(run_results, 1):
        case_id = r["case_id"]
        test_case = test_cases.get(case_id, {})

        # 补全缺失字段
        enriched = dict(r)
        enriched["eval_context"] = test_case.get("eval_context", {})
        enriched["resume_id"] = test_case.get("resume_id", "")
        enriched.setdefault("gold_intents", test_case.get("eval_context", {}).get("gold_intents", []))

        # 构建 Judge 上下文
        ctx = _build_case_context(case_id, enriched)

        # 调用 Judge
        jr = await judge_single_case(enriched, ctx)

        scores = jr.get("scores", {})
        resolved = jr.get("resolved", False)
        veto = jr.get("veto", False)
        rule_hit = jr.get("rule_hit")
        needs_rag = jr.get("needs_rag", False)

        judge_results.append({
            "case_id": case_id,
            "message": enriched.get("message", "")[:60],
            "scenario": enriched.get("scenario", ""),
            "resolved": resolved,
            "veto": veto,
            "rule_hit": rule_hit,
            "needs_rag": needs_rag,
            "scores": scores,
            "reason": jr.get("reason", "")[:200],
        })

        status = "✅" if resolved else "❌"
        print(f"[{idx:2d}/{len(run_results)}] {case_id} {status} resolved={resolved} "
              f"intent={scores.get('intent_accuracy',0)}/5 "
              f"faithfulness={scores.get('faithfulness',0)}/5 "
              f"relevance={scores.get('answer_relevance',0)}/5 "
              f"needs_rag={needs_rag}")

    # 汇总统计
    total = len(judge_results)
    resolved_count = sum(1 for j in judge_results if j["resolved"])
    veto_count = sum(1 for j in judge_results if j["veto"])
    rag_cases = [j for j in judge_results if j["needs_rag"]]
    non_rag_cases = [j for j in judge_results if not j["needs_rag"]]

    dim_avg = {}
    dim_keys = [
        "intent_accuracy", "slot_accuracy", "tool_correctness", "tool_execution",
        "response_accuracy", "response_completeness", "citation_quality",
        "coherence", "tone", "efficiency", "faithfulness", "answer_relevance"
    ]
    for k in dim_keys:
        vals = [j["scores"].get(k, 0) for j in judge_results]
        dim_avg[k] = round(sum(vals) / len(vals), 2) if vals else 0

    # RAG 子集统计
    rag_dim_avg = {}
    if rag_cases:
        for k in ["faithfulness", "answer_relevance"]:
            vals = [j["scores"].get(k, 0) for j in rag_cases]
            rag_dim_avg[k] = round(sum(vals) / len(vals), 2) if vals else 0

    summary = {
        "total_cases": total,
        "resolved_count": resolved_count,
        "resolved_rate": round(resolved_count / total, 4) if total else 0,
        "veto_count": veto_count,
        "veto_rate": round(veto_count / total, 4) if total else 0,
        "rag_cases_count": len(rag_cases),
        "non_rag_cases_count": len(non_rag_cases),
        "dimension_averages": dim_avg,
        "rag_dimension_averages": rag_dim_avg,
    }

    report = {
        "summary": summary,
        "cases": judge_results,
        "timestamp": __import__("datetime").datetime.now().isoformat(),
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("Judge 评估完成")
    print("=" * 60)
    print(f"总 case 数: {total}")
    print(f"Judge 通过率 (resolved): {resolved_count}/{total} = {summary['resolved_rate']:.2%}")
    print(f"否决项触发: {veto_count}/{total} = {summary['veto_rate']:.2%}")
    print(f"RAG case 数: {len(rag_cases)}")
    print(f"非 RAG case 数: {len(non_rag_cases)}")
    print()
    print("【12 维平均分】")
    for k, v in dim_avg.items():
        print(f"  {k}: {v}/5")
    if rag_dim_avg:
        print()
        print("【RAG 专属维度平均分】")
        for k, v in rag_dim_avg.items():
            print(f"  {k}: {v}/5")
    print(f"\n报告保存: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
