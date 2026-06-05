#!/usr/bin/env python3
"""对 run5_truncation_fix 的结果运行 Judge 评估"""
import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime as dt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.batch_eval_runner import run_judge_for_case

RUN_DIR = Path(__file__).resolve().parent / "results" / "run5_truncation_fix"

async def main():
    # 加载所有结果
    results = []
    for fpath in sorted(RUN_DIR.glob("eval_*.json")):
        with open(fpath, "r", encoding="utf-8") as f:
            results.append(json.load(f))

    print(f"加载结果: {len(results)} 条")

    # 调用 Judge
    judge_results = []
    for idx, r in enumerate(results, 1):
        case_id = r["case_id"]
        if r.get("status") != "success":
            judge_results.append({
                "case_id": case_id, "resolved": False,
                "reason": f"Case 执行失败: {r.get('error', 'unknown')}",
            })
            print(f"  [{idx}] {case_id} -> ❌ 执行失败")
            continue

        try:
            jr = await run_judge_for_case(r)
            judge_results.append(jr)
            scores = jr.get("scores", {})
            print(f"  [{idx}] {case_id} -> resolved={jr['resolved']} "
                  f"resp_acc={scores.get('response_accuracy', 0)}/5 "
                  f"faith={scores.get('faithfulness', 0)}/5")
        except Exception as e:
            import traceback
            traceback.print_exc()
            judge_results.append({
                "case_id": case_id, "resolved": False,
                "reason": f"Judge 失败: {e}",
            })

    # 保存报告
    resolved_count = sum(1 for j in judge_results if j["resolved"])
    report = {
        "summary": {
            "total_cases": len(judge_results),
            "resolved_count": resolved_count,
            "resolved_rate": round(resolved_count / len(judge_results), 4) if judge_results else 0,
        },
        "cases": judge_results,
        "timestamp": dt.now().isoformat(),
    }
    with open(RUN_DIR / "_report_judge.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 前后对比
    print(f"\n{'='*70}")
    print("【前后对比】")
    print(f"{'='*70}")

    old_results = {}
    old_path = Path(__file__).resolve().parent / "results" / "run5" / "_report_judge.json"
    if old_path.exists():
        with open(old_path, "r", encoding="utf-8") as f:
            old_report = json.load(f)
        for c in old_report.get("cases", []):
            old_results[c["case_id"]] = c

    improved = 0
    for jr in judge_results:
        cid = jr["case_id"]
        old = old_results.get(cid, {})
        old_res = old.get("resolved", False)
        new_res = jr["resolved"]
        old_str = "✅通过" if old_res else "❌失败"
        new_str = "✅通过" if new_res else "❌失败"
        marker = " 🎉" if (not old_res and new_res) else ""
        if not old_res and new_res:
            improved += 1
        print(f"  {cid}: {old_str} -> {new_str}{marker}")
        if not new_res:
            print(f"    原因: {jr.get('reason', '')[:150]}")

    print(f"\n  改善: {improved}/{len(judge_results)} 个 case 从失败变为通过")
    print(f"  当前通过率: {resolved_count}/{len(judge_results)} = {report['summary']['resolved_rate']:.2%}")
    print(f"  报告保存: {RUN_DIR / '_report_judge.json'}")

if __name__ == "__main__":
    asyncio.run(main())
