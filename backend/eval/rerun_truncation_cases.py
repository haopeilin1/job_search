#!/usr/bin/env python3
"""
重跑因 Judge 简历截断导致误判的 case
- 单轮：chen_02, chen_04, li_07, li_10
- 多轮 group：li_m1 (li_02 + li_11)

运行方式：
    cd /home/hpl/job_search/backend
    python eval/rerun_truncation_cases.py
"""

import asyncio
import aiohttp
import json
import time
import sys
import traceback
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.batch_eval_runner import (
    run_single_case, run_judge_for_case,
    _session_group_map, BASE_URL
)

DATASET_PATH = Path(__file__).resolve().parent / "test_dataset.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RUN_DIR = RESULTS_DIR / "run5_truncation_fix"

# 目标 case（包含多轮对话 group 的所有成员）
TARGET_CASES = {
    "eval_chen_02", "eval_chen_04",
    "eval_li_02", "eval_li_11",   # li_m1 group
    "eval_li_07", "eval_li_10",
}


async def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    # 读取测试集
    cases = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    # 筛选目标 case（保持原始顺序，确保多轮 group 按顺序执行）
    target_cases = [c for c in cases if c["session_id"] in TARGET_CASES]

    print(f"目标 case 数: {len(target_cases)}")
    for c in target_cases:
        grp = c.get("session_group")
        print(f"  - {c['session_id']} | group={grp}")

    # 清空 session_group 映射（确保从头开始）
    _session_group_map.clear()

    # 检查 backend
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{BASE_URL}/docs") as r:
                if r.status == 200:
                    print(f"✅ Backend 正常: {BASE_URL}")
                else:
                    print(f"⚠️ Backend 状态: {r.status}")
    except Exception as e:
        print(f"❌ Backend 检查失败: {e}")
        return

    timeout = aiohttp.ClientTimeout(total=600, connect=30)
    results = []

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for idx, case in enumerate(target_cases, 1):
            case_id = case["session_id"]
            print(f"\n[{idx}/{len(target_cases)}] 测试: {case_id} | {case['message'][:50]}...")

            result = await run_single_case(case, RUN_DIR, session)
            results.append(result)

            status_icon = "✅" if result["status"] == "success" else "❌"
            intent_ok = "✅" if result.get("intent_match") else "❌"
            tool_rate = result.get("tool_match_rate", 0)
            reply_ok = "✅" if result.get("has_reply") else "❌"
            latency = result.get("total_latency", 0)
            kb_count = len(result.get("kb_chunks", []))

            print(f"  {status_icon} status={result['status']} | intent={intent_ok} | tool_match={tool_rate} | reply={reply_ok} | latency={latency}s | kb={kb_count}")
            if result.get("error"):
                print(f"  ⚠️  error: {result['error'][:100]}")

    # 保存汇总
    with open(RUN_DIR / "_all_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 调用 Judge
    print(f"\n{'='*70}")
    print("【Judge 评估开始】")
    print(f"{'='*70}")

    judge_results = []
    for idx, r in enumerate(results, 1):
        case_id = r["case_id"]
        if r.get("status") != "success":
            judge_results.append({
                "case_id": case_id,
                "resolved": False,
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
            print(f"  [{idx}] {case_id} -> Judge 失败: {e}")
            traceback.print_exc()
            judge_results.append({
                "case_id": case_id,
                "resolved": False,
                "reason": f"Judge 调用失败: {e}",
            })

    # 保存 Judge 报告
    resolved_count = sum(1 for j in judge_results if j["resolved"])
    report = {
        "summary": {
            "total_cases": len(judge_results),
            "resolved_count": resolved_count,
            "resolved_rate": round(resolved_count / len(judge_results), 4) if judge_results else 0,
        },
        "cases": judge_results,
        "timestamp": datetime.now().isoformat(),
    }
    with open(RUN_DIR / "_report_judge.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print("【重跑完成】")
    print(f"{'='*70}")
    print(f"  总 case: {len(judge_results)}")
    print(f"  Judge 通过: {resolved_count}/{len(judge_results)} = {report['summary']['resolved_rate']:.2%}")
    print(f"  结果保存: {RUN_DIR}")

    # 打印前后对比
    print(f"\n【前后对比】")
    old_results = {}
    old_report_path = RESULTS_DIR / "run5" / "_report_judge.json"
    if old_report_path.exists():
        with open(old_report_path, "r", encoding="utf-8") as f:
            old_report = json.load(f)
        for c in old_report.get("cases", []):
            old_results[c["case_id"]] = c

    for jr in judge_results:
        cid = jr["case_id"]
        old = old_results.get(cid, {})
        old_res = "✅通过" if old.get("resolved") else "❌失败"
        new_res = "✅通过" if jr["resolved"] else "❌失败"
        print(f"  {cid}: {old_res} -> {new_res}")
        if not jr["resolved"]:
            print(f"    原因: {jr.get('reason', '')[:120]}")


if __name__ == "__main__":
    asyncio.run(main())
