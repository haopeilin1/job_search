#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chunked runner for intent_http_eval.py — processes one group per invocation
to stay within shell timeout limits.
"""

import asyncio
import json
import sys
from pathlib import Path
from collections import defaultdict

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intent_http_eval import (
    load_dataset,
    load_resumes,
    get_resume_id,
    eval_single_case,
    BASE_URL,
)

PROGRESS_FILE = Path(__file__).resolve().parent / "intent_http_progress.json"
RESULT_LOG = Path(__file__).resolve().parent / "intent_http_result.log"


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed_groups": [], "results": []}


def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def append_log(text: str):
    with open(RESULT_LOG, "a", encoding="utf-8") as f:
        f.write(text + "\n")


async def main():
    cases = load_dataset()
    resumes = load_resumes()

    multi_turn_cases = [c for c in cases if c.get("session_group")]
    groups = defaultdict(list)
    for c in multi_turn_cases:
        groups[c["session_group"]].append(c)

    progress = load_progress()
    completed = set(progress["completed_groups"])
    remaining = [g for g in sorted(groups.keys()) if g not in completed]

    print(f"多轮对话用例: {len(multi_turn_cases)} 条, 剩余 {len(remaining)} 个 group\n")
    append_log(f"多轮对话用例: {len(multi_turn_cases)} 条, 剩余 {len(remaining)} 个 group")

    if not remaining:
        print("所有 group 已完成，生成最终报告...\n")
        await print_final_report(progress["results"])
        return

    group_id = remaining[0]
    group_cases = sorted(groups[group_id], key=lambda x: x["session_id"])
    print(f"Group: {group_id}")
    append_log(f"Group: {group_id}")

    group_results = []
    group_session_map = {}

    async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
        for idx, case in enumerate(group_cases):
            reset_session = True
            sid = case["session_id"]

            if group_id in group_session_map:
                reset_session = False
                sid = group_session_map[group_id]
                print(f"  -> 复用 session: {sid}")
                append_log(f"  -> 复用 session: {sid}")
            else:
                group_session_map[group_id] = sid

            resume_id = get_resume_id(case, resumes)
            r = await eval_single_case(client, case, sid, reset_session, resume_id)
            group_results.append(r)

            mark = "OK" if r["match"] else "XX"
            line = f"  {mark} {r['case_id']}: gold={r['gold_intents']} pred={r['pred_intents']}"
            print(line)
            append_log(line)
            if r["error"]:
                err_line = f"     ERROR: {r['error']}"
                print(err_line)
                append_log(err_line)

    progress["completed_groups"].append(group_id)
    progress["results"].extend(group_results)
    save_progress(progress)

    print(f"\nGroup {group_id} 完成. 剩余 {len(remaining) - 1} 个 group.")
    append_log(f"\nGroup {group_id} 完成. 剩余 {len(remaining) - 1} 个 group.")

    if len(remaining) == 1:
        print("\n所有 group 已完成，生成最终报告...\n")
        append_log("\n所有 group 已完成，生成最终报告...")
        await print_final_report(progress["results"])


async def print_final_report(results):
    results.sort(key=lambda r: (r.get("group") or "", r["case_id"]))

    total = len(results)
    correct = sum(1 for r in results if r["match"])

    header = "\n" + "=" * 110
    print(header)
    append_log(header)
    line = f"{'Case ID':<18} {'Group':<10} {'Match':<6} {'Gold Intents':<25} {'Pred Intents':<25} {'Message'}"
    print(line)
    append_log(line)
    print("=" * 110)
    append_log("=" * 110)
    for r in results:
        gold_str = ", ".join(r["gold_intents"])
        pred_str = ", ".join(r["pred_intents"])
        match_mark = "PASS" if r["match"] else "FAIL"
        msg = r["message"][:30] + "..." if len(r["message"]) > 30 else r["message"]
        line = f"{r['case_id']:<18} {r.get('group',''):<10} {match_mark:<6} {gold_str:<25} {pred_str:<25} {msg}"
        print(line)
        append_log(line)
    print("=" * 110)
    append_log("=" * 110)
    summary = f"\n总计: {total} 条 | 命中: {correct} 条 | 准确率: {correct / total * 100:.1f}%"
    print(summary)
    append_log(summary)

    misses = [r for r in results if not r["match"]]
    if misses:
        detail = f"\n未命中详情 ({len(misses)} 条):"
        print(detail)
        append_log(detail)
        for r in misses:
            line1 = f"  {r['case_id']}: gold={r['gold_intents']} pred={r['pred_intents']}"
            line2 = f"    msg: {r['message']}"
            print(line1)
            print(line2)
            append_log(line1)
            append_log(line2)


if __name__ == "__main__":
    asyncio.run(main())
