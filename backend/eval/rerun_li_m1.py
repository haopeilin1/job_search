#!/usr/bin/env python3
"""重跑 li_m1 group (li_02 + li_11)"""
import asyncio
import aiohttp
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.batch_eval_runner import run_single_case, _session_group_map, BASE_URL

DATASET_PATH = Path(__file__).resolve().parent / "test_dataset.jsonl"
RUN_DIR = Path(__file__).resolve().parent / "results" / "run5_truncation_fix"
RUN_DIR.mkdir(parents=True, exist_ok=True)

async def main():
    # 读取 li_m1 group 的 case（保持顺序）
    cases = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                c = json.loads(line)
                if c.get("session_group") == "li_m1":
                    cases.append(c)

    print(f"li_m1 group case 数: {len(cases)}")
    for c in cases:
        print(f"  - {c['session_id']}")

    _session_group_map.clear()

    timeout = aiohttp.ClientTimeout(total=600, connect=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for case in cases:
            case_id = case["session_id"]
            print(f"\n>>> 测试: {case_id} | {case['message'][:50]}...")
            result = await run_single_case(case, RUN_DIR, session)
            status = "✅" if result["status"] == "success" else "❌"
            print(f"  {status} status={result['status']} | reply_len={result.get('reply_length', 0)}")
            if result.get("error"):
                print(f"  ⚠️ error: {result['error'][:100]}")

    print("\nli_m1 group 重跑完成")

if __name__ == "__main__":
    asyncio.run(main())
