"""
评估执行脚本 —— 批量运行测试用例，收集API响应与埋点事件。

用法：
    cd backend && python eval/run_eval.py
    cd backend && python eval/run_eval.py --batch A          # 只跑A批次
    cd backend && python eval/run_eval.py --case eval_a_t04  # 只跑单条
    cd backend && python eval/run_eval.py --batch E --output eval/results_E.jsonl
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.core import state as app_state

EVAL_DIR = Path(__file__).parent
DATASET_FILE = EVAL_DIR / "test_dataset.jsonl"
RESUMES_FILE = EVAL_DIR / "test_resumes.json"
CHAT_API = "http://localhost:8001/api/v1/chat"
DEFAULT_TIMEOUT = 60


def load_dataset(batch: Optional[str] = None, case_id: Optional[str] = None) -> List[dict]:
    cases = []
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))

    if case_id:
        cases = [c for c in cases if c["session_id"] == case_id]
    elif batch:
        batches = [b.strip().lower() for b in batch.split(",")]
        cases = [c for c in cases if any(c["session_id"].startswith(f"eval_{b}_") for b in batches)]

    return cases


def load_resumes() -> dict:
    with open(RESUMES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def inject_resume(resume_id: str, resume_data: dict):
    """将测试简历注入全局状态（绕过LLM解析，确保确定性）。"""
    text = resume_data["text"]
    structured = resume_data.get("structured", {})

    app_state.resumes_db[resume_id] = {
        "resume_id": resume_id,
        "is_active": False,
        "source_type": "eval",
        "parsed_schema": {
            "meta": {"raw_text": text},
            "basic_info": {
                "name": resume_data.get("name", ""),
                "years_exp": structured.get("years_of_experience"),
            },
            "skills": {
                "technical": structured.get("hard_skills", []),
                "soft": structured.get("soft_skills", []),
            },
        }
    }


def activate_resume(resume_id: str):
    if resume_id not in app_state.resumes_db:
        raise ValueError(f"简历 {resume_id} 未找到")
    app_state.active_resume_id = resume_id
    app_state.save_resumes()


async def run_case(case: dict) -> dict:
    payload = {
        "session_id": case["session_id"],
        "message": case["message"],
        "eval_context": case.get("eval_context", {}),
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        start = time.time()
        try:
            resp = await client.post(CHAT_API, json=payload)
            latency_ms = round((time.time() - start) * 1000, 2)
            return {
                "session_id": case["session_id"],
                "message": case["message"],
                "status_code": resp.status_code,
                "latency_ms": latency_ms,
                "response": resp.json(),
                "error": None,
            }
        except Exception as e:
            latency_ms = round((time.time() - start) * 1000, 2)
            return {
                "session_id": case["session_id"],
                "message": case["message"],
                "status_code": 0,
                "latency_ms": latency_ms,
                "response": None,
                "error": str(e),
            }


async def main():
    parser = argparse.ArgumentParser(description="求职雷达Agent评估执行脚本")
    parser.add_argument("--batch", type=str, default=None, help="只跑指定批次（A-F），多个用逗号分隔如 A,B,F")
    parser.add_argument("--case", type=str, default=None, help="只跑指定用例（如 eval_a_t01）")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    parser.add_argument("--use-active-resume", action="store_true", default=False, help="使用当前活跃简历（不注入测试简历）")
    args = parser.parse_args()

    cases = load_dataset(batch=args.batch, case_id=args.case)
    if not cases:
        print("[ERROR] 没有找到匹配的测试用例")
        return

    resumes_data = load_resumes()
    resumes_by_id = {r["id"]: r for r in resumes_data["resumes"]}
    session_resume_map = resumes_data.get("session_resume_map", {})

    # ── 备份用户原始简历状态 ──
    original_resumes = dict(app_state.resumes_db)
    original_active_id = app_state.active_resume_id
    print(f"[INFO] 备份原始简历状态 | 共{len(original_resumes)}份 | 当前活跃: {original_active_id}")

    try:
        if args.use_active_resume:
            # 使用当前活跃简历，不注入测试简历
            print(f"[INFO] 使用当前活跃简历: {original_active_id}")
        else:
            # 注入所有测试简历
            for rid, rdata in resumes_by_id.items():
                inject_resume(rid, rdata)
            app_state.save_resumes()
            print(f"[INFO] 已注入 {len(resumes_by_id)} 份测试简历")

        # 按session_id排序，确保同一session连续执行
        cases.sort(key=lambda c: c["session_id"])

        results = []
        current_resume_id = None

        for i, case in enumerate(cases, 1):
            sid = case["session_id"]
            needed_resume = session_resume_map.get(sid)

            if args.use_active_resume:
                # 始终使用当前活跃简历，不切换
                pass
            elif needed_resume and needed_resume != current_resume_id:
                activate_resume(needed_resume)
                current_resume_id = needed_resume
                print(f"[INFO] 激活简历: {needed_resume} ({resumes_by_id[needed_resume]['label']})")
            elif not args.use_active_resume and not needed_resume and current_resume_id:
                app_state.active_resume_id = None
                current_resume_id = None
                app_state.save_resumes()
                print("[INFO] 清除活跃简历（空简历场景）")

            print(f"[{i}/{len(cases)}] {sid}: {case['message'][:40]}...")
            result = await run_case(case)
            results.append({**result, "case": case})

            if result["error"]:
                print(f"  [ERROR] {result['error']}")
            else:
                print(f"  [OK] {result['latency_ms']}ms | status={result['status_code']}")

            await asyncio.sleep(0.5)

    finally:
        # ── 恢复用户原始简历状态 ──
        app_state.resumes_db = original_resumes
        app_state.active_resume_id = original_active_id
        app_state.save_resumes()
        print(f"[INFO] 已恢复原始简历状态 | 活跃简历: {original_active_id}")

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(args.output) if args.output else EVAL_DIR / f"eval_results_{timestamp}.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    # 统计
    success = sum(1 for r in results if r["status_code"] == 200)
    errors = sum(1 for r in results if r["error"])
    avg_latency = sum(r["latency_ms"] for r in results) / len(results) if results else 0
    print(f"\n[INFO] 评估完成: {len(results)} 条用例")
    print(f"[INFO] 结果保存: {output_file}")
    print(f"[INFO] 成功率: {success}/{len(results)} | 错误: {errors} | 平均延迟: {avg_latency:.0f}ms")


if __name__ == "__main__":
    asyncio.run(main())
