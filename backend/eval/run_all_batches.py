#!/usr/bin/env python3
"""
全量评测分批执行脚本
- 单轮本地调用，多轮 HTTP 调用
- 每批次独立保存，不覆盖
- 实时追加日志
- 最后自动合并报告
"""

import asyncio
import json
import subprocess
import sys
import time
import os
from pathlib import Path
from datetime import datetime

EVAL_DIR = Path(__file__).parent
BACKEND_DIR = EVAL_DIR.parent
PYTHON = BACKEND_DIR / "venv" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = "python"

BATCHES = ["gen", "sup", "wang", "li", "chen"]

# 运行编号，用于区分不同次运行（避免覆盖文件）
RUN_ID = datetime.now().strftime("%m%d_%H%M")


def log(msg: str):
    ts = datetime.now().isoformat()
    line = f"[{ts}] {msg}"
    print(line)
    sys.stdout.flush()


def run_batch(batch: str) -> dict:
    """运行单个批次，返回 metrics"""
    output_file = EVAL_DIR / f"v3_{RUN_ID}_{batch}.json"
    log_file = EVAL_DIR / f"v3_{RUN_ID}_{batch}.log"

    log(f"=" * 60)
    log(f"开始批次: {batch}")
    log(f"=" * 60)

    cmd = [
        str(PYTHON), "-u", str(EVAL_DIR / "run_eval_v3.py"),
        "--batch", batch,
        "--output", str(output_file),
    ]

    start = time.time()
    with open(log_file, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=str(BACKEND_DIR),
        )
        try:
            proc.wait(timeout=7200)  # 单批次最多2小时
        except subprocess.TimeoutExpired:
            proc.kill()
            log(f"[{batch}] 超时杀死!")
            return {"error": "timeout", "batch": batch}

    elapsed = time.time() - start
    exit_code = proc.returncode

    if exit_code != 0:
        log(f"[{batch}] 异常退出, code={exit_code}")
        return {"error": f"exit_code={exit_code}", "batch": batch}

    # 读取结果
    metrics = {}
    if output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            metrics = data.get("metrics", {})
            case_count = len(data.get("cases", []))
            success_rate = metrics.get("outcome", {}).get("task_success_rate", 0)
            log(f"[{batch}] 完成 | cases={case_count} | success_rate={success_rate:.1%} | time={elapsed:.0f}s")
        except Exception as e:
            log(f"[{batch}] 读取结果失败: {e}")
            metrics = {"error": str(e)}
    else:
        log(f"[{batch}] 结果文件未生成!")
        metrics = {"error": "no_output"}

    metrics["_batch"] = batch
    metrics["_elapsed_sec"] = round(elapsed, 1)
    return metrics


def merge_reports() -> Path:
    """合并所有批次报告为一个总报告"""
    all_cases = []
    batch_summaries = []

    for batch in BATCHES:
        output_file = EVAL_DIR / f"v3_{RUN_ID}_{batch}.json"
        if not output_file.exists():
            continue
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            cases = data.get("cases", [])
            all_cases.extend(cases)
            batch_summaries.append({
                "batch": batch,
                "case_count": len(cases),
                "metrics": data.get("metrics", {}),
            })
        except Exception as e:
            log(f"合并时读取 {batch} 失败: {e}")

    merged = {
        "total_cases": len(all_cases),
        "batches": batch_summaries,
        "cases": all_cases,
        "merged_at": datetime.now().isoformat(),
    }

    merged_file = EVAL_DIR / f"v3_merged_{int(time.time())}.json"
    with open(merged_file, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    log(f"合并报告已保存: {merged_file}")
    return merged_file


def main():
    log(f"全量评测开始 | 时间: {datetime.now().isoformat()}")
    log(f"批次顺序: {BATCHES}")

    all_metrics = []
    for batch in BATCHES:
        metrics = run_batch(batch)
        all_metrics.append(metrics)

    log(f"=" * 60)
    log("所有批次执行完毕，开始合并报告...")
    merged = merge_reports()

    # 保存执行摘要
    summary = {
        "started_at": datetime.now().isoformat(),
        "batches": BATCHES,
        "results": all_metrics,
        "merged_file": str(merged),
    }
    summary_file = EVAL_DIR / f"v3_summary_{int(time.time())}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log(f"执行摘要已保存: {summary_file}")
    log("[DONE] 全量评测完成!")


if __name__ == "__main__":
    main()
