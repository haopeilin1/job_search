#!/usr/bin/env python3
"""
批量全链路评测脚本 — 三轮稳定性测试
要求：
1. 每条保存完整过程及中间结果，保证可追溯
2. 每次跑完计算全部评测指标
3. 每条从前端 SSE 模拟真实用户
4. 共跑三轮验证稳定性
5. setsid+nohup 后台运行，SSH 断网不中断

用法：
    cd /home/hpl/job_search/backend
    setsid nohup python eval/batch_eval_runner.py --runs 3 > eval/results/runner.log 2>&1 &
"""

import asyncio
import aiohttp
import json
import time
import sys
import traceback
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE_URL = "http://127.0.0.1:8001"
STREAM_URL = f"{BASE_URL}/api/v1/chat/stream"
DATASET_PATH = Path(__file__).resolve().parent / "test_dataset.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# 意图别名映射
INTENT_ALIASES = {
    "position_explore": "explore",
    "match_assess": "assess",
    "interview_prepare": "prepare",
    "general_chat": "chat",
    "explore": "explore",
    "assess": "assess",
    "prepare": "prepare",
    "verify": "verify",
    "clarification": "clarification",
    "chat": "chat",
}


def _normalize_intent(intent: str) -> str:
    return INTENT_ALIASES.get(intent, intent)


def _normalize_intents(intents: List[str]) -> List[str]:
    return sorted(set(_normalize_intent(i) for i in intents))


async def run_single_case(case: dict, run_dir: Path, session: aiohttp.ClientSession) -> dict:
    """
    对单条 case 发送 SSE 流式请求，保存完整过程，返回评测结果。
    """
    case_id = case["session_id"]
    message = case["message"]
    resume_id = case.get("resume_id", "")
    gold_ctx = case.get("eval_context", {})
    gold_intents = _normalize_intents(gold_ctx.get("gold_intents", []))
    expected_tools = gold_ctx.get("expected_tools", [])

    result = {
        "case_id": case_id,
        "message": message,
        "gold_intents": gold_intents,
        "expected_tools": expected_tools,
        "scenario": gold_ctx.get("scenario", ""),
        "status": "pending",
        "error": None,
        "ttfb": None,
        "total_latency": None,
        "event_count": 0,
        "pred_intent": None,
        "pred_tools": [],
        "tools_called": [],
        "reply": "",
        "reply_length": 0,
        "intent_match": False,
        "tool_match_rate": 0.0,
        "has_reply": False,
        "events": [],
        "timestamp": datetime.now().isoformat(),
    }

    events = []
    t0 = time.time()
    ttfb = None

    payload = {
        "message": message,
        "session_id": f"{case_id}_{int(t0*1000)}",
        "resume_id": resume_id,
        "stream": True,
    }

    try:
        async with session.post(STREAM_URL, json=payload) as resp:
            ttfb = time.time() - t0
            result["ttfb"] = round(ttfb, 3)

            current_event_type = None
            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line:
                    continue
                if line.startswith("event: "):
                    current_event_type = line[7:]
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        events.append({"type": "done", "_time": time.time() - t0})
                        break
                    try:
                        data = json.loads(data_str)
                        data["_time"] = time.time() - t0
                        if current_event_type:
                            data["type"] = current_event_type
                        events.append(data)
                    except json.JSONDecodeError:
                        pass

        result["total_latency"] = round(time.time() - t0, 2)
        result["event_count"] = len(events)
        result["events"] = events

        # 提取 done 事件
        done_events = [e for e in events if e.get("type") == "done"]
        done_event = done_events[0] if done_events else {}

        # 预测意图
        pred_intent = done_event.get("intent")
        result["pred_intent"] = pred_intent
        result["intent_match"] = _normalize_intent(pred_intent) in gold_intents if pred_intent else False

        # 工具调用
        agent_tools = done_event.get("agent", {}).get("tools", []) if isinstance(done_event.get("agent"), dict) else []
        tools_called = [t.get("tool") for t in agent_tools if t.get("tool")]
        result["tools_called"] = tools_called
        result["pred_tools"] = agent_tools

        # 工具匹配率
        if expected_tools:
            matched = sum(1 for et in expected_tools if any(et in ct for ct in tools_called))
            result["tool_match_rate"] = round(matched / len(expected_tools), 2)
        else:
            result["tool_match_rate"] = 1.0

        # 工具执行成功率（实际调用的工具中 status=✅ 的比例）
        if agent_tools:
            success_tools = [t for t in agent_tools if t.get("status") == "✅"]
            result["tool_execution_success_rate"] = round(len(success_tools) / len(agent_tools), 2)
        else:
            result["tool_execution_success_rate"] = 0.0

        # 工具调用正确率（期望工具中，被调用且执行成功的比例）
        if expected_tools:
            correct_count = sum(1 for et in expected_tools if any(
                et in t.get("tool", "") and t.get("status") == "✅"
                for t in agent_tools
            ))
            result["tool_correct_rate"] = round(correct_count / len(expected_tools), 2)
        else:
            result["tool_correct_rate"] = 1.0

        # 回复
        reply = done_event.get("reply", {}).get("content", "") if isinstance(done_event.get("reply"), dict) else str(done_event.get("reply", ""))
        result["reply"] = reply
        result["reply_length"] = len(reply)
        result["has_reply"] = len(reply) > 50

        result["status"] = "success"

    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {str(e)}"
        result["total_latency"] = round(time.time() - t0, 2)
        traceback.print_exc()

    # 保存单条详细结果
    case_file = run_dir / f"{case_id}.json"
    with open(case_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def compute_metrics(results: List[dict]) -> dict:
    """计算本轮全部评测指标"""
    total = len(results)
    success_results = [r for r in results if r["status"] == "success"]
    error_results = [r for r in results if r["status"] == "error"]

    # 意图匹配
    intent_match_count = sum(1 for r in success_results if r["intent_match"])
    intent_match_rate = round(intent_match_count / len(success_results), 3) if success_results else 0

    # 工具匹配
    tool_match_rates = [r["tool_match_rate"] for r in success_results]
    avg_tool_match = round(sum(tool_match_rates) / len(tool_match_rates), 3) if tool_match_rates else 0
    full_tool_match = sum(1 for r in success_results if r["tool_match_rate"] >= 1.0)
    full_tool_match_rate = round(full_tool_match / len(success_results), 3) if success_results else 0

    # 工具执行成功率
    tool_exec_rates = [r.get("tool_execution_success_rate", 0) for r in all_results]
    avg_tool_exec_success = round(sum(tool_exec_rates) / len(tool_exec_rates), 3) if tool_exec_rates else 0

    # 工具调用正确率
    tool_correct_rates = [r.get("tool_correct_rate", 0) for r in all_results]
    avg_tool_correct = round(sum(tool_correct_rates) / len(tool_correct_rates), 3) if tool_correct_rates else 0

    # 回复完成
    has_reply_count = sum(1 for r in success_results if r["has_reply"])
    reply_rate = round(has_reply_count / len(success_results), 3) if success_results else 0

    # 延迟
    ttfs = [r["ttfb"] for r in success_results if r["ttfb"] is not None]
    latencies = [r["total_latency"] for r in success_results if r["total_latency"] is not None]

    # 按意图统计
    intent_stats = defaultdict(lambda: {"total": 0, "match": 0, "success": 0})
    for r in success_results:
        for intent in r.get("gold_intents", []):
            intent_stats[intent]["total"] += 1
            intent_stats[intent]["success"] += 1
            if r["intent_match"]:
                intent_stats[intent]["match"] += 1

    # 按场景统计
    scenario_stats = defaultdict(lambda: {"total": 0, "success": 0, "error": 0})
    for r in results:
        sc = r.get("scenario", "unknown")
        scenario_stats[sc]["total"] += 1
        if r["status"] == "success":
            scenario_stats[sc]["success"] += 1
        else:
            scenario_stats[sc]["error"] += 1

    metrics = {
        "total_cases": total,
        "success_cases": len(success_results),
        "error_cases": len(error_results),
        "success_rate": round(len(success_results) / total, 3),
        "intent_match_rate": intent_match_rate,
        "avg_tool_match_rate": avg_tool_match,
        "full_tool_match_rate": full_tool_match_rate,
        "tool_execution_success_rate": avg_tool_exec_success,
        "tool_correct_rate": avg_tool_correct,
        "reply_completion_rate": reply_rate,
        "avg_ttfb_sec": round(sum(ttfs) / len(ttfs), 2) if ttfs else 0,
        "avg_total_latency_sec": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "median_total_latency_sec": round(sorted(latencies)[len(latencies)//2], 2) if latencies else 0,
        "max_total_latency_sec": round(max(latencies), 2) if latencies else 0,
        "min_total_latency_sec": round(min(ttfs), 2) if ttfs else 0,
        "intent_breakdown": {k: dict(v) for k, v in intent_stats.items()},
        "scenario_breakdown": {k: dict(v) for k, v in scenario_stats.items()},
        "errors": [{"case_id": r["case_id"], "error": r["error"]} for r in error_results],
        "timestamp": datetime.now().isoformat(),
    }
    return metrics


async def run_single_round(cases: List[dict], run_idx: int) -> dict:
    """运行一轮评测"""
    run_dir = RESULTS_DIR / f"run{run_idx}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"【第 {run_idx} 轮评测开始】 用例数: {len(cases)}")
    print(f"{'='*70}")

    results = []
    async with aiohttp.ClientSession() as session:
        for idx, case in enumerate(cases, 1):
            case_id = case["session_id"]
            print(f"\n[{idx}/{len(cases)}] 测试: {case_id} | {case['message'][:50]}...")

            result = await run_single_case(case, run_dir, session)
            results.append(result)

            status_icon = "✅" if result["status"] == "success" else "❌"
            intent_ok = "✅" if result["intent_match"] else "❌"
            tool_rate = result["tool_match_rate"]
            reply_ok = "✅" if result["has_reply"] else "❌"
            latency = result.get("total_latency", 0)

            print(f"  {status_icon} status={result['status']} | intent={intent_ok} | tool_match={tool_rate} | reply={reply_ok} | latency={latency}s")
            if result["error"]:
                print(f"  ⚠️  error: {result['error'][:100]}")

            # 每 5 条保存一次中间进度
            if idx % 5 == 0:
                progress = {
                    "run": run_idx,
                    "completed": idx,
                    "total": len(cases),
                    "timestamp": datetime.now().isoformat(),
                }
                with open(run_dir / "_progress.json", "w", encoding="utf-8") as f:
                    json.dump(progress, f, ensure_ascii=False, indent=2)

    # 计算指标
    metrics = compute_metrics(results)

    # 保存完整结果
    with open(run_dir / "_all_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 保存报告
    with open(run_dir / "_report.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 打印报告
    print(f"\n{'='*70}")
    print(f"【第 {run_idx} 轮评测完成】")
    print(f"{'='*70}")
    print(f"  总用例: {metrics['total_cases']}")
    print(f"  成功: {metrics['success_cases']} ({metrics['success_rate']})")
    print(f"  失败: {metrics['error_cases']}")
    print(f"  意图匹配率: {metrics['intent_match_rate']}")
    print(f"  工具平均匹配率: {metrics['avg_tool_match_rate']}")
    print(f"  工具完全匹配率: {metrics['full_tool_match_rate']}")
    print(f"  回复完成率: {metrics['reply_completion_rate']}")
    print(f"  平均 TTFB: {metrics['avg_ttfb_sec']}s")
    print(f"  平均总延迟: {metrics['avg_total_latency_sec']}s")
    print(f"  中位总延迟: {metrics['median_total_latency_sec']}s")
    print(f"  最大延迟: {metrics['max_total_latency_sec']}s")
    if metrics['errors']:
        print(f"  错误列表:")
        for e in metrics['errors'][:5]:
            print(f"    - {e['case_id']}: {e['error'][:80]}")
    print(f"  结果保存: {run_dir}")

    return metrics


async def main(runs: int = 3, start_run: int = 1):
    """主入口：读取数据集，跑多轮评测"""
    # 读取数据集
    if not DATASET_PATH.exists():
        print(f"❌ 数据集不存在: {DATASET_PATH}")
        sys.exit(1)

    cases = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    print(f"✅ 加载测试集: {len(cases)} 条用例")
    print(f"✅ 结果保存目录: {RESULTS_DIR}")
    print(f"✅ 计划运行轮数: {runs}")

    # 先检查 backend 健康
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{BASE_URL}/docs") as r:
                if r.status == 200:
                    print(f"✅ Backend 服务正常: {BASE_URL}")
                else:
                    print(f"⚠️ Backend 返回状态: {r.status}")
    except Exception as e:
        print(f"⚠️ Backend 检查失败: {e}")

    all_round_metrics = []
    for run_idx in range(start_run, start_run + runs):
        metrics = await run_single_round(cases, run_idx)
        all_round_metrics.append(metrics)

    # 三轮汇总
    print(f"\n{'='*70}")
    print(f"【三轮稳定性汇总】")
    print(f"{'='*70}")

    for key in ["success_rate", "intent_match_rate", "avg_tool_match_rate", "full_tool_match_rate", "reply_completion_rate", "avg_total_latency_sec"]:
        values = [m[key] for m in all_round_metrics]
        avg = round(sum(values) / len(values), 3)
        std = round((sum((v - avg) ** 2 for v in values) / len(values)) ** 0.5, 4) if len(values) > 1 else 0
        print(f"  {key}: round1={values[0]} round2={values[1]} round3={values[2]} | avg={avg} std={std}")

    # 保存汇总
    summary = {
        "total_runs": runs,
        "total_cases": len(cases),
        "round_metrics": all_round_metrics,
        "timestamp": datetime.now().isoformat(),
    }
    with open(RESULTS_DIR / "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 全部评测完成！汇总保存: {RESULTS_DIR / '_summary.json'}")


if __name__ == "__main__":
    # 强制行缓冲，确保后台运行日志实时写入
    import sys, os
    if hasattr(sys.stdout, 'fileno'):
        sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
        sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3, help="运行轮数")
    parser.add_argument("--start-run", type=int, default=1, help="起始轮次编号")
    parser.add_argument("--case", type=str, default=None, help="只跑单条 case_id（调试用）")
    args = parser.parse_args()

    if args.case:
        # 单条调试模式
        cases = []
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            for line in f:
                c = json.loads(line.strip())
                if c["session_id"] == args.case:
                    cases.append(c)
        asyncio.run(run_single_round(cases, 0))
    else:
        asyncio.run(main(runs=args.runs, start_run=args.start_run))
