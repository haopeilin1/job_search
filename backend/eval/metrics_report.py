"""
评估指标报告 —— 从 eval_results + telemetry 计算核心指标。

用法：
    cd backend && python eval/metrics_report.py eval/eval_results_xxxx.jsonl
    cd backend && python eval/metrics_report.py                    # 自动找最新结果
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.core.config import settings

EVAL_DIR = Path(__file__).parent
TELEMETRY_FILE = Path(__file__).parent.parent / "logs" / "events.jsonl"


def load_results(path: Path) -> List[dict]:
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def load_telemetry(since_timestamp: float = 0) -> List[dict]:
    """加载指定时间之后的埋点事件。"""
    events = []
    if not TELEMETRY_FILE.exists():
        return events
    with open(TELEMETRY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
                # 事件格式可能是直接JSON或logger前缀格式
                # telemetry logger配置为纯JSON格式
                if isinstance(evt, dict) and "event" in evt:
                    if evt.get("timestamp", 0) >= since_timestamp:
                        events.append(evt)
            except Exception:
                continue
    return events


def compute_api_metrics(results: List[dict]) -> dict:
    total = len(results)
    success = sum(1 for r in results if r.get("status_code") == 200)
    errors = sum(1 for r in results if r.get("error"))
    latencies = [r["latency_ms"] for r in results if r.get("latency_ms")]
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    p99_lat = sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0

    # 按批次统计
    batch_stats = defaultdict(lambda: {"total": 0, "success": 0})
    for r in results:
        sid = r.get("session_id", "")
        batch = sid.split("_")[1] if "_" in sid else "unknown"
        batch_stats[batch]["total"] += 1
        if r.get("status_code") == 200:
            batch_stats[batch]["success"] += 1

    return {
        "total": total,
        "success": success,
        "errors": errors,
        "success_rate": round(success / total, 4) if total else 0,
        "avg_latency_ms": round(avg_lat, 2),
        "p99_latency_ms": round(p99_lat, 2),
        "batch_success": {b: round(v["success"] / v["total"], 4) for b, v in batch_stats.items()},
    }


def compute_intent_accuracy(results: List[dict], events: List[dict]) -> dict:
    """意图识别准确率：对比 gold_intents 与 predicted_intents。"""
    # 建立 session_id -> predicted_intents 映射
    session_intents = {}
    for evt in events:
        if evt.get("event") == "intent_classified":
            sid = evt.get("session_id") or evt.get("payload", {}).get("session_id")
            if sid:
                session_intents[sid] = evt.get("payload", {}).get("predicted_intents", [])

    total = 0
    exact_match = 0
    partial_match = 0
    missing = 0

    for r in results:
        case = r.get("case", {})
        eval_ctx = case.get("eval_context", {})
        gold = set(eval_ctx.get("gold_intents", []))
        if not gold:
            continue

        sid = case.get("session_id", "")
        pred = set(session_intents.get(sid, []))

        total += 1
        if gold == pred:
            exact_match += 1
        if gold & pred:
            partial_match += 1
        if not pred:
            missing += 1

    return {
        "total_cases": total,
        "exact_match": exact_match,
        "exact_accuracy": round(exact_match / total, 4) if total else 0,
        "partial_match": partial_match,
        "partial_accuracy": round(partial_match / total, 4) if total else 0,
        "missing_predictions": missing,
    }


def compute_tool_accuracy(results: List[dict], events: List[dict]) -> dict:
    """工具执行准确率：对比 expected_tools 与实际执行的工具链。"""
    # 按 turn_id/session_id 聚合 tool_executed 事件
    session_tools = defaultdict(list)
    for evt in events:
        if evt.get("event") == "tool_executed":
            sid = evt.get("session_id") or evt.get("payload", {}).get("session_id")
            tid = evt.get("turn_id") or evt.get("payload", {}).get("turn_id")
            tool_name = evt.get("payload", {}).get("tool_name")
            success = evt.get("payload", {}).get("success")
            if sid and tool_name:
                key = f"{sid}_{tid}"
                session_tools[key].append({"name": tool_name, "success": success})

    total = 0
    tool_match = 0
    all_tools_success = 0

    for r in results:
        case = r.get("case", {})
        eval_ctx = case.get("eval_context", {})
        expected = set(eval_ctx.get("expected_tools", []))
        if not expected:
            continue

        sid = case.get("session_id", "")
        # 简化：取该session所有tool（不严格按turn区分）
        actual = set()
        for key, tools in session_tools.items():
            if key.startswith(sid):
                for t in tools:
                    actual.add(t["name"])

        total += 1
        if expected == actual:
            tool_match += 1
        if all(t.get("success") for key, tools in session_tools.items() if key.startswith(sid) for t in tools):
            all_tools_success += 1

    return {
        "total_cases": total,
        "tool_chain_exact_match": tool_match,
        "tool_chain_accuracy": round(tool_match / total, 4) if total else 0,
        "all_tools_success": all_tools_success,
    }


def compute_core_metrics(events: List[dict]) -> dict:
    """4大核心指标：再来率、纠错率、单次成本、异常恢复率。"""
    # 按session统计轮次
    session_turns = defaultdict(int)
    for evt in events:
        sid = evt.get("session_id") or evt.get("payload", {}).get("session_id")
        if sid and evt.get("event") == "turn_completed":
            session_turns[sid] += 1

    # 再来率 = 多轮session中后续轮次与前面意图重复的比例
    # 简化：统计平均每个session的轮次数，>1.5视为有再来
    multi_turn_sessions = sum(1 for c in session_turns.values() if c > 1)
    avg_turns_per_session = sum(session_turns.values()) / len(session_turns) if session_turns else 0

    # 异常与恢复
    exceptions = [e for e in events if e.get("event") == "exception_occurred"]
    recovered = sum(1 for e in exceptions if e.get("payload", {}).get("recovered"))

    # 单次成本
    turns = [e for e in events if e.get("event") == "turn_completed"]
    costs = [e.get("payload", {}).get("cost_usd", 0) for e in turns]
    avg_cost = sum(costs) / len(costs) if costs else 0
    total_cost = sum(costs)

    # 双层召回粗筛信息
    coarse_meta = []
    for e in events:
        if e.get("event") == "tool_executed" and e.get("payload", {}).get("tool_name") == "global_rank":
            meta = e.get("payload", {}).get("extra", {}).get("_coarse_filter_meta")
            if meta:
                coarse_meta.append(meta)

    avg_filter_ratio = sum(m.get("filter_ratio", 0) for m in coarse_meta) / len(coarse_meta) if coarse_meta else 0
    avg_input_jds = sum(m.get("input_jds", 0) for m in coarse_meta) / len(coarse_meta) if coarse_meta else 0
    avg_output_jds = sum(m.get("output_jds", 0) for m in coarse_meta) / len(coarse_meta) if coarse_meta else 0

    return {
        "session_count": len(session_turns),
        "multi_turn_sessions": multi_turn_sessions,
        "avg_turns_per_session": round(avg_turns_per_session, 2),
        "retry_rate_approx": round(multi_turn_sessions / len(session_turns), 4) if session_turns else 0,
        "exception_count": len(exceptions),
        "recovered_count": recovered,
        "recovery_rate": round(recovered / len(exceptions), 4) if exceptions else 1.0,
        "turn_count": len(turns),
        "avg_cost_usd": round(avg_cost, 6),
        "total_cost_usd": round(total_cost, 6),
        "coarse_filter_runs": len(coarse_meta),
        "coarse_avg_filter_ratio": round(avg_filter_ratio, 4),
        "coarse_avg_input_jds": round(avg_input_jds, 2),
        "coarse_avg_output_jds": round(avg_output_jds, 2),
    }


def generate_report(results: List[dict], events: List[dict]) -> dict:
    api = compute_api_metrics(results)
    intent = compute_intent_accuracy(results, events)
    tool = compute_tool_accuracy(results, events)
    core = compute_core_metrics(events)

    return {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "api": api,
        "intent": intent,
        "tool": tool,
        "core": core,
    }


def print_report(report: dict):
    print("=" * 60)
    print("求职雷达 Agent 评估报告")
    print("=" * 60)

    api = report["api"]
    print(f"\n📊 API 层指标")
    print(f"  总用例: {api['total']} | 成功: {api['success']} | 错误: {api['errors']}")
    print(f"  成功率: {api['success_rate']*100:.1f}%")
    print(f"  平均延迟: {api['avg_latency_ms']:.0f}ms | P99: {api['p99_latency_ms']:.0f}ms")
    print(f"  各批次成功率:")
    for b, rate in api["batch_success"].items():
        print(f"    批次 {b}: {rate*100:.1f}%")

    intent = report["intent"]
    print(f"\n🎯 意图识别准确率")
    print(f"  测试用例: {intent['total_cases']}")
    print(f"  精确匹配: {intent['exact_match']} ({intent['exact_accuracy']*100:.1f}%)")
    print(f"  部分匹配: {intent['partial_match']} ({intent['partial_accuracy']*100:.1f}%)")
    print(f"  预测缺失: {intent['missing_predictions']}")

    tool = report["tool"]
    print(f"\n🔧 工具执行准确率")
    print(f"  测试用例: {tool['total_cases']}")
    print(f"  工具链精确匹配: {tool['tool_chain_exact_match']} ({tool['tool_chain_accuracy']*100:.1f}%)")

    core = report["core"]
    print(f"\n💰 核心指标")
    print(f"  Session数: {core['session_count']} | 多轮Session: {core['multi_turn_sessions']}")
    print(f"  平均轮次/Session: {core['avg_turns_per_session']}")
    print(f"  异常次数: {core['exception_count']} | 恢复次数: {core['recovered_count']}")
    print(f"  异常恢复率: {core['recovery_rate']*100:.1f}%")
    print(f"  总回合数: {core['turn_count']}")
    print(f"  平均单次成本: ${core['avg_cost_usd']:.6f}")
    print(f"  总成本: ${core['total_cost_usd']:.6f}")
    print(f"\n🔍 双层召回粗筛")
    print(f"  粗筛执行次数: {core['coarse_filter_runs']}")
    print(f"  平均输入JD数: {core['coarse_avg_input_jds']}")
    print(f"  平均输出JD数: {core['coarse_avg_output_jds']}")
    print(f"  平均过滤比例: {core['coarse_avg_filter_ratio']*100:.1f}%")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="评估指标报告")
    parser.add_argument("results_file", type=str, nargs="?", default=None, help="eval_results JSONL 文件路径")
    args = parser.parse_args()

    # 自动查找最新结果文件
    if args.results_file:
        results_path = Path(args.results_file)
    else:
        candidates = sorted(EVAL_DIR.glob("eval_results_*.jsonl"))
        if not candidates:
            print("[ERROR] 未找到 eval_results 文件，请指定路径")
            return
        results_path = candidates[-1]

    if not results_path.exists():
        print(f"[ERROR] 文件不存在: {results_path}")
        return

    print(f"[INFO] 加载结果: {results_path}")
    results = load_results(results_path)

    # 估算时间范围：只加载结果文件创建时间之后的telemetry
    since = results_path.stat().st_mtime
    print(f"[INFO] 加载埋点事件 (since {since})...")
    events = load_telemetry(since_timestamp=since)
    print(f"[INFO] 加载到 {len(events)} 条埋点事件")

    report = generate_report(results, events)
    print_report(report)

    # 保存报告
    report_path = results_path.with_suffix(".report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 报告已保存: {report_path}")


if __name__ == "__main__":
    main()
