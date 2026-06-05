#!/usr/bin/env python3
"""
批量全链路评测脚本 v2 — 集成 Judge 评估

修复内容（v2）：
1. 从 SSE done 事件的 debug_info 提取完整 kb_chunks，供 Judge 评判 faithfulness
2. 保存 eval_context 和 resume_id，确保 Judge 能获取完整上下文
3. 每轮评测结束后自动调用 v3.5 Judge 评估
4. Judge 结果纳入 _summary.json 和 _report.json

要求：
1. 每条保存完整过程及中间结果（含 kb_chunks / tool_executions_full），保证可追溯
2. 每次跑完计算全部评测指标 + Judge 12 维评分
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

from eval.judge_postprocess import judge_single_case, _build_case_context

BASE_URL = "http://127.0.0.1:8001"
STREAM_URL = f"{BASE_URL}/api/v1/chat/stream"
DATASET_PATH = Path(__file__).resolve().parent / "test_dataset.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# 特殊 case 定义
CLARIFICATION_CASES = {"eval_chen_13", "eval_li_14", "eval_gen_03", "eval_gen_06"}
BOUNDARY_CASES = {"eval_gen_02", "eval_gen_03"}
SPECIAL_CASES = CLARIFICATION_CASES | BOUNDARY_CASES

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
    "attribute_verify": "verify",
    "clarification": "clarification",
    "chat": "chat",
}


def _normalize_intent(intent: str) -> str:
    return INTENT_ALIASES.get(intent, intent)


def _normalize_intents(intents: List[str]) -> List[str]:
    return sorted(set(_normalize_intent(i) for i in intents))


# 全局 session_group 映射（跨 case 复用 session）
_session_group_map: Dict[str, str] = {}


def _extract_kb_chunks_from_debug(debug_info: dict) -> List[dict]:
    """从 debug_info 中提取完整的 kb_chunks（不截断）"""
    if not debug_info:
        return []
    # 优先使用 evidence_cache（完整 chunks）
    evidence_cache = debug_info.get("evidence_cache", [])
    if evidence_cache:
        return evidence_cache
    # 备选：从 task_graph 的 tasks 中提取
    chunks = []
    task_graph = debug_info.get("task_graph", {})
    for tid, task in task_graph.get("tasks", {}).items():
        if task.get("tool_name") in ("kb_retrieve", "external_search") and task.get("status") == "success":
            result = task.get("result", {})
            if isinstance(result, dict):
                chunks.extend(result.get("chunks", []))
    return chunks


def _build_tool_executions_full(debug_info: dict) -> List[dict]:
    """从 debug_info 构造 tool_executions_full（供 Judge 使用）"""
    if not debug_info:
        return []
    tool_outputs = debug_info.get("tool_outputs", [])
    executions = []
    for o in tool_outputs:
        executions.append({
            "tool_name": o.get("tool"),
            "task_id": o.get("task_id"),
            "status": "success",
            "output": o.get("result"),
        })
    # 补充 task_graph 中可能的失败任务
    task_graph = debug_info.get("task_graph", {})
    seen_task_ids = {e["task_id"] for e in executions}
    for tid, task in task_graph.get("tasks", {}).items():
        if tid not in seen_task_ids and task.get("tool_name"):
            executions.append({
                "tool_name": task.get("tool_name"),
                "task_id": tid,
                "status": "success" if task.get("status") == "success" else "failed",
                "output": task.get("result"),
            })
    return executions


async def run_single_case(case: dict, run_dir: Path, session: aiohttp.ClientSession) -> dict:
    """
    对单条 case 发送 SSE 流式请求，保存完整过程（含 kb_chunks / debug_info），返回评测结果。
    支持 session_group 复用 session（多轮对话）。
    """
    global _session_group_map
    case_id = case["session_id"]
    message = case["message"]
    resume_id = case.get("resume_id", "")
    gold_ctx = case.get("eval_context", {})
    gold_intents = _normalize_intents(gold_ctx.get("gold_intents", []))
    expected_tools = gold_ctx.get("expected_tools", [])
    session_group = case.get("session_group")

    # 确定 session_id 和是否重置
    if session_group:
        if session_group not in _session_group_map:
            _session_group_map[session_group] = f"batch_{session_group}_{int(time.time()*1000)}"
            reset_session = True
            print(f"  -> 新 session_group: {session_group}, sid={_session_group_map[session_group]}")
        else:
            reset_session = False
            print(f"  -> 复用 session_group: {session_group}, sid={_session_group_map[session_group]}")
        sid = _session_group_map[session_group]
    else:
        sid = f"{case_id}_{int(time.time()*1000)}"
        reset_session = True

    result = {
        "case_id": case_id,
        "message": message,
        "gold_intents": gold_intents,
        "expected_tools": expected_tools,
        "scenario": gold_ctx.get("scenario", ""),
        "eval_context": gold_ctx,
        "resume_id": resume_id,
        "status": "pending",
        "error": None,
        "ttfb": None,
        "total_latency": None,
        "event_count": 0,
        "pred_intent": None,
        "pred_intents": [],
        "pred_tools": [],
        "tools_called": [],
        "kb_chunks": [],
        "tool_executions_full": [],
        "debug_info": {},
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
        "session_id": sid,
        "stream": True,
        "eval_context": {
            "resume_id": resume_id,
            "reset_session": reset_session,
        },
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

        # 真实 LLM token 消耗（从后端通过 SSE done 事件上报）
        usage = done_event.get("usage", {})
        result["prompt_tokens"] = usage.get("prompt_tokens", 0)
        result["completion_tokens"] = usage.get("completion_tokens", 0)
        result["total_tokens"] = usage.get("total_tokens", 0)

        # 预测意图
        pred_intent = done_event.get("intent")
        result["pred_intent"] = pred_intent
        result["pred_intents"] = [pred_intent] if pred_intent else []
        result["intent_match"] = _normalize_intent(pred_intent) in gold_intents if pred_intent else False

        # 工具调用
        agent_tools = done_event.get("agent", {}).get("tools", []) if isinstance(done_event.get("agent"), dict) else []
        tools_called = [t.get("tool") for t in agent_tools if t.get("tool")]
        result["tools_called"] = tools_called
        result["pred_tools"] = agent_tools

        # 提取完整 debug_info（含 kb_chunks）
        debug_info = done_event.get("debug_info", {})
        result["debug_info"] = debug_info
        result["kb_chunks"] = _extract_kb_chunks_from_debug(debug_info)
        result["tool_executions_full"] = _build_tool_executions_full(debug_info)

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

        # 回复（兼容 dict 的 "text" 和 "content" 两种格式）
        raw_reply = done_event.get("reply", "")
        if isinstance(raw_reply, dict):
            reply = raw_reply.get("text") or raw_reply.get("content", "")
        else:
            reply = str(raw_reply)
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


async def run_judge_for_case(case_result: dict) -> dict:
    """对单条 case 结果调用 v3.5 Judge 评估"""
    case_id = case_result["case_id"]
    
    # 构造 Judge 需要的 case_result 格式
    case_for_judge = {
        "case_id": case_id,
        "message": case_result["message"],
        "gold_intents": case_result.get("gold_intents", []),
        "reply": case_result.get("reply", ""),
        "tools_called": [
            {"tool": t.get("tool", ""), "status": t.get("status", "")}
            for t in case_result.get("pred_tools", [])
        ],
        "pred_intents": case_result.get("pred_intents", []),
        "kb_chunks": case_result.get("kb_chunks", []),
        "tool_executions_full": case_result.get("tool_executions_full", []),
        "eval_context": case_result.get("eval_context", {}),
        "resume_id": case_result.get("resume_id", ""),
    }
    
    ctx = _build_case_context(case_id, case_for_judge)
    judge_result = await judge_single_case(case_for_judge, ctx)
    return judge_result


async def run_judge_for_run(results: List[dict], run_dir: Path) -> dict:
    """对一轮所有 case 调用 Judge，返回汇总报告"""
    print(f"\n{'='*70}")
    print(f"【Judge 评估开始】case 数: {len(results)}")
    print(f"{'='*70}")
    
    judge_results = []
    for idx, r in enumerate(results, 1):
        case_id = r["case_id"]
        if r.get("status") != "success":
            # 失败的 case Judge 直接判不通过
            judge_results.append({
                "case_id": case_id,
                "resolved": False,
                "scores": {k: 0 for k in [
                    "intent_accuracy", "slot_accuracy", "tool_correctness", "tool_execution",
                    "response_accuracy", "response_completeness", "citation_quality",
                    "coherence", "tone", "efficiency", "faithfulness", "answer_relevance"
                ]},
                "reason": f"Case 执行失败: {r.get('error', 'unknown')}",
                "rule_hit": None,
                "veto": False,
                "needs_rag": "kb_retrieve" in r.get("expected_tools", []),
            })
            print(f"  [{idx}/{len(results)}] {case_id} -> ❌ 执行失败，跳过 Judge")
            continue
        
        try:
            jr = await run_judge_for_case(r)
            judge_results.append(jr)
            scores = jr.get("scores", {})
            print(f"  [{idx}/{len(results)}] {case_id} -> resolved={jr['resolved']} "
                  f"intent={scores.get('intent_accuracy',0)}/5 "
                  f"faithfulness={scores.get('faithfulness',0)}/5 "
                  f"relevance={scores.get('answer_relevance',0)}/5")
        except Exception as e:
            print(f"  [{idx}/{len(results)}] {case_id} -> Judge 调用失败: {e}")
            judge_results.append({
                "case_id": case_id,
                "resolved": False,
                "scores": {k: 0 for k in [
                    "intent_accuracy", "slot_accuracy", "tool_correctness", "tool_execution",
                    "response_accuracy", "response_completeness", "citation_quality",
                    "coherence", "tone", "efficiency", "faithfulness", "answer_relevance"
                ]},
                "reason": f"Judge 调用失败: {e}",
                "rule_hit": None,
                "veto": False,
                "needs_rag": "kb_retrieve" in r.get("expected_tools", []),
            })
    
    # 汇总统计
    total = len(judge_results)
    resolved_count = sum(1 for j in judge_results if j["resolved"])
    veto_count = sum(1 for j in judge_results if j.get("veto"))
    rag_cases = [j for j in judge_results if j.get("needs_rag")]
    non_rag_cases = [j for j in judge_results if not j.get("needs_rag")]
    
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
        "timestamp": datetime.now().isoformat(),
    }
    
    # 保存 Judge 报告
    judge_path = run_dir / "_report_judge.json"
    with open(judge_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*70}")
    print(f"【Judge 评估完成】")
    print(f"{'='*70}")
    print(f"  总 case: {total}")
    print(f"  Judge 通过: {resolved_count}/{total} = {summary['resolved_rate']:.2%}")
    print(f"  否决项: {veto_count}/{total}")
    print(f"  RAG case: {len(rag_cases)} | 非 RAG: {len(non_rag_cases)}")
    print(f"  faithfulness(RAG): {rag_dim_avg.get('faithfulness', 'N/A')}/5")
    print(f"  answer_relevance(RAG): {rag_dim_avg.get('answer_relevance', 'N/A')}/5")
    print(f"  报告保存: {judge_path}")
    
    return summary


def compute_metrics(results: List[dict], exclude_special: bool = False) -> dict:
    """计算本轮全部评测指标
    
    Args:
        exclude_special: 是否排除 clarification 和边界 case
    """
    if exclude_special:
        filtered = [r for r in results if r["case_id"] not in SPECIAL_CASES]
    else:
        filtered = results
    
    total = len(filtered)
    success_results = [r for r in filtered if r["status"] == "success"]
    error_results = [r for r in filtered if r["status"] == "error"]

    # 意图匹配（仅对非 clarification case 统计）
    intent_eval_results = [r for r in success_results if r["case_id"] not in CLARIFICATION_CASES]
    intent_match_count = sum(1 for r in intent_eval_results if r["intent_match"])
    intent_match_rate = round(intent_match_count / len(intent_eval_results), 3) if intent_eval_results else 0

    # 工具匹配
    tool_match_rates = [r["tool_match_rate"] for r in success_results]
    avg_tool_match = round(sum(tool_match_rates) / len(tool_match_rates), 3) if tool_match_rates else 0
    full_tool_match = sum(1 for r in success_results if r["tool_match_rate"] >= 1.0)
    full_tool_match_rate = round(full_tool_match / len(success_results), 3) if success_results else 0

    # 工具执行成功率
    tool_exec_rates = [r.get("tool_execution_success_rate", 0) for r in filtered]
    avg_tool_exec_success = round(sum(tool_exec_rates) / len(tool_exec_rates), 3) if tool_exec_rates else 0

    # 工具调用正确率
    tool_correct_rates = [r.get("tool_correct_rate", 0) for r in filtered]
    avg_tool_correct = round(sum(tool_correct_rates) / len(tool_correct_rates), 3) if tool_correct_rates else 0

    # 回复完成
    has_reply_count = sum(1 for r in success_results if r["has_reply"])
    reply_rate = round(has_reply_count / len(success_results), 3) if success_results else 0

    # 延迟
    ttfs = [r["ttfb"] for r in success_results if r["ttfb"] is not None]
    latencies = [r["total_latency"] for r in success_results if r["total_latency"] is not None]

    # 按意图统计（排除 clarification）
    intent_stats = defaultdict(lambda: {"total": 0, "match": 0, "success": 0})
    for r in success_results:
        if r["case_id"] in CLARIFICATION_CASES:
            continue
        for intent in r.get("gold_intents", []):
            intent_stats[intent]["total"] += 1
            intent_stats[intent]["success"] += 1
            if r["intent_match"]:
                intent_stats[intent]["match"] += 1

    # 按场景统计
    scenario_stats = defaultdict(lambda: {"total": 0, "success": 0, "error": 0})
    for r in filtered:
        sc = r.get("scenario", "unknown")
        scenario_stats[sc]["total"] += 1
        if r["status"] == "success":
            scenario_stats[sc]["success"] += 1
        else:
            scenario_stats[sc]["error"] += 1

    # 真实 token 消耗汇总
    prompt_tokens_list = [r.get("prompt_tokens", 0) for r in success_results]
    completion_tokens_list = [r.get("completion_tokens", 0) for r in success_results]
    total_tokens_list = [r.get("total_tokens", 0) for r in success_results]

    metrics = {
        "total_cases": total,
        "success_cases": len(success_results),
        "error_cases": len(error_results),
        "success_rate": round(len(success_results) / total, 3) if total else 0,
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
        "total_prompt_tokens": sum(prompt_tokens_list),
        "total_completion_tokens": sum(completion_tokens_list),
        "total_tokens": sum(total_tokens_list),
        "avg_prompt_tokens": round(sum(prompt_tokens_list) / len(prompt_tokens_list), 1) if prompt_tokens_list else 0,
        "avg_completion_tokens": round(sum(completion_tokens_list) / len(completion_tokens_list), 1) if completion_tokens_list else 0,
        "avg_total_tokens": round(sum(total_tokens_list) / len(total_tokens_list), 1) if total_tokens_list else 0,
        "intent_breakdown": {k: dict(v) for k, v in intent_stats.items()},
        "scenario_breakdown": {k: dict(v) for k, v in scenario_stats.items()},
        "errors": [{"case_id": r["case_id"], "error": r["error"]} for r in error_results],
        "clarification_cases": list(CLARIFICATION_CASES),
        "boundary_cases": list(BOUNDARY_CASES),
        "excluded_cases": list(SPECIAL_CASES) if exclude_special else [],
        "timestamp": datetime.now().isoformat(),
    }
    return metrics


async def run_single_round(cases: List[dict], run_idx: int) -> dict:
    """运行一轮评测（支持断点重续）"""
    run_dir = RESULTS_DIR / f"run{run_idx}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 断点重续：检查已完成的 case（成功的才跳过，失败的需重试）
    completed_ids = set()
    failed_ids = set()
    for fpath in run_dir.glob("eval_*.json"):
        case_id = fpath.stem
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("status") == "success":
                completed_ids.add(case_id)
            else:
                failed_ids.add(case_id)
        except Exception:
            pass
    
    # 加载已有成功结果
    results = []
    for case in cases:
        case_id = case["session_id"]
        case_file = run_dir / f"{case_id}.json"
        if case_file.exists() and case_id in completed_ids:
            try:
                with open(case_file, "r", encoding="utf-8") as f:
                    results.append(json.load(f))
            except Exception:
                pass

    pending_cases = [c for c in cases if c["session_id"] not in completed_ids]
    
    print(f"\n{'='*70}")
    print(f"【第 {run_idx} 轮评测开始】 用例数: {len(cases)} | 已完成: {len(results)} | 待执行: {len(pending_cases)}")
    print(f"{'='*70}")

    # 增加超时：单条给足 10 分钟，防止不必要的 API 重试
    timeout = aiohttp.ClientTimeout(total=600, connect=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for idx, case in enumerate(pending_cases, 1):
            case_id = case["session_id"]
            global_idx = len(results) + idx
            print(f"\n[{global_idx}/{len(cases)}] 测试: {case_id} | {case['message'][:50]}...")

            result = await run_single_case(case, run_dir, session)
            results.append(result)

            status_icon = "✅" if result["status"] == "success" else "❌"
            intent_ok = "✅" if result["intent_match"] else "❌"
            tool_rate = result["tool_match_rate"]
            reply_ok = "✅" if result["has_reply"] else "❌"
            latency = result.get("total_latency", 0)
            kb_count = len(result.get("kb_chunks", []))

            print(f"  {status_icon} status={result['status']} | intent={intent_ok} | tool_match={tool_rate} | reply={reply_ok} | latency={latency}s | kb_chunks={kb_count}")
            if result["error"]:
                print(f"  ⚠️  error: {result['error'][:100]}")

            # 每 1 条保存一次中间进度（立即保存，防超时丢失）
            progress = {
                "run": run_idx,
                "completed": global_idx,
                "total": len(cases),
                "timestamp": datetime.now().isoformat(),
            }
            with open(run_dir / "_progress.json", "w", encoding="utf-8") as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)

    # 计算指标（全量）
    metrics_all = compute_metrics(results, exclude_special=False)
    # 计算指标（排除特殊 case）
    metrics = compute_metrics(results, exclude_special=True)

    # 保存完整结果
    with open(run_dir / "_all_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 保存报告（排除特殊 case 版本）
    with open(run_dir / "_report.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 保存全量报告
    with open(run_dir / "_report_all.json", "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, ensure_ascii=False, indent=2)

    # 保存澄清 case 专项报告
    clarification_results = [r for r in results if r["case_id"] in CLARIFICATION_CASES]
    with open(run_dir / "_clarification_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "total": len(clarification_results),
            "cases": [
                {
                    "case_id": r["case_id"],
                    "message": r["message"],
                    "pred_intent": r.get("pred_intent"),
                    "clarification_triggered": r.get("pred_intent") == "clarification",
                    "status": r["status"],
                }
                for r in clarification_results
            ],
            "timestamp": datetime.now().isoformat(),
        }, f, ensure_ascii=False, indent=2)

    # 保存边界 case 专项报告
    boundary_results = [r for r in results if r["case_id"] in BOUNDARY_CASES]
    with open(run_dir / "_boundary_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "total": len(boundary_results),
            "cases": [
                {
                    "case_id": r["case_id"],
                    "message": r["message"],
                    "pred_intent": r.get("pred_intent"),
                    "status": r["status"],
                    "reply_preview": r.get("reply", "")[:200],
                }
                for r in boundary_results
            ],
            "timestamp": datetime.now().isoformat(),
        }, f, ensure_ascii=False, indent=2)

    # ════════════════════════════════════════
    # 自动调用 Judge 评估
    # ════════════════════════════════════════
    judge_summary = await run_judge_for_run(results, run_dir)
    metrics["judge_summary"] = judge_summary
    metrics_all["judge_summary"] = judge_summary
    
    # 更新报告文件（包含 Judge 结果）
    with open(run_dir / "_report.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(run_dir / "_report_all.json", "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, ensure_ascii=False, indent=2)

    # 打印报告
    print(f"\n{'='*70}")
    print(f"【第 {run_idx} 轮评测完成】")
    print(f"{'='*70}")
    print(f"  总用例: {metrics_all['total_cases']} (排除特殊后: {metrics['total_cases']})")
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
    print(f"  真实 Token 消耗(平均): prompt={metrics['avg_prompt_tokens']} | completion={metrics['avg_completion_tokens']} | total={metrics['avg_total_tokens']}")
    print(f"  真实 Token 消耗(总计): prompt={metrics['total_prompt_tokens']} | completion={metrics['total_completion_tokens']} | total={metrics['total_tokens']}")
    print(f"  Judge 通过率: {judge_summary['resolved_rate']:.2%}")
    print(f"  Judge faithfulness(RAG): {judge_summary.get('rag_dimension_averages', {}).get('faithfulness', 'N/A')}/5")
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
        round_strs = " ".join([f"round{i+1}={v}" for i, v in enumerate(values)])
        print(f"  {key}: {round_strs} | avg={avg} std={std}")
    
    # Judge 汇总
    print(f"\n  Judge 通过率汇总:")
    for idx, m in enumerate(all_round_metrics, 1):
        js = m.get("judge_summary", {})
        print(f"    Round {idx}: {js.get('resolved_rate', 0):.2%} "
              f"(faithfulness={js.get('rag_dimension_averages', {}).get('faithfulness', 'N/A')}/5)")

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
