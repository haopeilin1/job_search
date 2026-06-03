#!/usr/bin/env python3
"""
批量 HTTP 端到端全链路测试 v3.5

特性：
1. 59条 case 全量 HTTP 测试（不走组件内部调用）
2. 多轮对话状态维护（session_group 复用 session）
3. 断点重续（progress.json 记录进度）
4. 每条测试完立即保存（防止超时丢失）
5. v3.5 Judge 评估（12维 0-5分 + resolved）
6. 特殊 case 标注：
   - clarification case（4条）不计入意图准确率，但标注是否触发澄清
   - 边界 case（2条）不计入指标，但额外说明
7. 超时自动记录错误并继续下一条

用法：
    cd backend && python eval/batch_http_eval_v3.py
    # 断点重续：重新运行即可，会自动跳过已完成的 case
"""

import asyncio
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.judge_postprocess import judge_single_case, _build_case_context

# ═══════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════

BASE_URL = "http://localhost:8001"
API_PREFIX = "/api/v1"
DATASET_FILE = Path(__file__).resolve().parent / "test_dataset.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "batch_results"
PROGRESS_FILE = RESULTS_DIR / "_progress.json"
REPORT_FILE = RESULTS_DIR / "_batch_report.json"
LOG_FILE = RESULTS_DIR / "_batch_run.log"

HTTP_TIMEOUT = 600.0          # HTTP 请求超时（秒）
CASE_DELAY_BETWEEN = 3.0      # 每条 case 之间的间隔（秒，给后端喘息）

# 特殊 case 标记
CLARIFICATION_CASES: Set[str] = {"eval_chen_13", "eval_li_14", "eval_gen_03", "eval_gen_06"}
BOUNDARY_CASES: Set[str] = {"eval_gen_02", "eval_gen_03"}

# ═══════════════════════════════════════════════════════════════════════
# 日志
# ═══════════════════════════════════════════════════════════════════════

def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ═══════════════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════════════

def load_cases() -> List[dict]:
    cases = []
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": [], "failed": [], "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}


def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def save_result(case_id: str, result: dict):
    path = RESULTS_DIR / f"{case_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def save_failure(case_id: str, error: str):
    path = RESULTS_DIR / f"{case_id}_failed.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"case_id": case_id, "error": error, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, f, ensure_ascii=False, indent=2)

# ═══════════════════════════════════════════════════════════════════════
# 执行顺序
# ═══════════════════════════════════════════════════════════════════════

def build_execution_order(cases: List[dict]) -> List[str]:
    """
    构建执行顺序，确保同一 session_group 的 case 按顺序执行。
    多轮对话的 case 在测试集中按顺序出现，直接按文件顺序即可。
    """
    return [c["session_id"] for c in cases]


def find_case(cases: List[dict], case_id: str) -> Optional[dict]:
    for c in cases:
        if c["session_id"] == case_id:
            return c
    return None

# ═══════════════════════════════════════════════════════════════════════
# HTTP 单条执行
# ═══════════════════════════════════════════════════════════════════════

async def run_single_http_case(
    case: dict,
    session_id: str,
    reset_session: bool,
    resume_id: str,
) -> dict:
    """通过 HTTP 执行单条 case，返回完整结果（含 debug_info）"""
    case_id = case["session_id"]
    message = case["message"]
    eval_ctx = case.get("eval_context", {})

    payload = {
        "message": message,
        "session_id": session_id,
        "eval_context": {
            **eval_ctx,
            "reset_session": reset_session,
            "resume_id": resume_id,  # 放入 eval_context，供后端正确切换简历
        },
    }

    t0 = time.time()
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        resp = await client.post(f"{BASE_URL}{API_PREFIX}/chat", json=payload)

    latency = time.time() - t0

    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")

    body = resp.json()

    reply = body.get("reply", {})
    reply_text = reply.get("content", "") if isinstance(reply, dict) else str(reply)
    intent = body.get("intent", "")
    is_clarification = body.get("is_clarification", False)
    debug_info = body.get("debug_info", {})

    # 提取工具调用
    task_graph = debug_info.get("task_graph", {})
    tool_calls = []
    for tid, task in task_graph.get("tasks", {}).items():
        if task.get("tool_name"):
            tool_calls.append({
                "task_id": tid,
                "tool_name": task.get("tool_name"),
                "status": task.get("status"),
            })

    # 提取意图
    intent_info = debug_info.get("intent", {})
    demands = intent_info.get("demands", [])
    pred_intents = [d.get("intent", "") for d in demands if d.get("intent")]

    # 提取完整 kb_chunks
    kb_chunks = []
    for tid, task in task_graph.get("tasks", {}).items():
        if task.get("tool_name") == "kb_retrieve" and task.get("status") == "success":
            result = task.get("result", {})
            if isinstance(result, dict):
                kb_chunks = result.get("chunks", [])
                break

    return {
        "case_id": case_id,
        "message": message,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "latency_seconds": round(latency, 2),
        "session_id": session_id,
        "reset_session": reset_session,
        "prediction": {
            "intent": intent,
            "is_clarification": is_clarification,
            "pred_intents": pred_intents,
            "reply_full": reply_text,
        },
        "ground_truth": {
            "gold_intents": eval_ctx.get("gold_intents", []),
            "gold_slots": eval_ctx.get("gold_slots", {}),
            "expected_tools": eval_ctx.get("expected_tools", []),
            "scenario": eval_ctx.get("scenario", ""),
            "notes": eval_ctx.get("notes", ""),
        },
        "tool_calls": tool_calls,
        "kb_chunks": kb_chunks,
        "eval_context": eval_ctx,
        "resume_id": resume_id,
        "debug_info": debug_info,
        "raw_response": body,
    }

# ═══════════════════════════════════════════════════════════════════════
# Judge
# ═══════════════════════════════════════════════════════════════════════

async def judge_case(result: dict) -> dict:
    """调用 v3.5 Judge 评估"""
    case_result_for_judge = {
        "case_id": result["case_id"],
        "message": result["message"],
        "gold_intents": result["ground_truth"]["gold_intents"],
        "reply": result["prediction"]["reply_full"],
        "tools_called": [{"tool": t["tool_name"], "status": t["status"]} for t in result["tool_calls"]],
        "pred_intents": result["prediction"]["pred_intents"],
        "kb_chunks": result["kb_chunks"],
        "tool_executions_full": result["tool_calls"],
        "eval_context": result["eval_context"],
        "resume_id": result["resume_id"],
    }

    ctx = _build_case_context(result["case_id"], case_result_for_judge)
    judge_result = await judge_single_case(case_result_for_judge, ctx)
    return judge_result

# ═══════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════

async def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cases = load_cases()
    total = len(cases)
    log(f"=" * 70)
    log(f"批量 HTTP 全链路测试 v3.5 启动")
    log(f"总 case 数: {total}")
    log(f"Clarification case: {len(CLARIFICATION_CASES)} 条 → {sorted(CLARIFICATION_CASES)}")
    log(f"Boundary case: {len(BOUNDARY_CASES)} 条 → {sorted(BOUNDARY_CASES)}")
    log(f"结果目录: {RESULTS_DIR}")
    log(f"=" * 70)

    progress = load_progress()
    completed: Set[str] = set(progress.get("completed", []))
    failed: Set[str] = set(progress.get("failed", []))

    log(f"已完成的 case: {len(completed)} 条")
    log(f"之前失败的 case: {len(failed)} 条")

    execution_order = build_execution_order(cases)

    # 维护 session 状态
    session_map: Dict[str, str] = {}  # session_group -> session_id

    for idx, case_id in enumerate(execution_order, 1):
        if case_id in completed:
            log(f"[{idx}/{total}] {case_id} → 已完成，跳过")
            continue

        case = find_case(cases, case_id)
        if not case:
            log(f"[{idx}/{total}] {case_id} → ❌ 找不到 case 定义")
            continue

        sg = case.get("session_group")
        resume_id = case.get("resume_id", "")

        # 确定 session_id 和是否重置
        if sg:
            if sg not in session_map:
                session_map[sg] = f"batch_{sg}_{int(time.time()*1000)}"
                reset_session = True
                log(f"[{idx}/{total}] {case_id} → 新 session_group: {sg}, session_id={session_map[sg]}")
            else:
                reset_session = False
                log(f"[{idx}/{total}] {case_id} → 复用 session_group: {sg}, session_id={session_map[sg]}")
            session_id = session_map[sg]
        else:
            session_id = f"batch_single_{case_id}_{int(time.time()*1000)}"
            reset_session = True

        log(f"[{idx}/{total}] {case_id} → 开始执行 | msg='{case['message'][:50]}...' | reset={reset_session}")

        try:
            result = await run_single_http_case(case, session_id, reset_session, resume_id)

            # Judge
            t_judge = time.time()
            judge_result = await judge_case(result)
            result["judge"] = judge_result
            judge_time = time.time() - t_judge

            # 立即保存
            save_result(case_id, result)

            # 更新进度
            completed.add(case_id)
            progress["completed"] = sorted(list(completed))
            save_progress(progress)

            scores = judge_result.get("scores", {})
            resolved = judge_result.get("resolved", False)
            log(
                f"[{idx}/{total}] {case_id} → ✅ 完成 | "
                f"latency={result['latency_seconds']:.1f}s | judge={judge_time:.1f}s | "
                f"resolved={resolved} | "
                f"intent={scores.get('intent_accuracy', 0)}/5 | "
                f"faithfulness={scores.get('faithfulness', 0)}/5"
            )

            # 如果是多轮对话且当前轮失败了，可能需要重置 group
            if sg and not resolved:
                log(f"[{idx}/{total}] {case_id} → ⚠️ 任务未完成，session_group {sg} 下轮将重置")
                del session_map[sg]

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            log(f"[{idx}/{total}] {case_id} → ❌ 失败 | {error_msg}")
            traceback.print_exc()

            failed.add(case_id)
            progress["failed"] = sorted(list(failed))
            save_progress(progress)
            save_failure(case_id, error_msg)

            # 多轮对话失败时重置 session_group
            if sg and sg in session_map:
                del session_map[sg]

        # 间隔
        if idx < total:
            await asyncio.sleep(CASE_DELAY_BETWEEN)

    # 生成最终报告
    await generate_final_report(cases, completed, failed)
    log("=" * 70)
    log("批量测试完成")
    log(f"完成: {len(completed)}/{total} | 失败: {len(failed)}/{total}")
    log(f"报告: {REPORT_FILE}")
    log("=" * 70)


async def generate_final_report(cases: List[dict], completed: Set[str], failed: Set[str]):
    """生成最终汇总报告"""
    log("生成最终报告...")

    all_results = []
    for case_id in completed:
        path = RESULTS_DIR / f"{case_id}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                all_results.append(json.load(f))

    # 收集统计
    total = len(cases)
    resolved_count = 0
    scores_sum: Dict[str, float] = {}
    scores_count: Dict[str, int] = {}

    # 特殊 case 统计
    clarification_results = []
    boundary_results = []

    for r in all_results:
        case_id = r["case_id"]
        judge = r.get("judge", {})
        scores = judge.get("scores", {})

        if judge.get("resolved"):
            resolved_count += 1

        for k, v in scores.items():
            scores_sum[k] = scores_sum.get(k, 0) + v
            scores_count[k] = scores_count.get(k, 0) + 1

        # 特殊 case
        if case_id in CLARIFICATION_CASES:
            clarification_results.append({
                "case_id": case_id,
                "message": r["message"],
                "is_clarification_triggered": r["prediction"].get("is_clarification", False),
                "pred_intents": r["prediction"].get("pred_intents", []),
                "resolved": judge.get("resolved", False),
            })

        if case_id in BOUNDARY_CASES:
            boundary_results.append({
                "case_id": case_id,
                "message": r["message"],
                "intent": r["prediction"].get("intent", ""),
                "is_clarification": r["prediction"].get("is_clarification", False),
                "reply_preview": r["prediction"].get("reply_full", "")[:100],
                "resolved": judge.get("resolved", False),
            })

    # 平均分
    dim_avg = {}
    for k in scores_sum:
        dim_avg[k] = round(scores_sum[k] / scores_count[k], 2) if scores_count[k] > 0 else 0

    # 意图识别准确率（排除 clarification 和 boundary）
    intent_correct = 0
    intent_total = 0
    for r in all_results:
        case_id = r["case_id"]
        if case_id in CLARIFICATION_CASES or case_id in BOUNDARY_CASES:
            continue
        judge = r.get("judge", {})
        scores = judge.get("scores", {})
        if scores.get("intent_accuracy", 0) >= 3:
            intent_correct += 1
        intent_total += 1

    intent_acc_rate = round(intent_correct / intent_total, 2) if intent_total > 0 else 0

    report = {
        "meta": {
            "total_cases": total,
            "completed": len(completed),
            "failed": len(failed),
            "resolved": resolved_count,
            "resolved_rate": round(resolved_count / total, 2) if total > 0 else 0,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "dimension_averages": dim_avg,
        "intent_accuracy": {
            "excluded_cases": sorted(list(CLARIFICATION_CASES | BOUNDARY_CASES)),
            "reason": "Clarification和边界case不计入意图准确率",
            "correct": intent_correct,
            "total": intent_total,
            "rate": intent_acc_rate,
        },
        "clarification_cases": {
            "count": len(clarification_results),
            "cases": clarification_results,
        },
        "boundary_cases": {
            "count": len(boundary_results),
            "cases": boundary_results,
        },
        "failed_cases": sorted(list(failed)),
        "case_results": [
            {
                "case_id": r["case_id"],
                "latency": r["latency_seconds"],
                "resolved": r.get("judge", {}).get("resolved", False),
                "intent": r["prediction"].get("intent", ""),
                "scores": r.get("judge", {}).get("scores", {}),
            }
            for r in all_results
        ],
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    log(f"报告已保存: {REPORT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
