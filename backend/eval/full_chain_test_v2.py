#!/usr/bin/env python3
"""
全链路 HTTP 端到端测试 —— 统一 v3.5 Judge 体系（唯一入口）

功能：
1. 通过 HTTP 调用 /api/v1/chat（评测模式，返回 debug_info）
2. 记录完整中间链路（Query改写/意图识别/任务图/Reflection/EvidenceCache）
3. v3.5 Judge 评估：12 维 0-5 分 + resolved + faithfulness（基于检索 chunks 交叉验证）
4. 保存完整测试报告（JSON）

用法：
    cd backend && python eval/full_chain_test_v2.py --case eval_chen_03
    cd backend && python eval/full_chain_test_v2.py --case eval_chen_03 --no-judge
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 复用项目已有的 v3.5 Judge 体系
from eval.judge_postprocess import judge_single_case, _build_case_context

BASE_URL = "http://localhost:8001"
API_PREFIX = "/api/v1"
DATASET_FILE = Path(__file__).resolve().parent / "test_dataset.jsonl"


def load_case(case_id: str) -> Optional[dict]:
    """从测试集中加载指定 case"""
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        for line in f:
            case = json.loads(line.strip())
            if case["session_id"] == case_id:
                return case
    return None


def extract_replan_info(task_graph: dict) -> dict:
    """从任务图中提取 Replan 相关信息"""
    replan_info = {
        "replan_count": task_graph.get("replan_count", 0),
        "replan_reason": task_graph.get("replan_reason", ""),
        "replan_triggered": task_graph.get("replan_count", 0) > 0,
    }
    tasks = task_graph.get("tasks", {})
    task_states = []
    for tid, task in tasks.items():
        task_states.append({
            "task_id": tid,
            "tool_name": task.get("tool_name"),
            "status": task.get("status"),
            "description": task.get("description", "")[:100],
            "observation": task.get("observation", "")[:200],
        })
    replan_info["task_states"] = task_states
    return replan_info


def extract_tool_calls(task_graph: dict) -> List[dict]:
    """提取所有工具调用的详细信息（报告用，允许截断）"""
    tools = []
    tasks = task_graph.get("tasks", {})
    for tid, task in tasks.items():
        if task.get("tool_name"):
            tools.append({
                "task_id": tid,
                "tool_name": task.get("tool_name"),
                "task_type": task.get("task_type"),
                "description": task.get("description", ""),
                "parameters": task.get("parameters", {}),
                "resolved_params": task.get("resolved_params", {}),
                "dependencies": task.get("dependencies", []),
                "status": task.get("status"),
                "result": _truncate_result(task.get("result")),
                "observation": task.get("observation", "")[:300],
                "started_at": task.get("started_at"),
                "finished_at": task.get("finished_at"),
            })
    return tools


def extract_kb_chunks(task_graph: dict) -> List[dict]:
    """提取完整的 kb_retrieve chunks（用于 Judge faithfulness 评判，不截断）"""
    tasks = task_graph.get("tasks", {})
    for tid, task in tasks.items():
        if task.get("tool_name") == "kb_retrieve" and task.get("status") == "success":
            result = task.get("result", {})
            if isinstance(result, dict):
                return result.get("chunks", [])
    return []


def _truncate_result(result: Any) -> Any:
    """截断过长的结果，避免报告过大"""
    if result is None:
        return None
    if isinstance(result, dict):
        truncated = {}
        for k, v in result.items():
            if k == "chunks" and isinstance(v, list):
                truncated[k] = [c.get("content", "")[:200] if isinstance(c, dict) else str(c)[:200] for c in v[:3]]
                if len(v) > 3:
                    truncated[k + "_note"] = f"共{len(v)}条，已截断"
            elif isinstance(v, str) and len(v) > 500:
                truncated[k] = v[:500] + "...[截断]"
            elif isinstance(v, list) and len(v) > 10:
                truncated[k] = v[:10]
                truncated[k + "_note"] = f"共{len(v)}项，已截断"
            else:
                truncated[k] = v
        return truncated
    if isinstance(result, str) and len(result) > 500:
        return result[:500] + "...[截断]"
    return result


def extract_reflection_info(reflection_result: dict) -> dict:
    """提取 Reflection 详细信息"""
    if not reflection_result:
        return {"has_reflection": False}

    info = {
        "has_reflection": True,
        "suggested_action": reflection_result.get("suggested_action"),
        "confidence": reflection_result.get("confidence"),
        "reason": reflection_result.get("reason", ""),
        "problematic_task": reflection_result.get("problematic_task", ""),
        "is_complete": reflection_result.get("is_complete"),
        "has_conflict": reflection_result.get("has_conflict"),
        "missing_info": reflection_result.get("missing_info", []),
    }

    sr = reflection_result.get("semantic_relevance", {})
    if sr:
        info["semantic_relevance"] = {
            "is_relevant": sr.get("is_relevant"),
            "issue": sr.get("issue", ""),
            "suggested_new_query": sr.get("suggested_new_query", ""),
        }

    sc = reflection_result.get("source_conflict_analysis", {})
    if sc:
        info["source_conflict_analysis"] = {
            "has_conflict": sc.get("has_conflict"),
            "analysis": sc.get("analysis", "")[:300],
            "recommendation": sc.get("recommendation", "")[:300],
            "severity": sc.get("severity", ""),
        }

    return info


async def run_single_case(case: dict, do_judge: bool = True) -> dict:
    """执行单个 case 的全链路测试"""
    case_id = case["session_id"]
    message = case["message"]
    eval_ctx = case.get("eval_context", {})
    resume_id = case.get("resume_id", "cc50dcfb-aeba-41a3-989a-9e485aa3f")

    session_id = f"test_{case_id}_{int(time.time() * 1000)}"

    # 构造请求体（评测模式）
    payload = {
        "message": message,
        "session_id": session_id,
        "eval_context": {
            **eval_ctx,
            "reset_session": True,
            "resume_id": resume_id,  # 放入 eval_context，供后端正确切换简历
        },
    }

    print(f"\n{'='*80}")
    print(f"【全链路测试】Case: {case_id}")
    print(f"用户消息: {message}")
    print(f"期望意图: {eval_ctx.get('gold_intents', [])}")
    print(f"期望槽位: {eval_ctx.get('gold_slots', {})}")
    print(f"{'='*80}")

    t0 = time.time()

    async with httpx.AsyncClient(timeout=600.0) as client:
        resp = await client.post(
            f"{BASE_URL}{API_PREFIX}/chat",
            json=payload,
        )

    total_latency = time.time() - t0

    if resp.status_code != 200:
        print(f"❌ HTTP 错误: {resp.status_code}")
        print(f"响应: {resp.text[:500]}")
        return {"error": f"HTTP {resp.status_code}", "response_text": resp.text[:1000]}

    body = resp.json()

    # 提取基础信息
    reply = body.get("reply", {})
    reply_text = reply.get("content", "") if isinstance(reply, dict) else str(reply)
    intent = body.get("intent", "")
    is_clarification = body.get("is_clarification", False)

    print(f"\n✅ 请求成功 | 总延迟: {total_latency:.2f}s")
    print(f"预测意图: {intent}")
    print(f"是否澄清: {is_clarification}")
    print(f"回复长度: {len(reply_text)} 字符")
    print(f"回复预览: {reply_text[:150]}...")

    # 提取 debug_info（评测模式特有）
    debug_info = body.get("debug_info", {})

    # 1. Query 改写
    rewrite = debug_info.get("rewrite", {})
    rewrite_info = {
        "original_message": case["message"],
        "rewritten_query": rewrite.get("rewritten_query", ""),
        "search_keywords": rewrite.get("search_keywords", ""),
        "is_follow_up": rewrite.get("is_follow_up", False),
        "follow_up_type": rewrite.get("follow_up_type", ""),
        "resolved_references": rewrite.get("resolved_references", {}),
    }

    # 2. 意图识别
    intent_info = debug_info.get("intent", {})
    demands = intent_info.get("demands", [])
    pred_intents = [d.get("intent", "") for d in demands if d.get("intent")]

    # 3. 任务规划图
    task_graph = debug_info.get("task_graph", {})
    replan_info = extract_replan_info(task_graph)
    tool_calls = extract_tool_calls(task_graph)

    # 4. Reflection
    reflection_result = debug_info.get("reflection_result", {})
    reflection_info = extract_reflection_info(reflection_result)

    # 5. Evidence Cache
    evidence_cache = debug_info.get("evidence_cache", [])

    # 6. 聚合 Prompt
    final_prompt = debug_info.get("final_aggregation_prompt", {})

    # 7. 提取完整 kb_chunks（用于 Judge faithfulness）
    kb_chunks = extract_kb_chunks(task_graph)

    # 打印中间链路摘要
    print(f"\n{'─'*80}")
    print("【中间链路摘要】")
    print(f"{'─'*80}")

    print(f"\n[Query 改写]")
    print(f"  改写后: {rewrite_info['rewritten_query'][:80]}...")
    print(f"  搜索关键词: {rewrite_info['search_keywords']}")
    print(f"  是否追问: {rewrite_info['is_follow_up']} ({rewrite_info['follow_up_type']})")

    print(f"\n[意图识别]")
    for d in demands:
        print(f"  意图: {d.get('intent')} | 置信度: {d.get('confidence', 0):.2f}")
    print(f"  是否需要澄清: {intent_info.get('needs_clarification', False)}")

    print(f"\n[任务规划图]")
    print(f"  任务数量: {len(tool_calls)}")
    print(f"  Replan 次数: {replan_info['replan_count']}")
    print(f"  Replan 原因: {replan_info['replan_reason'] or '无'}")
    for t in tool_calls:
        status_icon = "✅" if t['status'] == 'success' else "❌" if t['status'] == 'failed' else "⏭"
        print(f"  {status_icon} {t['task_id']}: {t['tool_name']} | status={t['status']}")

    print(f"\n[Reflection]")
    if reflection_info["has_reflection"]:
        print(f"  建议动作: {reflection_info['suggested_action']}")
        print(f"  置信度: {reflection_info['confidence']}")
        print(f"  原因: {reflection_info['reason'][:100]}...")
        if "semantic_relevance" in reflection_info:
            sr = reflection_info["semantic_relevance"]
            print(f"  语义相关性: {'✅ 相关' if sr['is_relevant'] else '❌ 不相关'} | {sr.get('issue', '')}")
        if "source_conflict_analysis" in reflection_info:
            sc = reflection_info["source_conflict_analysis"]
            print(f"  来源冲突: {'✅ 有冲突' if sc['has_conflict'] else '✅ 无冲突'} | severity={sc.get('severity', 'N/A')}")
    else:
        print("  无 Reflection 数据")

    print(f"\n[Evidence Cache]")
    print(f"  缓存条数: {len(evidence_cache)}")

    # 构建完整报告
    report = {
        "case_id": case_id,
        "message": message,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "latency": {
            "total_seconds": round(total_latency, 2),
        },
        "prediction": {
            "intent": intent,
            "is_clarification": is_clarification,
            "reply_preview": reply_text[:300],
            "reply_full": reply_text,
        },
        "ground_truth": {
            "gold_intents": eval_ctx.get("gold_intents", []),
            "gold_slots": eval_ctx.get("gold_slots", {}),
            "expected_tools": eval_ctx.get("expected_tools", []),
            "scenario": eval_ctx.get("scenario", ""),
            "notes": eval_ctx.get("notes", ""),
        },
        "intermediate_chain": {
            "query_rewrite": rewrite_info,
            "intent_recognition": intent_info,
            "task_graph": {
                "global_status": task_graph.get("global_status", ""),
                **replan_info,
                "tool_calls": tool_calls,
            },
            "reflection": reflection_info,
            "evidence_cache": {
                "count": len(evidence_cache),
                "items": evidence_cache[:5] if evidence_cache else [],
            },
            "final_aggregation_prompt": {
                "system_prompt": final_prompt.get("system_prompt", "")[:500] + "...[截断]" if len(final_prompt.get("system_prompt", "")) > 500 else final_prompt.get("system_prompt", ""),
                "user_prompt": final_prompt.get("user_prompt", "")[:500] + "...[截断]" if len(final_prompt.get("user_prompt", "")) > 500 else final_prompt.get("user_prompt", ""),
            },
        },
        "raw_response": body,
    }

    # v3.5 Judge 评估
    if do_judge and reply_text:
        # 构建 v3.5 Judge 需要的 case_result 格式
        case_result_for_judge = {
            "case_id": case_id,
            "message": message,
            "gold_intents": eval_ctx.get("gold_intents", []),
            "reply": reply_text,
            "tools_called": [{"tool": t["tool_name"], "status": t["status"]} for t in tool_calls],
            "pred_intents": pred_intents,
            "kb_chunks": kb_chunks,
            "tool_executions_full": tool_calls,
            "eval_context": eval_ctx,
            "resume_id": resume_id,
        }

        ctx = _build_case_context(case_id, case_result_for_judge)
        judge_result = await judge_single_case(case_result_for_judge, ctx)

        # 格式化输出到控制台
        print(f"\n{'─'*80}")
        print("【Judge 评估 (v3.5)】")
        print(f"{'─'*80}")
        scores = judge_result.get("scores", {})
        print(f"任务完成: {'✅ resolved' if judge_result.get('resolved') else '❌ unresolved'}")
        print(f"规则触发: {judge_result.get('rule_hit') or '无'}")
        print(f"否决项: {'是' if judge_result.get('veto') else '否'}")
        print(f"\n【关键维度】")
        for dim in ["intent_accuracy", "slot_accuracy", "tool_correctness", "tool_execution",
                    "response_accuracy", "response_completeness"]:
            print(f"  {dim}: {scores.get(dim, 0)}/5")
        print(f"\n【辅助维度】")
        for dim in ["citation_quality", "coherence", "tone", "efficiency"]:
            print(f"  {dim}: {scores.get(dim, 0)}/5")
        print(f"\n【RAG 专属】")
        for dim in ["faithfulness", "answer_relevance"]:
            print(f"  {dim}: {scores.get(dim, 0)}/5")
        print(f"\n评价理由: {judge_result.get('reason', '')[:200]}...")

        report["judge"] = judge_result

    return report


async def main():
    parser = argparse.ArgumentParser(description="全链路 HTTP 端到端测试 — v3.5 Judge 体系")
    parser.add_argument("--case", required=True, help="测试用例 ID，如 eval_chen_03")
    parser.add_argument("--no-judge", action="store_true", help="跳过 Judge 评估")
    parser.add_argument("--output", default=None, help="输出报告路径")
    args = parser.parse_args()

    case = load_case(args.case)
    if not case:
        print(f"❌ 未找到 case: {args.case}")
        print(f"可用 cases:")
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            for line in f:
                c = json.loads(line.strip())
                print(f"  {c['session_id']}: {c['message'][:50]}...")
        return

    report = await run_single_case(case, do_judge=not args.no_judge)

    output_path = Path(args.output) if args.output else Path(__file__).resolve().parent / f"full_chain_v3_{args.case}_{int(time.time())}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*80}")
    print(f"✅ 完整测试报告已保存: {output_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    asyncio.run(main())
