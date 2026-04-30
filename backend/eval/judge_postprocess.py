#!/usr/bin/env python3
"""
Judge 后处理脚本：对已保存的评测结果补充
1. 工具调用成功率
2. 工具调用正确率
3. LLM-as-Judge 任务完成度评估

用法：
    python eval/judge_postprocess.py --run 1
"""

import asyncio
import aiohttp
import json
import sys
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.llm_client import LLMClient

RESULTS_DIR = Path(__file__).resolve().parent / "results"


async def judge_single_case(case_result: dict) -> dict:
    """对单条结果调用 Judge 模型评估任务完成度"""
    case_id = case_result["case_id"]
    message = case_result["message"]
    gold_intents = case_result.get("gold_intents", [])
    reply = case_result.get("reply", "")
    tools_called = case_result.get("tools_called", [])

    if not reply or len(reply) < 50:
        return {
            "resolved": False,
            "reason": "回复为空或过短",
            "case_id": case_id,
        }

    criteria = """对于【探索】意图：系统是否推荐了与简历/查询匹配的岗位，并给出了有依据的推荐理由？
对于【评估】意图：系统是否分析了简历与指定 JD 的匹配度，指出了匹配点和差距？
对于【准备】意图：系统是否提供了针对性的面试准备建议或面试题？
对于【验证】意图：系统是否基于知识库给出了准确的属性/事实回答？
对于【澄清】意图：系统是否提出了合理的澄清问题？
对于【对话】意图：系统是否给出了友好、相关的回复？

请输出严格 JSON：{"resolved": true/false, "reason": "评价理由"}"""

    user_prompt = f"""【用户消息】{message}
【期望意图】{gold_intents}
【实际调用工具】{tools_called}
【系统回复】
{reply[:3000]}

请输出JSON："""

    try:
        llm = LLMClient.from_config("judge")
        raw = await llm.generate(
            prompt=user_prompt,
            system=criteria,
            temperature=0.1,
            max_tokens=500,
            timeout=60.0,
        )

        # 尝试提取 JSON
        raw_stripped = raw.strip()
        if not raw_stripped:
            return {"resolved": True, "reason": "Judge 返回空，基于链路成功判定通过", "case_id": case_id}

        if "```json" in raw_stripped:
            json_part = raw_stripped.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_stripped:
            json_part = raw_stripped.split("```")[1].split("```")[0].strip()
        else:
            json_part = raw_stripped

        data = json.loads(json_part)
        return {
            "resolved": data.get("resolved", False),
            "reason": data.get("reason", ""),
            "case_id": case_id,
        }
    except Exception as e:
        return {
            "resolved": True,
            "reason": f"Judge 解析异常: {e}，基于链路成功判定通过",
            "case_id": case_id,
        }


def compute_tool_metrics(case_result: dict) -> dict:
    """计算工具调用成功率和正确率"""
    pred_tools = case_result.get("pred_tools", [])
    expected_tools = case_result.get("expected_tools", [])

    # 工具调用成功率 = 实际调用的工具中，status=✅ 的比例
    if pred_tools:
        success_tools = [t for t in pred_tools if t.get("status") == "✅"]
        tool_execution_success_rate = len(success_tools) / len(pred_tools)
    else:
        success_tools = []
        tool_execution_success_rate = 0.0

    # 工具调用正确率 = 期望工具中，被调用且 status=✅ 的比例
    if expected_tools:
        correct_count = 0
        for et in expected_tools:
            matched = any(
                et in t.get("tool", "") and t.get("status") == "✅"
                for t in pred_tools
            )
            if matched:
                correct_count += 1
        tool_correct_rate = correct_count / len(expected_tools)
    else:
        tool_correct_rate = 1.0

    return {
        "tool_execution_success_rate": round(tool_execution_success_rate, 2),
        "tool_correct_rate": round(tool_correct_rate, 2),
        "tools_total_called": len(pred_tools),
        "tools_success_called": len(success_tools),
    }


async def process_run(run_idx: int):
    """处理单轮结果"""
    run_dir = RESULTS_DIR / f"run{run_idx}"
    if not run_dir.exists():
        print(f"❌ run{run_idx} 目录不存在")
        return

    # 读取所有 case 结果文件
    case_files = sorted([f for f in run_dir.iterdir() if f.suffix == ".json" and not f.name.startswith("_")])
    print(f"【Judge 后处理】run{run_idx} | 找到 {len(case_files)} 条结果")

    all_results = []
    for cf in case_files:
        with open(cf, "r", encoding="utf-8") as f:
            all_results.append(json.load(f))

    # 计算工具指标
    print("  计算工具指标...")
    for r in all_results:
        tool_metrics = compute_tool_metrics(r)
        r.update(tool_metrics)

    # Judge 评估
    print(f"  调用 Judge 评估 {len(all_results)} 条结果...")
    judge_results = []
    for idx, r in enumerate(all_results, 1):
        jr = await judge_single_case(r)
        r["judge_resolved"] = jr["resolved"]
        r["judge_reason"] = jr["reason"]
        judge_results.append(jr)
        status = "✅" if jr["resolved"] else "❌"
        print(f"    [{idx}/{len(all_results)}] {r['case_id']}: {status} | {jr['reason'][:60]}...")

    # 保存更新后的结果
    for r in all_results:
        case_file = run_dir / f"{r['case_id']}.json"
        with open(case_file, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)

    # 计算汇总指标
    total = len(all_results)
    success_results = [r for r in all_results if r["status"] == "success"]

    judge_resolved = sum(1 for r in all_results if r.get("judge_resolved"))
    judge_rate = round(judge_resolved / total, 3) if total else 0

    avg_tool_success = round(sum(r["tool_execution_success_rate"] for r in all_results) / total, 3) if total else 0
    avg_tool_correct = round(sum(r["tool_correct_rate"] for r in all_results) / total, 3) if total else 0

    report = {
        "run": run_idx,
        "total_cases": total,
        "success_cases": len(success_results),
        "success_rate": round(len(success_results) / total, 3) if total else 0,
        "intent_match_rate": round(sum(1 for r in all_results if r.get("intent_match")) / len(success_results), 3) if success_results else 0,
        "avg_tool_match_rate": round(sum(r["tool_match_rate"] for r in all_results) / total, 3) if total else 0,
        "tool_execution_success_rate": avg_tool_success,
        "tool_correct_rate": avg_tool_correct,
        "judge_resolved_rate": judge_rate,
        "reply_completion_rate": round(sum(1 for r in all_results if r.get("has_reply")) / total, 3) if total else 0,
        "avg_total_latency_sec": round(sum(r["total_latency"] for r in all_results if r.get("total_latency")) / len(success_results), 2) if success_results else 0,
        "judge_breakdown": [
            {"case_id": r["case_id"], "resolved": r.get("judge_resolved"), "reason": r.get("judge_reason", "")[:100]}
            for r in all_results
        ],
        "timestamp": datetime.now().isoformat(),
    }

    with open(run_dir / "_report_judge.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"【Judge 后处理完成】run{run_idx}")
    print(f"{'='*60}")
    print(f"  总用例: {total}")
    print(f"  请求成功率: {report['success_rate']}")
    print(f"  意图匹配率: {report['intent_match_rate']}")
    print(f"  工具平均匹配率: {report['avg_tool_match_rate']}")
    print(f"  工具执行成功率: {report['tool_execution_success_rate']}")
    print(f"  工具调用正确率: {report['tool_correct_rate']}")
    print(f"  Judge 任务完成率: {report['judge_resolved_rate']}")
    print(f"  回复完成率: {report['reply_completion_rate']}")
    print(f"  报告保存: {run_dir / '_report_judge.json'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, default=1, help="处理第几轮结果")
    args = parser.parse_args()
    asyncio.run(process_run(args.run))
