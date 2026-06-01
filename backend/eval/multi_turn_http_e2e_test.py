#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多轮对话 HTTP 端到端测试 —— 完整链路 + Judge 评估

走 HTTP /api/v1/chat，按 session_group 复用 session，测试：
1. 多轮对话历史记录是否正确传入
2. 指代消解（"这个岗""上面那个"等）是否正常工作
3. 意图识别 + 完整回复质量
4. Judge 评估任务完成度

用法:
    cd backend && python eval/multi_turn_http_e2e_test.py
"""

import asyncio
import json
import sys
import io
import time
import logging
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import httpx

# 强制 UTF-8 输出（解决 Windows 终端乱码）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Judge 后处理
from eval.judge_postprocess import judge_single_case, _build_case_context

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
for name in ("httpx", "httpcore", "urllib3"):
    logging.getLogger(name).setLevel(logging.WARNING)

EVAL_DIR = Path(__file__).resolve().parent
DATASET_FILE = EVAL_DIR / "test_dataset.jsonl"
RESUMES_FILE = EVAL_DIR / "test_resumes.json"
BASE_URL = "http://127.0.0.1:8003"
CHAT_URL = f"{BASE_URL}/api/v1/chat"

INTENT_ALIASES = {
    "position_explore": "explore",
    "match_assess": "assess",
    "interview_prepare": "prepare",
    "general_chat": "chat",
    "attribute_verify": "verify",
    "resume_manage": "manage",
    "explore": "explore",
    "assess": "assess",
    "prepare": "prepare",
    "verify": "verify",
    "manage": "manage",
    "chat": "chat",
    "clarification": "clarification",
}


def normalize_intent(intent: str) -> str:
    return INTENT_ALIASES.get(intent, intent)


def load_dataset() -> list:
    cases = []
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def load_resumes() -> dict:
    with open(RESUMES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_resume_id(case: dict, resumes: dict) -> str:
    rid = case.get("resume_id")
    if not rid:
        mapping = resumes.get("session_resume_map", {})
        rid = mapping.get(case.get("session_id", ""), "")
    return rid


async def activate_resume(client: httpx.AsyncClient, resume_id: str):
    if not resume_id:
        return
    try:
        resp = await client.put(
            f"{BASE_URL}/api/v1/resumes/{resume_id}/activate", timeout=10.0
        )
        if resp.status_code != 200:
            logger.warning(f"激活简历失败: {resp.status_code} {resp.text[:200]}")
    except Exception as e:
        logger.warning(f"激活简历异常: {e}")


@dataclass
class TurnResult:
    """单轮对话结果"""
    case_id: str
    group: Optional[str]
    turn_index: int
    message: str
    gold_intents: List[str]
    gold_slots: dict
    pred_intents: List[str]
    pred_slots: dict
    reply: str = ""
    needs_clarification: bool = False
    tools_called: List[str] = field(default_factory=list)
    tool_details: List[dict] = field(default_factory=list)
    route_meta: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    error: Optional[str] = None
    judge_result: Optional[dict] = None
    # 多轮追踪
    session_history_before: List[dict] = field(default_factory=list)
    session_history_after: List[dict] = field(default_factory=list)
    working_memory_turns: int = 0


async def eval_single_turn(
    client: httpx.AsyncClient,
    case: dict,
    session_id: str,
    reset_session: bool,
    resume_id: str,
    turn_index: int,
) -> TurnResult:
    """评估单轮对话"""
    message = case["message"]
    eval_ctx = case.get("eval_context", {})
    gold_intents = sorted(set(eval_ctx.get("gold_intents", [])))
    gold_slots = eval_ctx.get("gold_slots", {})

    # 激活简历
    await activate_resume(client, resume_id)

    payload = {
        "session_id": session_id,
        "message": message,
        "eval_context": {"reset_session": True} if reset_session else {},
    }

    t0 = time.time()
    try:
        resp = await client.post(
            CHAT_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=300.0,
        )
        resp.raise_for_status()
        data = resp.json()
        latency_ms = (time.time() - t0) * 1000
    except Exception as e:
        return TurnResult(
            case_id=case["session_id"],
            group=case.get("session_group"),
            turn_index=turn_index,
            message=message,
            gold_intents=gold_intents,
            gold_slots=gold_slots,
            pred_intents=["ERROR"],
            pred_slots={},
            error=str(e),
        )

    # 提取预测意图
    route_meta = data.get("route_meta", {})
    demands = route_meta.get("demands", [])
    is_clarification = data.get("is_clarification", False)

    pred_intents = set()
    pred_slots = {}
    tools_called = []
    tool_details = []

    for d in demands:
        it = d.get("intent_type") or d.get("intent") or ""
        pred_intents.add(normalize_intent(it))
        if d.get("entities"):
            pred_slots.update(d["entities"])
        if d.get("tools"):
            for t in d["tools"]:
                tools_called.append(t.get("tool", ""))
                tool_details.append({
                    "tool": t.get("tool", ""),
                    "status": t.get("status", ""),
                })
    if is_clarification:
        pred_intents.add("clarification")

    pred_intents = sorted(pred_intents)
    reply = data.get("reply", "") or ""

    return TurnResult(
        case_id=case["session_id"],
        group=case.get("session_group"),
        turn_index=turn_index,
        message=message,
        gold_intents=gold_intents,
        gold_slots=gold_slots,
        pred_intents=pred_intents,
        pred_slots=pred_slots,
        reply=reply,
        needs_clarification=is_clarification,
        tools_called=tools_called,
        tool_details=tool_details,
        route_meta=route_meta,
        latency_ms=latency_ms,
    )


async def run_judge_for_turn(turn: TurnResult, case: dict, resumes: dict) -> dict:
    """对单轮结果调用 Judge 评估"""
    eval_ctx = case.get("eval_context", {})

    case_result = {
        "case_id": turn.case_id,
        "message": turn.message,
        "gold_intents": turn.gold_intents,
        "reply": turn.reply,
        "tools_called": turn.tools_called,
        "pred_intents": turn.pred_intents,
        "pred_slots": turn.pred_slots,
        "tool_details": turn.tool_details,
    }

    ctx = _build_case_context(turn.case_id, case_result)
    # 补充 resume
    resume_id = case.get("resume_id", "")
    if resume_id and resume_id in resumes:
        r = resumes[resume_id]
        ps = r.get("parsed_schema", {})
        ctx["resume_info"] = f"姓名：{ps.get('name', '未知')} | 经验：{ps.get('total_years', '未知')}年 | 技能：{', '.join(ps.get('skills', [])[:5])}"

    ctx["gold_slots"] = turn.gold_slots
    ctx["expected_tools"] = eval_ctx.get("expected_tools", [])
    ctx["scenario"] = eval_ctx.get("scenario", "")
    ctx["notes"] = eval_ctx.get("notes", "")
    ctx["follow_up_type"] = eval_ctx.get("follow_up_type", "")

    try:
        jr = await judge_single_case(case_result, ctx)
        return jr
    except Exception as e:
        logger.warning(f"Judge 调用失败 {turn.case_id}: {e}")
        return {
            "resolved": False,
            "scores": {},
            "reason": f"Judge 调用失败: {e}",
            "case_id": turn.case_id,
            "rule_hit": None,
            "veto": False,
        }


async def main():
    cases = load_dataset()
    resumes = load_resumes()

    # 只取有多轮对话的用例
    multi_turn_cases = [c for c in cases if c.get("session_group")]
    print(f"多轮对话用例: {len(multi_turn_cases)} 条\n")

    if not multi_turn_cases:
        print("未找到多轮对话用例（session_group 为空）")
        return

    # 按 group 分组
    groups = defaultdict(list)
    for c in multi_turn_cases:
        groups[c["session_group"]].append(c)

    all_turns: List[TurnResult] = []

    async with httpx.AsyncClient(timeout=180.0, trust_env=False) as client:
        for group_id in sorted(groups.keys()):
            group_cases = sorted(groups[group_id], key=lambda x: x["session_id"])
            print(f"\n{'='*80}")
            print(f"【Group: {group_id}】共 {len(group_cases)} 轮")
            print(f"{'='*80}")

            session_id = None
            for idx, case in enumerate(group_cases):
                reset_session = (idx == 0)
                sid = case["session_id"] if reset_session else session_id
                session_id = sid

                resume_id = get_resume_id(case, resumes)
                print(f"\n  Turn {idx+1}: {case['session_id']} | msg: {case['message'][:40]}...")
                if not reset_session:
                    print(f"    -> 复用 session: {sid}")

                turn = await eval_single_turn(
                    client, case, sid, reset_session, resume_id, idx
                )

                # Judge 评估
                print(f"    -> 调用 Judge...")
                jr = await run_judge_for_turn(turn, case, resumes)
                turn.judge_result = jr

                # 打印结果
                match = set(turn.pred_intents) == set(turn.gold_intents)
                mark = "[PASS]" if match else "[FAIL]"
                intent_str = ",".join(turn.pred_intents)
                gold_str = ",".join(turn.gold_intents)
                print(f"    {mark} Intent: gold=[{gold_str}] pred=[{intent_str}]")
                print(f"       Clarification: {turn.needs_clarification}")
                print(f"       Tools: {turn.tools_called}")
                print(f"       Reply: {turn.reply[:100]}...")
                print(f"       Latency: {turn.latency_ms:.0f}ms")
                print(f"       WM turns after: {turn.working_memory_turns}")

                if jr:
                    resolved = jr.get("resolved", False)
                    veto = jr.get("veto", False)
                    scores = jr.get("scores", {})
                    print(f"       Judge: resolved={resolved} veto={veto}")
                    if scores:
                        key_scores = {
                            k: scores.get(k, 0)
                            for k in ["intent_accuracy", "response_accuracy", "response_completeness", "faithfulness"]
                        }
                        print(f"       Scores: {key_scores}")

                all_turns.append(turn)

    # ═══════════════════════════════════════════════════════
    # 汇总报告
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("【汇总报告】")
    print(f"{'='*80}")

    # 按 group 统计
    for group_id in sorted(groups.keys()):
        group_turns = [t for t in all_turns if t.group == group_id]
        print(f"\nGroup: {group_id}")
        for t in group_turns:
            match = set(t.pred_intents) == set(t.gold_intents)
            jr = t.judge_result or {}
            resolved = jr.get("resolved", False)
            mark = "[OK]" if (match and resolved) else ("[WARN]" if match else "[ERR]")
            print(f"  {mark} {t.case_id}: intents={t.pred_intents} judge_resolved={resolved}")
            if t.error:
                print(f"     ERROR: {t.error}")

    # 全局统计
    total = len(all_turns)
    intent_correct = sum(1 for t in all_turns if set(t.pred_intents) == set(t.gold_intents))
    judge_pass = sum(1 for t in all_turns if t.judge_result and t.judge_result.get("resolved", False))
    avg_latency = sum(t.latency_ms for t in all_turns) / total if total else 0

    print(f"\n{'='*80}")
    print(f"总计: {total} 轮")
    print(f"意图准确率: {intent_correct}/{total} = {intent_correct/total*100:.1f}%")
    print(f"Judge 通过率: {judge_pass}/{total} = {judge_pass/total*100:.1f}%")
    print(f"平均延迟: {avg_latency:.0f}ms")

    # 保存详细结果
    output_path = EVAL_DIR / f"multi_turn_http_e2e_{int(time.time())}.json"
    output_data = {
        "summary": {
            "total_turns": total,
            "intent_correct": intent_correct,
            "intent_accuracy": intent_correct / total * 100 if total else 0,
            "judge_pass": judge_pass,
            "judge_pass_rate": judge_pass / total * 100 if total else 0,
            "avg_latency_ms": avg_latency,
        },
        "turns": [
            {
                "case_id": t.case_id,
                "group": t.group,
                "turn_index": t.turn_index,
                "message": t.message,
                "gold_intents": t.gold_intents,
                "pred_intents": t.pred_intents,
                "needs_clarification": t.needs_clarification,
                "tools_called": t.tools_called,
                "reply": t.reply,
                "latency_ms": t.latency_ms,
                "working_memory_turns": t.working_memory_turns,
                "judge_result": t.judge_result,
                "error": t.error,
            }
            for t in all_turns
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
