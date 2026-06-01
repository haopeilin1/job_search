#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多轮对话 HTTP 端到端测试 —— 精简版（只跑关键 clarify 场景）

重点验证：
1. session 复用（历史记录是否正确传入）
2. 指代消解（"这个岗""上面那个"）
3. 完整链路 + Judge 评估

用法:
    cd backend && PYTHONUNBUFFERED=1 python eval/multi_turn_focused_test.py
"""

import asyncio
import json
import sys
import io
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.judge_postprocess import judge_single_case, _build_case_context

# 强制 UTF-8 + 无缓冲输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
for name in ("httpx", "httpcore", "urllib3"):
    logging.getLogger(name).setLevel(logging.WARNING)

EVAL_DIR = Path(__file__).resolve().parent
BASE_URL = "http://127.0.0.1:8003"
CHAT_URL = f"{BASE_URL}/api/v1/chat"

INTENT_ALIASES = {
    "position_explore": "explore", "match_assess": "assess",
    "interview_prepare": "prepare", "general_chat": "chat",
    "attribute_verify": "verify", "resume_manage": "manage",
    "explore": "explore", "assess": "assess", "prepare": "prepare",
    "verify": "verify", "manage": "manage", "chat": "chat",
    "clarification": "clarification",
}


def normalize_intent(intent: str) -> str:
    return INTENT_ALIASES.get(intent, intent)


def load_dataset() -> list:
    cases = []
    with open(EVAL_DIR / "test_dataset.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def load_resumes() -> dict:
    with open(EVAL_DIR / "test_resumes.json", "r", encoding="utf-8") as f:
        return json.load(f)


async def activate_resume(client: httpx.AsyncClient, resume_id: str):
    if not resume_id:
        return
    try:
        resp = await client.put(f"{BASE_URL}/api/v1/resumes/{resume_id}/activate", timeout=10.0)
        if resp.status_code != 200:
            print(f"[WARN] 激活简历失败: {resp.status_code}", flush=True)
    except Exception as e:
        print(f"[WARN] 激活简历异常: {e}", flush=True)


@dataclass
class TurnResult:
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
    latency_ms: float = 0.0
    error: Optional[str] = None
    judge_result: Optional[dict] = None


async def eval_single_turn(client, case, session_id, reset_session, resume_id, turn_index) -> TurnResult:
    message = case["message"]
    eval_ctx = case.get("eval_context", {})
    gold_intents = sorted(set(eval_ctx.get("gold_intents", [])))
    gold_slots = eval_ctx.get("gold_slots", {})

    await activate_resume(client, resume_id)

    payload = {
        "session_id": session_id,
        "message": message,
        "eval_context": {"reset_session": True} if reset_session else {},
    }

    t0 = time.time()
    try:
        resp = await client.post(CHAT_URL, json=payload, headers={"Content-Type": "application/json"}, timeout=300.0)
        resp.raise_for_status()
        data = resp.json()
        latency_ms = (time.time() - t0) * 1000
    except Exception as e:
        return TurnResult(
            case_id=case["session_id"], group=case.get("session_group"),
            turn_index=turn_index, message=message,
            gold_intents=gold_intents, gold_slots=gold_slots,
            pred_intents=["ERROR"], pred_slots={}, error=str(e),
        )

    route_meta = data.get("route_meta", {})
    demands = route_meta.get("demands", [])
    is_clarification = data.get("is_clarification", False)

    pred_intents = set()
    pred_slots = {}
    tools_called = []

    for d in demands:
        it = d.get("intent_type") or d.get("intent") or ""
        pred_intents.add(normalize_intent(it))
        if d.get("entities"):
            pred_slots.update(d["entities"])
        if d.get("tools"):
            for t in d["tools"]:
                tools_called.append(t.get("tool", ""))
    if is_clarification:
        pred_intents.add("clarification")

    pred_intents = sorted(pred_intents)
    reply = data.get("reply", "") or ""

    return TurnResult(
        case_id=case["session_id"], group=case.get("session_group"),
        turn_index=turn_index, message=message,
        gold_intents=gold_intents, gold_slots=gold_slots,
        pred_intents=pred_intents, pred_slots=pred_slots,
        reply=reply, needs_clarification=is_clarification,
        tools_called=tools_called, latency_ms=latency_ms,
    )


async def run_judge_for_turn(turn: TurnResult, case: dict, resumes: dict) -> dict:
    eval_ctx = case.get("eval_context", {})
    case_result = {
        "case_id": turn.case_id,
        "message": turn.message,
        "gold_intents": turn.gold_intents,
        "reply": turn.reply,
        "tools_called": turn.tools_called,
        "pred_intents": turn.pred_intents,
        "pred_slots": turn.pred_slots,
        "tool_details": [],
    }
    ctx = _build_case_context(turn.case_id, case_result)
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
        print(f"[WARN] Judge 调用失败 {turn.case_id}: {e}", flush=True)
        return {"resolved": False, "scores": {}, "reason": f"Judge 调用失败: {e}", "case_id": turn.case_id, "veto": False}


async def main():
    cases = load_dataset()
    resumes = load_resumes()

    # 只跑 clarify 相关的多轮 group（最核心的指代消解场景）
    target_groups = ["chen_m2", "li_m2"]
    filtered = [c for c in cases if c.get("session_group") in target_groups]
    print(f"目标多轮对话用例: {len(filtered)} 条 (groups={target_groups})\n", flush=True)

    if not filtered:
        print("未找到目标多轮对话用例", flush=True)
        return

    groups = {}
    for c in filtered:
        g = c["session_group"]
        groups.setdefault(g, []).append(c)

    all_turns = []

    async with httpx.AsyncClient(timeout=180.0, trust_env=False) as client:
        for group_id in sorted(groups.keys()):
            group_cases = sorted(groups[group_id], key=lambda x: x["session_id"])
            print(f"\n{'='*70}", flush=True)
            print(f"[Group: {group_id}] 共 {len(group_cases)} 轮", flush=True)
            print(f"{'='*70}", flush=True)

            session_id = None
            for idx, case in enumerate(group_cases):
                reset_session = (idx == 0)
                sid = case["session_id"] if reset_session else session_id
                session_id = sid
                resume_id = case.get("resume_id", "")

                print(f"\n  Turn {idx+1}: {case['session_id']}", flush=True)
                print(f"    Msg: {case['message']}", flush=True)
                if not reset_session:
                    print(f"    -> 复用 session: {sid}", flush=True)

                turn = await eval_single_turn(client, case, sid, reset_session, resume_id, idx)

                # Judge
                print(f"    -> Judge 评估中...", flush=True)
                jr = await run_judge_for_turn(turn, case, resumes)
                turn.judge_result = jr

                match = set(turn.pred_intents) == set(turn.gold_intents)
                mark = "[PASS]" if match else "[FAIL]"
                print(f"    {mark} Intent: gold={turn.gold_intents} pred={turn.pred_intents}", flush=True)
                print(f"       Clarify: {turn.needs_clarification}", flush=True)
                print(f"       Tools: {turn.tools_called}", flush=True)
                print(f"       Reply: {turn.reply[:120]}...", flush=True)
                print(f"       Latency: {turn.latency_ms:.0f}ms", flush=True)

                if jr:
                    resolved = jr.get("resolved", False)
                    veto = jr.get("veto", False)
                    scores = jr.get("scores", {})
                    print(f"       Judge: resolved={resolved} veto={veto}", flush=True)
                    if scores:
                        key_scores = {k: scores.get(k, 0) for k in ["intent_accuracy", "response_accuracy", "response_completeness", "faithfulness"]}
                        print(f"       Scores: {key_scores}", flush=True)

                all_turns.append(turn)

    # 汇总
    print(f"\n{'='*70}", flush=True)
    print("[汇总报告]", flush=True)
    print(f"{'='*70}", flush=True)

    total = len(all_turns)
    intent_correct = sum(1 for t in all_turns if set(t.pred_intents) == set(t.gold_intents))
    judge_pass = sum(1 for t in all_turns if t.judge_result and t.judge_result.get("resolved", False))
    avg_latency = sum(t.latency_ms for t in all_turns) / total if total else 0

    print(f"总计: {total} 轮", flush=True)
    print(f"意图准确率: {intent_correct}/{total} = {intent_correct/total*100:.1f}%", flush=True)
    print(f"Judge 通过率: {judge_pass}/{total} = {judge_pass/total*100:.1f}%", flush=True)
    print(f"平均延迟: {avg_latency:.0f}ms", flush=True)

    # 保存结果
    output_path = EVAL_DIR / f"multi_turn_focused_{int(time.time())}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
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
                    "case_id": t.case_id, "group": t.group, "turn_index": t.turn_index,
                    "message": t.message, "gold_intents": t.gold_intents,
                    "pred_intents": t.pred_intents, "needs_clarification": t.needs_clarification,
                    "tools_called": t.tools_called, "reply": t.reply,
                    "latency_ms": t.latency_ms, "judge_result": t.judge_result,
                    "error": t.error,
                }
                for t in all_turns
            ],
        }, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存: {output_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
