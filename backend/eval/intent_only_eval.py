#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
意图识别专项评测脚本
只测意图识别（QueryRewrite -> LLMIntentRouter.route_multi），不执行后续规划/工具链

用法:
    cd backend && python eval/intent_only_eval.py
"""

import asyncio
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.memory import SessionMemory, DialogueTurn
from app.core.query_rewrite import QueryRewriter
from app.core.llm_intent import LLMIntentRouter, LLMIntentType
from app.core.llm_client import LLMClient

EVAL_DIR = Path(__file__).resolve().parent
DATASET_FILE = EVAL_DIR / "test_dataset.jsonl"
RESUMES_FILE = EVAL_DIR / "test_resumes.json"

INTENT_TYPE_MAP = {
    LLMIntentType.EXPLORE: "explore",
    LLMIntentType.ASSESS: "assess",
    LLMIntentType.VERIFY: "verify",
    LLMIntentType.PREPARE: "prepare",
    LLMIntentType.MANAGE: "manage",
    LLMIntentType.CHAT: "chat",
}


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


def get_resume_text(case: dict, resumes: dict) -> str:
    resume_id = case.get("resume_id")
    if not resume_id:
        mapping = resumes.get("session_resume_map", {})
        resume_id = mapping.get(case.get("session_id", ""), "")
    for r in resumes.get("resumes", []):
        if r.get("id") == resume_id:
            return r.get("text", "")
    return ""


def set_session_resume(session: SessionMemory, resume_text: str):
    if not hasattr(session, "global_slots"):
        session.global_slots = {}
    if resume_text:
        session.global_slots["resume_text"] = resume_text
        session.global_slots["resume_available"] = True
    else:
        session.global_slots["resume_text"] = "尚未上传简历"
        session.global_slots["resume_available"] = False


def add_turn_to_session(session: SessionMemory, user_msg: str, assistant_reply: str, intent: str, rewritten_query: str):
    turn_id = len(session.working_memory.turns) + 1
    turn = DialogueTurn(
        turn_id=turn_id,
        user_message=user_msg,
        assistant_reply=assistant_reply,
        intent=intent,
        rewritten_query=rewritten_query,
    )
    session.working_memory.append(turn)
    session.working_memory.pop_overflow()


async def eval_single_case(case: dict, session: SessionMemory, rewriter: QueryRewriter, router: LLMIntentRouter) -> dict:
    message = case["message"]
    eval_ctx = case.get("eval_context", {})
    gold_intents = sorted(set(eval_ctx.get("gold_intents", [])))

    rewrite_result = await rewriter.rewrite(raw_query=message, session=session)
    multi_result = await router.route_multi(
        rewrite_result=rewrite_result,
        session=session,
        attachments=[],
    )

    pred_intents = set()
    for cand in multi_result.candidates:
        pred_intents.add(INTENT_TYPE_MAP.get(cand.intent_type, cand.intent_type.value))
    if multi_result.needs_clarification:
        pred_intents.add("clarification")

    pred_intents = sorted(pred_intents)
    match = set(pred_intents) == set(gold_intents)

    primary = multi_result.primary_intent.value if multi_result.primary_intent else "none"

    # 将本轮解析出的 company/position 同步到 session.global_slots，供下一轮 _check_clarification_need 使用
    for cand in multi_result.candidates:
        if cand.slots.get("company"):
            session.global_slots["company"] = cand.slots["company"]
        if cand.slots.get("position"):
            session.global_slots["position"] = cand.slots["position"]

    add_turn_to_session(
        session=session,
        user_msg=message,
        assistant_reply="[intent_only_eval]",
        intent=primary,
        rewritten_query=rewrite_result.rewritten_query,
    )

    return {
        "case_id": case["session_id"],
        "session_group": case.get("session_group"),
        "message": message,
        "gold_intents": gold_intents,
        "pred_intents": pred_intents,
        "match": match,
        "needs_clarification": multi_result.needs_clarification,
        "primary_intent": primary,
    }


async def main():
    cases = load_dataset()
    resumes = load_resumes()
    print(f"加载 {len(cases)} 条测试用例\n", flush=True)

    rewriter = QueryRewriter()
    llm_for_intent = LLMClient.from_config("chat")
    router = LLMIntentRouter(chat_llm=llm_for_intent)

    groups = defaultdict(list)
    singles = []
    for c in cases:
        g = c.get("session_group")
        if g:
            groups[g].append(c)
        else:
            singles.append(c)

    results = []

    # 处理有 session_group 的多轮用例
    for group_id, group_cases in sorted(groups.items()):
        group_cases.sort(key=lambda c: c["session_id"])
        session = SessionMemory(session_id=group_cases[0]["session_id"])
        resume_text = get_resume_text(group_cases[0], resumes)
        set_session_resume(session, resume_text)

        for case in group_cases:
            r = await eval_single_case(case, session, rewriter, router)
            results.append(r)
            print(f"  -> {r['case_id']}: gold={r['gold_intents']} pred={r['pred_intents']} match={r['match']}", flush=True)

    # 处理单条用例
    for case in singles:
        session = SessionMemory(session_id=case["session_id"])
        resume_text = get_resume_text(case, resumes)
        set_session_resume(session, resume_text)

        r = await eval_single_case(case, session, rewriter, router)
        results.append(r)
        print(f"  -> {r['case_id']}: gold={r['gold_intents']} pred={r['pred_intents']} match={r['match']}", flush=True)

    # 按原始顺序排序
    case_order = {c["session_id"]: i for i, c in enumerate(cases)}
    results.sort(key=lambda r: case_order.get(r["case_id"], 999))

    total = len(results)
    correct = sum(1 for r in results if r["match"])

    print("\n" + "=" * 110, flush=True)
    print(f"{'Case ID':<18} {'Match':<6} {'Gold Intents':<30} {'Pred Intents':<30} {'Message'}", flush=True)
    print("=" * 110, flush=True)
    for r in results:
        gold_str = ", ".join(r["gold_intents"])
        pred_str = ", ".join(r["pred_intents"])
        match_mark = "PASS" if r["match"] else "FAIL"
        msg = r["message"][:35] + "..." if len(r["message"]) > 35 else r["message"]
        print(f"{r['case_id']:<18} {match_mark:<6} {gold_str:<30} {pred_str:<30} {msg}", flush=True)
    print("=" * 110, flush=True)
    print(f"\n总计: {total} 条 | 命中: {correct} 条 | 准确率: {correct / total * 100:.1f}%", flush=True)

    misses = [r for r in results if not r["match"]]
    if misses:
        print(f"\n未命中详情 ({len(misses)} 条):", flush=True)
        for r in misses:
            gold_str = ", ".join(r["gold_intents"])
            pred_str = ", ".join(r["pred_intents"])
            print(f"  {r['case_id']}: gold=[{gold_str}] pred=[{pred_str}]", flush=True)
            print(f"    msg: {r['message']}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
