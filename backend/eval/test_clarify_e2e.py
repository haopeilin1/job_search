#!/usr/bin/env python3
"""
端到端验证 clarify 后意图恢复：模拟 run_v2_eval.py 的两轮流程
"""
import asyncio
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).parent
sys.path.insert(0, str(EVAL_DIR.parent))

from app.core.memory import SessionMemory, DialogueTurn, PendingClarification
from app.core.query_rewrite import QueryRewriter
from app.core.llm_intent import LLMIntentRouter
from app.core.new_arch_adapter import multi_intent_result_to_intent_result


async def run_turn(session: SessionMemory, message: str, resume_text: str = ""):
    """模拟 run_single_case 的核心流程"""
    rewriter = QueryRewriter()
    rw = await rewriter.rewrite(raw_query=message, session=session)

    router = LLMIntentRouter()
    multi_result = await router.route_multi(
        rewrite_result=rw,
        session=session,
        attachments=[],
        raw_message=message,
    )
    intent_result = multi_intent_result_to_intent_result(multi_result)

    # 澄清场景保存（与 run_v2_eval.py 一致）
    if intent_result.needs_clarification:
        turn_id = len(session.working_memory.turns) + 1
        turn = DialogueTurn(
            turn_id=turn_id,
            user_message=message,
            assistant_reply=intent_result.clarification_question or "抱歉，我没有完全理解您的意思，能再详细说明一下吗？",
            intent=intent_result.demands[0].intent_type if intent_result.demands else "chat",
            rewritten_query=rw.rewritten_query,
            evidence_score=0.0,
        )
        session.working_memory.append(turn)
        primary = intent_result.demands[0].intent_type if intent_result.demands else "chat"
        session.pending_clarification = PendingClarification(
            pending_intent=primary,
            missing_slots=intent_result.missing_entities or [],
            clarification_question=intent_result.clarification_question or "",
            expected_slot_types=intent_result.missing_entities or [],
            created_turn_id=turn_id,
            resolved_slots=dict(intent_result.resolved_entities or {}),
        )
        if intent_result.resolved_entities:
            if not hasattr(session, "global_slots"):
                session.global_slots = {}
            for k, v in intent_result.resolved_entities.items():
                if v is not None:
                    session.global_slots[k] = v
        return {
            "needs_clarification": True,
            "primary": primary,
            "question": intent_result.clarification_question,
        }
    else:
        # 正常执行后保存对话历史
        turn_id = len(session.working_memory.turns) + 1
        turn = DialogueTurn(
            turn_id=turn_id,
            user_message=message,
            assistant_reply="",
            intent=intent_result.demands[0].intent_type if intent_result.demands else "chat",
            rewritten_query=rw.rewritten_query,
            evidence_score=0.0,
        )
        session.working_memory.append(turn)
        # 保存 resolved_slots 到 global_slots
        if intent_result.resolved_entities:
            if not hasattr(session, "global_slots"):
                session.global_slots = {}
            for k, v in intent_result.resolved_entities.items():
                if v is not None:
                    session.global_slots[k] = v
        return {
            "needs_clarification": False,
            "primary": intent_result.demands[0].intent_type if intent_result.demands else "chat",
            "demands": [d.intent_type for d in intent_result.demands],
            "resolved_entities": intent_result.resolved_entities,
        }


async def test_case(group_name: str, turn1_msg: str, turn1_gold: str, turn2_msg: str, turn2_gold: str):
    print(f"\n{'='*60}")
    print(f"Test: {group_name}")
    print(f"{'='*60}")

    session = SessionMemory(session_id=f"test_{group_name}")
    session.global_slots = {"resume_available": True}

    # Turn 1
    print(f"\n[Turn 1] {turn1_msg}")
    r1 = await run_turn(session, turn1_msg)
    print(f"  result: needs_clarification={r1['needs_clarification']}, primary={r1['primary']}")
    if r1['needs_clarification']:
        print(f"  question: {r1['question']}")
    assert r1['needs_clarification'], f"Turn 1 应触发澄清"

    # Turn 2
    print(f"\n[Turn 2] {turn2_msg}")
    r2 = await run_turn(session, turn2_msg)
    print(f"  result: needs_clarification={r2['needs_clarification']}, primary={r2['primary']}, demands={r2.get('demands', [])}")
    print(f"  resolved_entities={r2.get('resolved_entities', {})}")

    success = (
        not r2['needs_clarification']
        and r2['primary'] == turn2_gold
    )
    print(f"\n[RESULT] {'PASS' if success else 'FAIL'} (expected primary={turn2_gold})")
    return success


async def main():
    results = []

    # eval_chen_13 → eval_chen_14
    r1 = await test_case(
        "chen_m2",
        "分析一下这个岗",
        "clarification",
        "蚂蚁集团AI产品经理",
        "attribute_verify",
    )
    results.append(r1)

    # eval_li_14 → eval_li_15
    r2 = await test_case(
        "li_m2",
        "分析这个Java岗",
        "clarification",
        "阿里巴巴后端开发",
        "attribute_verify",
    )
    results.append(r2)

    print(f"\n{'='*60}")
    print(f"Overall: {'ALL PASS' if all(results) else 'SOME FAILED'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
