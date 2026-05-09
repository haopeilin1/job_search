#!/usr/bin/env python3
"""
测试 clarify 后意图恢复场景：
  eval_chen_13 ("分析一下这个岗") → 触发澄清
  eval_chen_14 ("蚂蚁集团AI产品经理") → 期望恢复为 ASSESS
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


async def test_chen_m2():
    session = SessionMemory(session_id="eval_chen_14")

    # ── 模拟 eval_chen_13 的执行结果 ──
    # 上轮用户说"分析一下这个岗"，系统触发澄清
    turn13 = DialogueTurn(
        turn_id=1,
        user_message="分析一下这个岗",
        assistant_reply="您想了解哪个公司的什么信息？请提供公司名称。",
        intent="chat",
        rewritten_query="分析一下这个岗",
        evidence_score=0.0,
    )
    session.working_memory.append(turn13)
    session.pending_clarification = PendingClarification(
        pending_intent="chat",
        missing_slots=["company", "position"],
        clarification_question="您想了解哪个公司的什么信息？请提供公司名称。",
        expected_slot_types=["company", "position"],
        created_turn_id=1,
        resolved_slots={},
    )
    # 设置 global_slots：resume_available 供 _check_clarification_need 使用
    session.global_slots = {"resume_available": True}

    # ── 本轮 eval_chen_14 ──
    message = "蚂蚁集团AI产品经理"
    rewriter = QueryRewriter()
    rw = await rewriter.rewrite(raw_query=message, session=session)
    print(f"[QueryRewrite] rewritten='{rw.rewritten_query}' follow_up_type={rw.follow_up_type}")

    router = LLMIntentRouter()
    multi_result = await router.route_multi(
        rewrite_result=rw,
        session=session,
        attachments=[],
        raw_message=message,
    )
    intent_result = multi_intent_result_to_intent_result(multi_result)

    print(f"[Intent] primary={multi_result.primary_intent.value if multi_result.primary_intent else 'None'}")
    print(f"[Intent] candidates={[c.intent_type.value for c in multi_result.candidates]}")
    print(f"[Intent] needs_clarification={multi_result.needs_clarification}")
    print(f"[Intent] demands={[d.intent_type for d in intent_result.demands]}")
    print(f"[Intent] resolved_entities={intent_result.resolved_entities}")
    print(f"[Intent] missing_entities={intent_result.missing_entities}")
    print(f"[Intent] clarification_question={intent_result.clarification_question}")

    # 期望：ASSESS，needs_clarification=False，company=蚂蚁集团，position=AI产品经理
    success = (
        multi_result.primary_intent.value == "verify"
        and not multi_result.needs_clarification
    )
    print(f"\n[RESULT] {'PASS' if success else 'FAIL'}")
    return success


async def test_li_m2():
    session = SessionMemory(session_id="eval_li_15")

    turn14 = DialogueTurn(
        turn_id=1,
        user_message="分析这个Java岗",
        assistant_reply="您想了解哪个公司的什么信息？请提供公司名称。",
        intent="chat",
        rewritten_query="分析这个Java岗",
        evidence_score=0.0,
    )
    session.working_memory.append(turn14)
    session.pending_clarification = PendingClarification(
        pending_intent="chat",
        missing_slots=["company", "position"],
        clarification_question="您想了解哪个公司的什么信息？请提供公司名称。",
        expected_slot_types=["company", "position"],
        created_turn_id=1,
        resolved_slots={},
    )
    session.global_slots = {"resume_available": True}

    message = "阿里巴巴后端开发"
    rewriter = QueryRewriter()
    rw = await rewriter.rewrite(raw_query=message, session=session)
    print(f"[QueryRewrite] rewritten='{rw.rewritten_query}' follow_up_type={rw.follow_up_type}")

    router = LLMIntentRouter()
    multi_result = await router.route_multi(
        rewrite_result=rw,
        session=session,
        attachments=[],
        raw_message=message,
    )
    intent_result = multi_intent_result_to_intent_result(multi_result)

    print(f"[Intent] primary={multi_result.primary_intent.value if multi_result.primary_intent else 'None'}")
    print(f"[Intent] candidates={[c.intent_type.value for c in multi_result.candidates]}")
    print(f"[Intent] needs_clarification={multi_result.needs_clarification}")
    print(f"[Intent] demands={[d.intent_type for d in intent_result.demands]}")
    print(f"[Intent] resolved_entities={intent_result.resolved_entities}")

    success = (
        multi_result.primary_intent.value == "verify"
        and not multi_result.needs_clarification
    )
    print(f"\n[RESULT] {'PASS' if success else 'FAIL'}")
    return success


async def main():
    print("=" * 60)
    print("Test 1: eval_chen_14 (蚂蚁集团AI产品经理)")
    print("=" * 60)
    r1 = await test_chen_m2()

    print("\n" + "=" * 60)
    print("Test 2: eval_li_15 (阿里巴巴后端开发)")
    print("=" * 60)
    r2 = await test_li_m2()

    print("\n" + "=" * 60)
    print(f"Overall: {'ALL PASS' if r1 and r2 else 'SOME FAILED'}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
