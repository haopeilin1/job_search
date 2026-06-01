"""测试 clarify 场景下 eval_chen_14 的意图识别修复"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.llm_intent import (
    LLMIntentRouter, LLMIntentType,
    LLMRuleRegistry, LLMRuleResult, RuleStrength
)
from app.core.memory import SessionMemory, DialogueTurn
from app.core.query_rewrite import QueryRewriteResult


async def test_chen_m2_scenario():
    """模拟 chen_m2 多轮场景:
    Turn 1: "分析一下这个岗" → 期望 clarification
    Turn 2: "蚂蚁集团AI产品经理" → 期望 verify (修复后)
    """
    router = LLMIntentRouter()
    session = SessionMemory(session_id="test_chen_m2")

    # ========== Turn 1: "分析一下这个岗" ==========
    print("=" * 60)
    print("Turn 1: 分析一下这个岗")
    print("=" * 60)

    rewrite1 = QueryRewriteResult(
        rewritten_query="分析一下这个岗",
        follow_up_type="none",
        original_message="分析一下这个岗",
    )

    result1 = await router.route_multi(
        rewrite_result=rewrite1,
        session=session,
        attachments=[],
        raw_message="分析一下这个岗",
    )

    primary1 = result1.primary_intent
    cand1 = result1.candidates[0] if result1.candidates else None
    print(f"  识别意图: {primary1.value if primary1 else 'None'}")
    print(f"  needs_clarification: {result1.needs_clarification}")
    print(f"  candidates: {len(result1.candidates)}")
    if cand1:
        print(f"  candidate[0] intent: {cand1.intent_type.value}")
        print(f"  candidate[0] slots: {json.dumps(cand1.slots, ensure_ascii=False, indent=2)}")
        print(f"  candidate[0] needs_clarification: {cand1.needs_clarification}")

    # 手动模拟系统回复澄清
    print("\n  [INFO] 注入 Turn 1 澄清状态")
    session.working_memory.append(DialogueTurn(
        turn_id=1,
        user_message="分析一下这个岗",
        assistant_reply=cand1.clarification_question if cand1 else "您想了解哪个公司的什么信息？请提供公司名称。",
        intent="clarification",
        rewritten_query="分析一下这个岗",
    ))
    # 注入 pending_clarification（模拟 chat.py 中的行为）
    from app.core.memory import PendingClarification as RealPendingClarification
    session.pending_clarification = RealPendingClarification(
        pending_intent="verify",
        missing_slots=["company", "position"],
        clarification_question=cand1.clarification_question if cand1 else "您想了解哪个公司的什么信息？",
        resolved_slots={},
        created_turn_id=1,
    )

    # ========== Turn 2: "蚂蚁集团AI产品经理" ==========
    print("\n" + "=" * 60)
    print("Turn 2: 蚂蚁集团AI产品经理")
    print("=" * 60)

    rewrite2 = QueryRewriteResult(
        rewritten_query="蚂蚁集团AI产品经理",
        follow_up_type="clarify",
        original_message="蚂蚁集团AI产品经理",
    )

    # 调试：检查 clarify 后处理条件
    print(f"\n  [DEBUG] pending_clarification: {session.pending_clarification}")
    if session.pending_clarification:
        pc = session.pending_clarification
        print(f"    pending_intent: {pc.pending_intent}")
        print(f"    missing_slots: {pc.missing_slots}")
        print(f"    resolved_slots: {pc.resolved_slots}")
        is_clarify = bool(pc.missing_slots) or not bool(pc.resolved_slots)
        print(f"    is_clarify_scenario: {is_clarify}")
        _OLD_INTENT_MAP = {
            "match_assess": "assess", "attribute_verify": "verify",
            "interview_prepare": "prepare", "global_explore": "explore",
            "file_manage": "manage", "general_chat": "chat",
        }
        mapped = _OLD_INTENT_MAP.get(pc.pending_intent, pc.pending_intent)
        print(f"    mapped_pending: {mapped}")
        print(f"    in_list: {mapped in ('chat', 'assess', 'verify')}")
        if session.working_memory.turns:
            print(f"    last_user_msg: {session.working_memory.turns[-1].user_message}")

    result2 = await router.route_multi(
        rewrite_result=rewrite2,
        session=session,
        attachments=[],
        raw_message="蚂蚁集团AI产品经理",
    )

    primary2 = result2.primary_intent
    cand2 = result2.candidates[0] if result2.candidates else None
    print(f"  识别意图: {primary2.value if primary2 else 'None'}")
    print(f"  needs_clarification: {result2.needs_clarification}")
    print(f"  candidates: {len(result2.candidates)}")
    if cand2:
        print(f"  candidate[0] intent: {cand2.intent_type.value}")
        print(f"  candidate[0] slots: {json.dumps(cand2.slots, ensure_ascii=False, indent=2)}")
        print(f"  candidate[0] needs_clarification: {cand2.needs_clarification}")

    # 验证
    print("\n" + "=" * 60)
    print("验证结果")
    print("=" * 60)

    expected_intent = LLMIntentType.VERIFY
    expected_attrs = ["综合情况"]
    actual_intent = cand2.intent_type if cand2 else None
    actual_attrs = cand2.slots.get("attributes", []) if cand2 else []

    intent_ok = actual_intent == expected_intent
    attrs_ok = actual_attrs == expected_attrs

    print(f"  意图判断: {'PASS' if intent_ok else 'FAIL'} (期望={expected_intent.value}, 实际={actual_intent.value if actual_intent else 'None'})")
    print(f"  属性槽位: {'PASS' if attrs_ok else 'FAIL'} (期望={expected_attrs}, 实际={actual_attrs})")

    if intent_ok and attrs_ok:
        print("\n  PASS 修复验证通过！eval_chen_14 将正确映射为 VERIFY + qa_synthesize")
    else:
        print("\n  FAIL 修复未生效，仍需进一步排查")

    return intent_ok and attrs_ok


if __name__ == "__main__":
    ok = asyncio.run(test_chen_m2_scenario())
    sys.exit(0 if ok else 1)
