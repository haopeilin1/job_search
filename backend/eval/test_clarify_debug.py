#!/usr/bin/env python3
"""调试校准器在 clarify 场景下的行为"""
import asyncio
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).parent
sys.path.insert(0, str(EVAL_DIR.parent))

from app.core.memory import SessionMemory, DialogueTurn, PendingClarification
from app.core.query_rewrite import QueryRewriteResult
from app.core.llm_intent import (
    LLMIntentRouter, SmallModelCalibrator, LLMRuleResult, 
    LLMIntentType, RuleStrength, INTENT_CALIBRATION_SYSTEM, INTENT_CALIBRATION_EXAMPLES
)

async def debug_calibrator():
    session = SessionMemory(session_id="eval_chen_14")

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

    rewrite_result = QueryRewriteResult(
        rewritten_query="蚂蚁集团AI产品经理",
        search_keywords="蚂蚁集团AI产品经理",
        resolved_references={},
        is_follow_up=False,
        follow_up_type="none",
        original_message="蚂蚁集团AI产品经理",
    )

    # 模拟虚拟规则
    virtual_rule = LLMRuleResult(
        intent=LLMIntentType.CHAT,
        strength=RuleStrength.WEAK,
        rule_name="clarify_virtual",
        metadata={},
        trigger="",
    )

    calibrator = SmallModelCalibrator()
    
    # 构建 prompt
    working_history, compressed_history, long_term_profile = calibrator._build_memory_context(session)
    system_prompt = f"{INTENT_CALIBRATION_SYSTEM}\n\n{INTENT_CALIBRATION_EXAMPLES}"
    system_prompt = system_prompt.format(
        rule_intent="chat",
        rule_strength="WEAK",
        rule_name="clarify_virtual",
        rewritten_query=rewrite_result.rewritten_query,
        follow_up_type=rewrite_result.follow_up_type,
        working_history=working_history,
        compressed_history=compressed_history,
        long_term_profile=long_term_profile,
    )

    print("=" * 60)
    print("SYSTEM PROMPT")
    print("=" * 60)
    print(system_prompt[:3000])
    print("...")
    print(system_prompt[-1500:])

    print("\n" + "=" * 60)
    print("WORKING HISTORY")
    print("=" * 60)
    print(working_history)

    print("\n" + "=" * 60)
    print("CALLING CALIBRATOR")
    print("=" * 60)
    candidate = await calibrator._calibrate_single_intent(
        rule_result=virtual_rule,
        rewrite_result=rewrite_result,
        session=session,
    )
    print(f"intent={candidate.intent_type.value}")
    print(f"confidence={candidate.confidence}")
    print(f"reason={candidate.reason}")
    print(f"slots={candidate.slots}")
    print(f"missing_slots={candidate.missing_slots}")
    print(f"needs_clarification={candidate.needs_clarification}")


if __name__ == "__main__":
    asyncio.run(debug_calibrator())
