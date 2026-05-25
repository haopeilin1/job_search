#!/usr/bin/env python3
"""验证澄清文案修复是否生效"""
import asyncio
import sys
sys.path.insert(0, '.')

from app.core.llm_intent import MultiIntentResult, IntentCandidate, LLMIntentType
from app.core.new_arch_adapter import multi_intent_result_to_intent_result


def test_clarify_question():
    """测试不同意图组合生成的澄清问题"""
    test_cases = [
        # (意图列表, 期望关键词)
        ([(LLMIntentType.VERIFY, 0.5), (LLMIntentType.EXPLORE, 0.4)], "核实"),
        ([(LLMIntentType.ASSESS, 0.5)], "匹配度"),
        ([(LLMIntentType.VERIFY, 0.5)], "哪方面信息"),
        ([(LLMIntentType.EXPLORE, 0.5)], "具体岗位"),
        ([(LLMIntentType.PREPARE, 0.5)], "面试"),
        ([(LLMIntentType.CHAT, 0.5)], "详细说说"),
    ]

    print("=" * 60)
    print("澄清问题生成测试")
    print("=" * 60)

    all_passed = True
    for intents, expected_keyword in test_cases:
        candidates = [
            IntentCandidate(intent_type=it, confidence=conf, reason='test', slots={})
            for it, conf in intents
        ]
        multi = MultiIntentResult(
            candidates=candidates,
            primary_intent=intents[0][0],
            needs_clarification=True,
            clarification_reason='多意图置信度均偏低，需要用户确认具体需求',
        )
        result = multi_intent_result_to_intent_result(multi)
        question = result.clarification_question
        passed = expected_keyword in question
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {intents[0][0].value} -> '{question}'")
        if not passed:
            print(f"       Expected keyword: '{expected_keyword}'")
            all_passed = False

    print()
    if all_passed:
        print("All tests passed! The fix is active.")
    else:
        print("Some tests failed!")
    return all_passed


if __name__ == "__main__":
    test_clarify_question()
