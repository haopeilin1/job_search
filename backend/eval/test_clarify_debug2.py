#!/usr/bin/env python3
"""调试 eval_li_15 的 clarify 场景"""
import asyncio
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).parent
sys.path.insert(0, str(EVAL_DIR.parent))

from app.core.memory import SessionMemory, DialogueTurn, PendingClarification
from app.core.query_rewrite import QueryRewriteResult
from app.core.llm_intent import LLMIntentRouter, LLMRuleResult, LLMIntentType, RuleStrength

async def debug_li_m2():
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

    message = "阿里巴巴后端开发"
    rewriter = LLMIntentRouter()
    
    # 先模拟 QueryRewrite
    rewrite_result = QueryRewriteResult(
        rewritten_query=message,
        search_keywords=message,
        resolved_references={},
        is_follow_up=False,
        follow_up_type="none",
        original_message=message,
    )

    router = LLMIntentRouter()
    multi_result = await router.route_multi(
        rewrite_result=rewrite_result,
        session=session,
        attachments=[],
        raw_message=message,
    )

    print(f"primary={multi_result.primary_intent.value if multi_result.primary_intent else 'None'}")
    print(f"candidates={[c.intent_type.value for c in multi_result.candidates]}")
    print(f"needs_clarification={multi_result.needs_clarification}")
    for c in multi_result.candidates:
        print(f"  candidate: intent={c.intent_type.value}, conf={c.confidence}, reason={c.reason}")
        print(f"    slots={c.slots}")

if __name__ == "__main__":
    asyncio.run(debug_li_m2())
