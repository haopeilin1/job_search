#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).parent
sys.path.insert(0, str(EVAL_DIR.parent))

from app.core.memory import SessionMemory, DialogueTurn, PendingClarification
from app.core.query_rewrite import QueryRewriteResult
from app.core.llm_intent import LLMIntentRouter

async def debug():
    session = SessionMemory(session_id="eval_chen_14")
    session.global_slots = {"resume_available": True}

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

    router = LLMIntentRouter()
    multi_result = await router.route_multi(
        rewrite_result=rewrite_result,
        session=session,
        attachments=[],
        raw_message="蚂蚁集团AI产品经理",
    )

    print(f"primary={multi_result.primary_intent.value if multi_result.primary_intent else 'None'}")
    print(f"candidates={[c.intent_type.value for c in multi_result.candidates]}")
    print(f"needs_clarification={multi_result.needs_clarification}")
    print(f"global_slots={multi_result.global_slots}")
    print(f"clarification_reason={multi_result.clarification_reason}")

asyncio.run(debug())
