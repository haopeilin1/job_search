#!/usr/bin/env python3
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).parent
sys.path.insert(0, str(EVAL_DIR.parent))

from app.core.memory import SessionMemory
from app.core.llm_intent import LLMFallbackClassifier, IntentCandidate, LLMIntentType

session = SessionMemory(session_id="test")
session.global_slots = {"resume_available": True}

candidates = [
    IntentCandidate(
        intent_type=LLMIntentType.ASSESS,
        confidence=0.82,
        reason="test",
        slots={"company": "蚂蚁集团", "position": "AI产品经理"},
    )
]

fallback = LLMFallbackClassifier()
global_slots = fallback._merge_global_slots(candidates, session=session)
print("global_slots:", global_slots)

needs_clarify, reason = fallback._check_clarification_need(candidates, global_slots)
print("needs_clarify:", needs_clarify)
print("reason:", reason)
