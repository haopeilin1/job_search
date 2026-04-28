#!/usr/bin/env python3
"""
长期记忆持久化端到端测试

验证点：
1. 多轮对话后，长期记忆按 user_id 落库
2. 新 session 传入相同 user_id，能恢复长期记忆
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from app.core.intent import IntentRouter
from app.core.agent import EnhancedAgentOrchestrator
from app.core.memory import SessionMemory, MemoryManager
from app.core.llm_client import LLMClient
from app.core.db import load_long_term_memory, load_session_meta, delete_long_term_memory


async def test():
    llm = LLMClient.from_chat_config()
    intent_router = IntentRouter(llm_client=llm)
    agent = EnhancedAgentOrchestrator()

    user_id = "test_user_002"
    session_id_1 = "session_A"

    # 清理旧数据
    delete_long_term_memory(user_id)
    delete_long_term_memory(session_id_1)

    session = SessionMemory(session_id=session_id_1)

    test_queries = [
        "我擅长Python和RAG，想在北京找AI产品岗位",
        "百度有什么要求",
        "阿里的呢",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n[Round {i}] User: {query}")

        history_context = session.working_memory.get_recent_context(2) if session.working_memory.turns else ""
        route_meta = await intent_router.route(
            message=query,
            attachments=[],
            context={},
            history_context=history_context,
        )

        ctx, session = await agent.run(
            intent_result=route_meta,
            message=query,
            attachments=[],
            resume_text="",
            session=session,
        )

        if session.working_memory.turns:
            session.working_memory.turns[-1].assistant_reply = f"[reply{i}]"

        print(f"  -> turns={len(session.working_memory.turns)}, cache={len(session.evidence_cache)}")

    # 手动绑定 user_id 并持久化
    if not session.long_term:
        session.long_term = MemoryManager.load_long_term(user_id) or type('obj', (object,), {
            'user_id': user_id, 'entities': {}, 'preferences': {},
            'resume_fingerprint': '', 'topic_flags': {}, 'last_updated': 0
        })()
    session.long_term.user_id = user_id
    await agent.memory_manager.rotate_memory(session, session.working_memory.turns[-1], persist=True)

    print(f"\n[Save] LongTermMemory saved with user_id={user_id}")

    # 模拟新 session 传入相同 user_id
    print("\n" + "=" * 50)
    print("New session with same user_id -> recover LongTermMemory")
    print("=" * 50)

    lt = load_long_term_memory(user_id)
    if lt:
        print(f"[DB] user_id={lt.user_id}")
        print(f"[DB] entities={lt.entities}")
        print(f"[DB] preferences={lt.preferences}")
        print("[PASS] LongTermMemory persist OK")
    else:
        print("[FAIL] LongTermMemory not found in DB")

    print("\n[Test] Completed")


if __name__ == "__main__":
    asyncio.run(test())
