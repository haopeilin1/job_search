#!/usr/bin/env python3
"""
多轮对话 + 记忆机制 端到端测试
"""

import sys
import asyncio
from pathlib import Path

# Windows 控制台默认 GBK，强制设置 UTF-8
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BACKEND_ROOT = Path(__file__).parent.parent.resolve()
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.intent import IntentRouter
from app.core.agent import EnhancedAgentOrchestrator
from app.core.llm_client import LLMClient


async def test():
    llm = LLMClient.from_chat_config()
    intent_router = IntentRouter(llm_client=llm)
    agent = EnhancedAgentOrchestrator()
    session = None

    test_queries = [
        "百度有什么要求",
        "具体说说硬性要求",
        "阿里的呢",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"  Round {i} | User: {query}")
        print(f"{'='*60}")

        history_context = session.working_memory.get_recent_context(2) if session and session.working_memory.turns else ""
        route_meta = await intent_router.route(
            message=query,
            attachments=[],
            context={},
            history_context=history_context,
        )
        print(f"[Intent] {route_meta.intent.value} | rule={route_meta.rule}")

        ctx, session = await agent.run(
            intent_result=route_meta,
            message=query,
            attachments=[],
            resume_text="",
            session=session,
        )

        print(f"[Memory] working_turns={len(session.working_memory.turns)}, "
              f"compressed={len(session.compressed_memories)}, "
              f"evidence_cache={len(session.evidence_cache)}")

        for t, r in zip(ctx.selected_tools, ctx.tool_results):
            status = "OK" if r.success else "FAIL"
            print(f"[Tool] {t.name} -> {status}")

        print(f"[Chunks] kb_chunks={len(ctx.kb_chunks)}")

        if hasattr(ctx, "system_prompt"):
            print(f"[System] {ctx.system_prompt[:100]}...")
        if hasattr(ctx, "user_prompt"):
            print(f"[UserPrompt] len={len(ctx.user_prompt)}")

    print(f"\n{'='*60}")
    print("  Test completed")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(test())
