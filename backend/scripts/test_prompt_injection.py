#!/usr/bin/env python3
"""
验证：三层记忆是否真正注入 LLM prompt
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from app.core.intent import IntentRouter
from app.core.agent import EnhancedAgentOrchestrator
from app.core.memory import SessionMemory
from app.core.llm_client import LLMClient


async def test():
    llm = LLMClient.from_chat_config()
    intent_router = IntentRouter(llm_client=llm)
    agent = EnhancedAgentOrchestrator()

    session = SessionMemory(session_id="test_prompt_001")
    queries = [
        "百度有什么要求",
        "具体说说硬性要求",
        "阿里的呢",
    ]

    for i, query in enumerate(queries, 1):
        history_context = session.working_memory.get_recent_context(2) if session.working_memory.turns else ""
        route_meta = await intent_router.route(
            message=query, attachments=[], context={},
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

        print(f"\n{'='*60}")
        print(f"Round {i} | User: {query}")
        print(f"{'='*60}")
        print(f"[Prompt 结构分析]")

        user_prompt = ctx.user_prompt

        checks = {
            "【最近对话历史】": "【最近对话历史】" in user_prompt,
            "【更早对话摘要】": "【更早对话摘要】" in user_prompt,
            "【用户问题】": "【用户问题】" in user_prompt,
            "【改写后的问题】": "【改写后的问题】" in user_prompt,
            "【检索策略】": "【检索策略】" in user_prompt,
            "【本轮检索证据】/【复用历史证据】": "【本轮检索证据】" in user_prompt or "【复用历史证据】" in user_prompt,
            "【用户画像】": "【用户画像】" in user_prompt,
        }

        for name, ok in checks.items():
            status = "YES" if ok else "NO"
            print(f"  {status} {name}")

        # 打印最近对话历史的实际内容
        if "【最近对话历史】" in user_prompt:
            start = user_prompt.find("【最近对话历史】")
            end = user_prompt.find("【", start + 1)
            snippet = user_prompt[start:end if end > start else start + 200]
            print(f"\n  [最近对话历史内容]\n  {snippet[:300].replace(chr(10), ' ')}")

    print(f"\n{'='*60}")
    print("Test completed")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(test())
