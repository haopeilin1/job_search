#!/usr/bin/env python3
"""
验证4个bug修复的测试脚本

测试序列：
1. "百度AI产品经理有什么要求"
2. "我符合它的要求吗"          → 应检查简历，提示未上传
3. "具体说说硬性要求"           → 展开型追问，REUSE
4. "那阿里呢"                  → 扩展型追问，FULL
5. "用表格对比一下"             → 格式型追问，NO_RETRIEVAL
6. "我的要求符合它吗"           → 验证长期记忆不混入检索结果
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
    session = SessionMemory(session_id="test_fixes_001")

    queries = [
        "百度AI产品经理有什么要求",
        "我符合它的要求吗",
        "具体说说硬性要求",
        "那阿里呢",
        "用表格对比一下",
        "我的要求符合它吗",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"  ROUND {i} | {query}")
        print(f"{'='*80}")

        history = session.working_memory.get_recent_context(2) if session.working_memory.turns else ""
        route_meta = await intent_router.route(
            message=query, attachments=[], context={},
            history_context=history,
        )
        print(f"[Intent] {route_meta.intent.value}")

        agent_ctx, session = await agent.run(
            intent_result=route_meta,
            message=query,
            attachments=[],
            resume_text="",
            session=session,
        )

        print(f"[Tools] {[t.name for t in agent_ctx.selected_tools]}")
        print(f"[Memory] working={len(session.working_memory.turns)}, compressed={len(session.compressed_memories)}")

        # 检查 user prompt 中的历史是否排除了当前轮
        user_prompt = agent_ctx.user_prompt
        history_section = ""
        if "【最近对话历史】" in user_prompt:
            start = user_prompt.find("【最近对话历史】")
            end = user_prompt.find("【", start + 1)
            history_section = user_prompt[start:end if end > start else start + 500]

        # 检查当前轮是否出现在历史中
        current_in_history = query in history_section
        print(f"[Check] 当前轮出现在历史中: {current_in_history} {'FAIL!' if current_in_history else 'OK'}")

        # 检查长期记忆
        if session.long_term:
            print(f"[LongTerm] entities={list(session.long_term.entities.keys())}")
            print(f"[LongTerm] companies={session.long_term.entities.get('公司', [])}")

        # 模拟 LLM 回复并回填
        if session.working_memory.turns:
            session.working_memory.turns[-1].assistant_reply = f"[reply{i}]"
            await agent.memory_manager.rotate_memory(
                session, session.working_memory.turns[-1], persist=True
            )

    print(f"\n{'='*80}")
    print("  Test completed")
    print(f"{'='*80}")


if __name__ == "__main__":
    asyncio.run(test())
