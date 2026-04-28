#!/usr/bin/env python3
"""
多轮对话 + RAG 完整链路测试

测试目标：
1. 固定检索策略为混合召回+重排序（向量20 + BM25 20 → 混合 → 重排序 top-10）
2. 验证多轮对话记忆机制（工作内存、压缩记忆、长期记忆）
3. 在终端完整打印每一轮的 system prompt、user prompt、LLM 回复

注意：本脚本直接在终端运行，不依赖前端。
"""

import sys
import asyncio
import json
from pathlib import Path
from datetime import datetime

# Windows 控制台默认 GBK，强制设置 UTF-8
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BACKEND_ROOT = Path(__file__).parent.parent.resolve()
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.intent import IntentRouter
from app.core.agent import EnhancedAgentOrchestrator
from app.core.memory import SessionMemory
from app.core.llm_client import LLMClient
from app.core.db import delete_long_term_memory, delete_session_meta


def print_banner(title: str, width: int = 90):
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_section(title: str, content: str, max_lines: int = None):
    """打印带标题的区块内容"""
    print(f"\n{'─' * 60}")
    print(f"【{title}】")
    print(f"{'─' * 60}")
    lines = content.split("\n")
    if max_lines and len(lines) > max_lines:
        for line in lines[:max_lines]:
            print(line)
        print(f"... ({len(lines) - max_lines} 行省略)")
    else:
        for line in lines:
            print(line)


async def run_multi_turn_test():
    """执行多轮对话完整链路测试"""

    # ── 初始化 ──
    llm = LLMClient.from_chat_config()
    intent_router = IntentRouter(llm_client=llm)
    agent = EnhancedAgentOrchestrator()

    # 固定 session_id 方便追踪
    session_id = "demo_session_001"
    user_id = "demo_user_001"

    # 清理旧数据，确保测试环境干净
    delete_long_term_memory(user_id)
    delete_long_term_memory(session_id)
    delete_session_meta(session_id)

    session = SessionMemory(session_id=session_id)

    # 测试问题序列（覆盖展开型、扩展型、格式型追问）
    test_queries = [
        "百度AI产品经理有什么要求",
        "具体说说硬性要求",
        "那阿里呢",
        "用表格总结一下百度和阿里的对比",
    ]

    print_banner("多轮对话 + RAG 完整链路测试", width=90)
    print(f"Session ID: {session_id}")
    print(f"User ID: {user_id}")
    print(f"检索策略: 混合召回(70%向量+30%BM25) + CrossEncoder重排序")
    print(f"测试轮数: {len(test_queries)}")
    print(f"时间: {datetime.now().isoformat()}")

    for i, query in enumerate(test_queries, 1):
        print_banner(f"ROUND {i} / {len(test_queries)} | 用户: {query}", width=90)

        # ── Step 1: 意图识别 ──
        history_context = session.working_memory.get_recent_context(2) if session.working_memory.turns else ""
        route_meta = await intent_router.route(
            message=query,
            attachments=[],
            context={},
            history_context=history_context,
        )
        print(f"[意图识别] {route_meta.intent.value} | layer={route_meta.layer} | rule={route_meta.rule}")

        # ── Step 2: Agent 执行（含检索决策、工具链、记忆更新） ──
        agent_ctx, session = await agent.run(
            intent_result=route_meta,
            message=query,
            attachments=[],
            resume_text="",
            session=session,
        )

        print(f"[Agent] rewritten_query={agent_ctx.rewritten_query}")
        print(f"[Agent] tools={[t.name for t in agent_ctx.selected_tools]}")
        print(f"[Agent] kb_chunks={len(agent_ctx.kb_chunks)}")
        print(f"[记忆状态] working_turns={len(session.working_memory.turns)} | "
              f"compressed={len(session.compressed_memories)} | "
              f"evidence_cache={len(session.evidence_cache)}")

        # ── Step 3: 打印完整 System Prompt ──
        print_section("SYSTEM PROMPT（完整）", agent_ctx.system_prompt)

        # ── Step 4: 打印完整 User Prompt ──
        print_section("USER PROMPT（完整）", agent_ctx.user_prompt)

        # ── Step 5: LLM 生成回复 ──
        print("\n[LLM 生成中...]")
        reply_text = await llm.generate(
            prompt=agent_ctx.user_prompt,
            system=agent_ctx.system_prompt,
            temperature=0.7,
            max_tokens=1500,
        )

        # ── Step 6: 回填 assistant_reply ──
        if session.working_memory.turns:
            session.working_memory.turns[-1].assistant_reply = reply_text

        # ── Step 7: 打印回复 ──
        print_section("ASSISTANT REPLY（完整）", reply_text)

        # ── 可选：每轮结束后强制持久化 ──
        if session.working_memory.turns:
            await agent.memory_manager.rotate_memory(
                session, session.working_memory.turns[-1], persist=True
            )

    # ── 测试结束：打印长期记忆快照 ──
    print_banner("测试结束 | 长期记忆快照", width=90)
    if session.long_term:
        print(f"[长期记忆] user_id={session.long_term.user_id}")
        print(f"[实体] {json.dumps(session.long_term.entities, ensure_ascii=False, indent=2)}")
        print(f"[偏好] {json.dumps(session.long_term.preferences, ensure_ascii=False, indent=2)}")
    else:
        print("[长期记忆] 未生成")

    print_banner("所有轮次测试完成", width=90)


if __name__ == "__main__":
    asyncio.run(run_multi_turn_test())
