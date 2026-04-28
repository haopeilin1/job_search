#!/usr/bin/env python3
"""
交互式多轮对话测试终端

用法:
    cd backend
    python scripts/interactive_chat.py

交互命令:
    /quit 或 /exit      退出对话
    /history            查看当前工作内存中的对话历史
    /memory             查看长期记忆快照
    /clear              清空当前会话
    /prompt             仅打印 prompt 不调用 LLM（节省 token）
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


def print_banner(title: str, char: str = "=", width: int = 90):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def print_section(title: str, content: str):
    print(f"\n{'─' * 60}")
    print(f"【{title}】")
    print(f"{'─' * 60}")
    print(content)


class InteractiveChat:
    def __init__(self):
        self.llm = LLMClient.from_chat_config()
        self.intent_router = IntentRouter(llm_client=self.llm)
        self.agent = EnhancedAgentOrchestrator()
        self.session = SessionMemory(session_id=f"interactive_{int(datetime.now().timestamp())}")
        self.round = 0
        self.prompt_only_mode = False

    async def handle_command(self, cmd: str) -> bool:
        """处理斜杠命令，返回 True 表示继续对话，False 表示退出"""
        cmd = cmd.strip().lower()

        if cmd in ("/quit", "/exit", "q", "quit", "exit"):
            print_banner("再见！", char="-")
            return False

        if cmd == "/history":
            print_banner("工作内存（最近3轮）")
            print(self.session.working_memory.get_recent_context(3))
            return True

        if cmd == "/memory":
            print_banner("长期记忆快照")
            if self.session.long_term:
                print(f"user_id: {self.session.long_term.user_id}")
                print(f"entities: {json.dumps(self.session.long_term.entities, ensure_ascii=False, indent=2)}")
                print(f"preferences: {json.dumps(self.session.long_term.preferences, ensure_ascii=False, indent=2)}")
            else:
                print("长期记忆为空")
            return True

        if cmd == "/clear":
            self.session = SessionMemory(session_id=f"interactive_{int(datetime.now().timestamp())}")
            self.round = 0
            print("[系统] 会话已清空")
            return True

        if cmd == "/prompt":
            self.prompt_only_mode = not self.prompt_only_mode
            mode = "开启" if self.prompt_only_mode else "关闭"
            print(f"[系统] Prompt-only 模式已{mode}（下一轮生效）")
            return True

        if cmd == "/help":
            print("可用命令：/quit, /history, /memory, /clear, /prompt, /help")
            return True

        return None  # 不是命令，是普通消息

    async def chat_round(self, message: str):
        """执行一轮对话"""
        self.round += 1
        print_banner(f"ROUND {self.round} | 用户: {message}")

        # ── 意图识别 ──
        history_context = self.session.working_memory.get_recent_context(2) if self.session.working_memory.turns else ""
        route_meta = await self.intent_router.route(
            message=message,
            attachments=[],
            context={},
            history_context=history_context,
        )
        print(f"[意图识别] {route_meta.intent.value} | layer={route_meta.layer} | rule={route_meta.rule}")

        # ── Agent 执行 ──
        agent_ctx, self.session = await self.agent.run(
            intent_result=route_meta,
            message=message,
            attachments=[],
            resume_text="",
            session=self.session,
        )

        print(f"[Agent] rewritten_query={agent_ctx.rewritten_query}")
        print(f"[Agent] tools={[t.name for t in agent_ctx.selected_tools]}")
        print(f"[Agent] kb_chunks={len(agent_ctx.kb_chunks)}")
        print(f"[记忆状态] working_turns={len(self.session.working_memory.turns)} | "
              f"compressed={len(self.session.compressed_memories)} | "
              f"evidence_cache={len(self.session.evidence_cache)}")

        # ── 打印完整 Prompt ──
        print_section("SYSTEM PROMPT（完整）", agent_ctx.system_prompt)
        print_section("USER PROMPT（完整）", agent_ctx.user_prompt)

        # ── Prompt-only 模式跳过 LLM 调用 ──
        if self.prompt_only_mode:
            print("\n[系统] Prompt-only 模式，跳过 LLM 调用")
            reply_text = "[Prompt-only 模式，无回复]"
        else:
            print("\n[LLM 生成中...]")
            reply_text = await self.llm.generate(
                prompt=agent_ctx.user_prompt,
                system=agent_ctx.system_prompt,
                temperature=0.7,
                max_tokens=1500,
            )

        # ── 回填 assistant_reply ──
        if self.session.working_memory.turns:
            self.session.working_memory.turns[-1].assistant_reply = reply_text

        # ── 打印回复 ──
        if not self.prompt_only_mode:
            print_section("ASSISTANT REPLY（完整）", reply_text)

        # ── 持久化 ──
        if self.session.working_memory.turns:
            await self.agent.memory_manager.rotate_memory(
                self.session, self.session.working_memory.turns[-1], persist=True
            )

    async def run(self):
        print_banner("交互式多轮对话测试终端", width=90)
        print("检索策略: 混合召回(70%向量+30%BM25) + CrossEncoder重排序")
        print("命令: /quit 退出, /history 查看历史, /memory 查看画像, /clear 清空, /prompt 仅打印prompt")
        print("输入你的问题开始对话...\n")

        while True:
            try:
                user_input = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break

            if not user_input:
                continue

            result = await self.handle_command(user_input)
            if result is False:
                break
            if result is True:
                continue

            await self.chat_round(user_input)


if __name__ == "__main__":
    asyncio.run(InteractiveChat().run())
