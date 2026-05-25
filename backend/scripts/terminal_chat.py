#!/usr/bin/env python3
"""
终端对话脚本 -- 直接在终端与求职雷达 AI 助手交互

用法:
    cd backend && source venv/Scripts/activate && python scripts/terminal_chat.py

功能:
    - 完整的 LLM 路线(Query改写 -> 意图识别 -> Plan -> ReAct执行 -> LLM聚合)
    - 完整的记忆系统(SessionMemory + 长期记忆 + 证据缓存)
    - 过程信息实时打印 + 详细日志写入文件

快捷键:
    - Ctrl+C / 输入 exit / quit 退出
    - 输入 /clear 清空当前会话记忆
    - 输入 /debug 切换详细模式
    - 输入 /memory 查看当前记忆状态
"""

import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Windows 终端强制 UTF-8
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except Exception:
        pass

# 把 backend 目录加入路径
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

os.chdir(BACKEND_DIR)

from app.core.config import settings
from app.core.llm_client import LLMClient
from app.core.memory import SessionMemory, MemoryManager, DialogueTurn
from app.core.query_rewrite import QueryRewriter
from app.core.db import load_session_meta, load_long_term_memory
from app.services.handlers import ChatRequest
from app.services.handlers import _get_resume_text
from app.routers.chat import (
    _get_or_create_session,
    _handle_llm_route_v2,
    _handle_rule_route,
    _build_memory_state,
)

# ═══════════════════════════════════════════════════════
# 1. 日志配置
# ═══════════════════════════════════════════════════════

LOG_DIR = BACKEND_DIR / "logs" / "terminal_chat"
LOG_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"terminal_chat_{TIMESTAMP}.jsonl"

# 同时配置标准日志
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("terminal_chat")


# ═══════════════════════════════════════════════════════
# 2. 终端输出工具
# ═══════════════════════════════════════════════════════

def _print_banner():
    print("""
==========================================
       求职雷达 -- 终端对话模式
==========================================
  命令: /clear 清空记忆  /debug 切换详细模式
        /memory 查看记忆  exit/quit 退出
==========================================
""")


import re

_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+"
)

def _safe_print(text: str) -> str:
    """移除/替换终端可能无法显示的字符（GBK兼容 + surrogate清理）"""
    if not text:
        return text
    # 先清理非法 surrogate 字符
    text = text.encode("utf-8", "surrogatepass").decode("utf-8", "replace")
    replacements = {
        "✅": "[OK]",
        "❌": "[FAIL]",
        "⏳": "...",
        "─": "-",
        "╔": "+",
        "╗": "+",
        "╠": "+",
        "╣": "+",
        "╚": "+",
        "╝": "+",
        "║": "|",
        "━": "=",
        "┃": "|",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = _EMOJI_PATTERN.sub("", text)
    return text


def _print_step(label: str, content: str):
    print(_safe_print(f"  [{label}] {content}"))


def _print_reply(text: str):
    print(_safe_print(f"\n[AI回复]\n{text}\n"))


def _print_error(text: str):
    print(_safe_print(f"[错误] {text}\n"))


def _print_divider():
    print("-" * 60)


# ═══════════════════════════════════════════════════════
# 3. 核心对话循环
# ═══════════════════════════════════════════════════════

async def process_message(
    message: str,
    session: SessionMemory,
    session_id: str,
    debug_mode: bool,
) -> dict:
    """处理单条消息，返回完整响应"""

    resume_text = _get_resume_text()

    # -- Query 改写 --
    t0 = time.time()
    rewriter = QueryRewriter()
    rewrite_result = await rewriter.rewrite(raw_query=message, session=session)
    rewrite_latency = (time.time() - t0) * 1000

    # -- 构建 ChatRequest --
    request = ChatRequest(
        message=message,
        session_id=session_id,
        type="text",
    )

    # -- 路线选择 --
    agent_mode = settings.AGENT_MODE
    if agent_mode == "auto":
        effective_mode = settings.DEFAULT_AGENT_MODE
    else:
        effective_mode = agent_mode
    use_llm_agent = effective_mode == "llm"

    # -- 执行 --
    t1 = time.time()
    if use_llm_agent:
        response = await _handle_llm_route_v2(
            request=request,
            session=session,
            session_id=session_id,
            message_with_ocr=message,
            rewrite_result=rewrite_result,
            resume_text=resume_text,
        )
    else:
        response = await _handle_rule_route(
            request=request,
            session=session,
            session_id=session_id,
            message_with_ocr=message,
            rewrite_result=rewrite_result,
            resume_text=resume_text,
        )
    exec_latency = (time.time() - t1) * 1000

    # --  enrich response with meta info --
    response["_meta"] = {
        "rewrite_latency_ms": round(rewrite_latency, 2),
        "exec_latency_ms": round(exec_latency, 2),
        "total_latency_ms": round((time.time() - t0) * 1000, 2),
        "rewritten_query": rewrite_result.rewritten_query,
        "search_keywords": rewrite_result.search_keywords,
        "is_follow_up": rewrite_result.is_follow_up,
        "follow_up_type": rewrite_result.follow_up_type,
    }

    return response


def print_process_summary(response: dict, debug_mode: bool):
    """打印过程摘要到终端"""

    route_meta = response.get("route_meta", {})
    agent = response.get("agent", {})
    llm_agent = response.get("llm_agent", {})
    memory = response.get("memory", {})
    meta = response.get("_meta", {})
    reply = response.get("reply", {})

    intent = route_meta.get("intent", "unknown")
    confidence = route_meta.get("confidence", 0)
    plan_tasks = route_meta.get("plan_tasks", 0)
    is_clarification = response.get("is_clarification", False)

    _print_divider()

    # 意图识别
    _print_step("意图", f"{intent} (置信度: {confidence:.2f})")

    if is_clarification:
        _print_step("状态", "需要澄清")
        reply_text = reply.get("text", "") or reply.get("content", "")
        if reply_text:
            _print_reply(reply_text)
        return

    # Query 改写
    _print_step("改写", f"{meta.get('rewritten_query', '')}")
    _print_step("关键词", f"{meta.get('search_keywords', '')}")

    # Plan
    _print_step("Plan", f"{plan_tasks} 个任务")

    # 工具执行
    tools = agent.get("tools", [])
    if tools:
        tool_names = []
        for t in tools:
            name = t.get("tool", "")
            status = t.get("status", "")
            if "fail" in status.lower() or "error" in status.lower():
                tool_names.append(f"{name}(fail)")
            else:
                tool_names.append(name)
        _print_step("工具", " -> ".join(tool_names))

    # LLM Agent 工具输出摘要
    tool_outputs = llm_agent.get("tool_outputs", [])
    if tool_outputs and debug_mode:
        for to in tool_outputs:
            _print_step("输出", f"{to.get('tool')} ({to.get('task_id')})")

    # 记忆状态
    _print_step(
        "记忆",
        f"工作记忆={memory.get('working_turns', 0)}轮 "
        f"压缩块={memory.get('compressed_blocks', 0)} "
        f"证据缓存={memory.get('evidence_cache_size', 0)}",
    )

    # 延迟
    _print_step(
        "延迟",
        f"改写={meta.get('rewrite_latency_ms', 0):.0f}ms "
        f"执行={meta.get('exec_latency_ms', 0):.0f}ms "
        f"总计={meta.get('total_latency_ms', 0):.0f}ms",
    )

    # 回复
    reply_text = reply.get("text", "") or reply.get("content", "")
    if debug_mode:
        _print_step("回复", reply_text[:500])
    else:
        _print_reply(reply_text)


def _safe_json(obj) -> str:
    """安全 JSON 序列化，移除 surrogate 字符"""
    try:
        text = json.dumps(obj, ensure_ascii=False, default=str)
        # 移除/替换 surrogate 字符
        return text.encode("utf-8", "surrogatepass").decode("utf-8", "replace")
    except Exception:
        # 如果 still 失败，用 ensure_ascii
        return json.dumps(obj, ensure_ascii=True, default=str)


def write_log(response: dict):
    """将完整响应写入 JSONL 日志"""
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(_safe_json(response) + "\n")
    except Exception as e:
        logger.warning(f"日志写入失败: {e}")


# ═══════════════════════════════════════════════════════
# 4. 主循环
# ═══════════════════════════════════════════════════════

async def main():
    # 强制 stdin/stdout 使用 UTF-8（Windows 管道模式下需要）
    import io
    if hasattr(sys.stdin, "buffer"):
        sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    _print_banner()

    # 初始化 session
    session_id = f"terminal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_id, session = _get_or_create_session(session_id, user_id="terminal_user")
    print(f"会话 ID: {session_id}\n")

    debug_mode = False
    turn_count = 0

    while True:
        try:
            raw_input = input("你: ")
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        # 清理非法 surrogate 字符（Windows 管道输入时可能出现）
        user_input = raw_input.encode("utf-8", "surrogatepass").decode("utf-8", "ignore").strip()

        if not user_input:
            continue

        # 命令处理
        if user_input.lower() in ("exit", "quit", "/q"):
            print("再见!")
            break

        if user_input == "/clear":
            session_id, session = _get_or_create_session(None, user_id="terminal_user")
            print("[会话记忆已清空]\n")
            continue

        if user_input == "/debug":
            debug_mode = not debug_mode
            mode_str = "开启" if debug_mode else "关闭"
            print(f"[详细模式已{mode_str}]\n")
            continue

        if user_input == "/memory":
            state = _build_memory_state(session)
            print("\n当前记忆状态:")
            print(json.dumps(state, indent=2, ensure_ascii=False))
            print()
            continue

        # 处理消息
        turn_count += 1
        print("[处理中...]")

        try:
            response = await process_message(user_input, session, session_id, debug_mode)
            write_log(response)

            if not debug_mode:
                print_process_summary(response, debug_mode=False)
            else:
                print_process_summary(response, debug_mode=True)
                # debug 模式下额外打印完整 JSON
                _print_divider()
                print(json.dumps(response, indent=2, ensure_ascii=False, default=str)[:2000])
                print()

        except Exception as e:
            _print_error(str(e))
            import traceback
            traceback.print_exc()

    # 退出时打印日志路径
    print(f"\n完整日志已保存至: {LOG_FILE}\n")


if __name__ == "__main__":
    asyncio.run(main())
