#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多轮对话测试脚本 —— 完整中间过程可视化

使用方法：
    cd backend
    python eval/test_multi_turn_conversation.py

支持交互命令：
    /memory      查看当前记忆模块状态
    /reset       重置会话（清空所有记忆）
    /topic       强制切换话题（清空证据缓存，模拟话题切换）
    /turns       查看工作记忆中的所有对话轮次
    /chunks      查看证据缓存中的召回结果
    /quit        退出

注意：
- 本脚本直接调用后端各核心模块，不经过 HTTP 层
- 默认使用 LLM 路线（settings.AGENT_MODE = "llm"）
- 每轮会输出：Query改写 → 意图识别 → Plan规划 → ReAct执行 → 最终回复 → 记忆状态
"""

import asyncio
import json
import os
import sys
import io
import time
import logging
from typing import Any, Dict, List, Optional
from dataclasses import asdict

# ── 强制 UTF-8 输出（解决 Windows 终端乱码）──
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# ── 路径设置 ──
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# ── 日志配置 ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 关闭过于 verbose 的第三方日志
for name in ("httpx", "httpcore", "openai", "chromadb", "urllib3"):
    logging.getLogger(name).setLevel(logging.WARNING)

# ── 后端模块导入 ──
from app.core.config import settings
from app.core.llm_client import LLMClient, TIMEOUT_HEAVY, TIMEOUT_STANDARD, TIMEOUT_LIGHT
from app.core.memory import (
    SessionMemory, WorkingMemory, DialogueTurn,
    MemoryManager, CompressedMemory, LongTermMemory, PendingClarification,
)
from app.core.query_rewrite import QueryRewriter, QueryRewriteResult
from app.core.llm_intent import LLMIntentRouter, LLMIntentType
from app.core.llm_planner import TaskGraphPlanner
from app.core.new_arch_adapter import convert_task_graph, multi_intent_result_to_intent_result
from app.core.react_executor import ReActExecutor
from app.core.task_graph import TaskGraph as NewTaskGraph
from app.services.handlers import ChatRequest, ChatReply, _get_resume_text
from app.routers.chat import _get_or_create_session, _build_memory_state


# =======================================================
# 工具函数：美化打印
# =======================================================

def _print_section(title: str, width: int = 80):
    """打印带标题的分隔线"""
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def _print_subsection(title: str):
    """打印子标题"""
    print(f"\n  ┌─ {title}")


def _print_kv(key: str, value: Any, indent: int = 4):
    """打印键值对"""
    prefix = " " * indent
    if isinstance(value, dict):
        print(f"{prefix}{key}:")
        for k, v in value.items():
            _print_kv(k, v, indent + 2)
    elif isinstance(value, list):
        print(f"{prefix}{key}: [共 {len(value)} 项]")
        for i, item in enumerate(value[:20]):  # 最多显示20项
            if isinstance(item, dict):
                print(f"{prefix}  [{i}] {json.dumps(item, ensure_ascii=False, default=str)[:200]}")
            else:
                print(f"{prefix}  [{i}] {str(item)[:200]}")
        if len(value) > 20:
            print(f"{prefix}  ... 还有 {len(value) - 20} 项")
    else:
        val_str = str(value)
        if len(val_str) > 300:
            val_str = val_str[:300] + " ... [截断]"
        print(f"{prefix}{key}: {val_str}")


def _format_chunk(chunk: dict, idx: int) -> str:
    """格式化单个 chunk 的显示"""
    meta = chunk.get("metadata", {})
    company = meta.get("company", "未知公司")
    position = meta.get("position", "未知岗位")
    score = chunk.get("score", chunk.get("hybrid_score", 0))
    content = chunk.get("content", "")
    if len(content) > 200:
        content = content[:200] + "..."
    return (
        f"    [{idx}] {company} · {position} | score={score:.3f}\n"
        f"        {content.replace(chr(10), ' ')}"
    )


def _format_ranking(rank: dict, idx: int) -> str:
    """格式化排序结果的显示"""
    company = rank.get("company", "未知")
    position = rank.get("position", "未知")
    score = rank.get("match_score", rank.get("score", 0))
    reason = rank.get("recommend_reason", rank.get("reason", ""))
    priority = rank.get("apply_priority", "")
    return (
        f"    [{idx}] {company} · {position} | 匹配度={score:.1f} | 优先级={priority}\n"
        f"        推荐理由: {reason[:120]}"
    )


# =======================================================
# 核心测试类
# =======================================================

class MultiTurnTester:
    """多轮对话测试器"""

    def __init__(self, session_id: Optional[str] = None, user_id: Optional[str] = None):
        self.user_id = user_id or "test_user"
        self.session_id = session_id or f"test_{int(time.time() * 1000)}"
        self.session: Optional[SessionMemory] = None
        self.turn_count = 0
        self._init_session()

    def _init_session(self):
        """初始化/重置会话"""
        # 直接创建新 session，不走数据库恢复（测试脚本保持干净）
        self.session = SessionMemory(session_id=self.session_id)
        self.session.long_term = LongTermMemory(user_id=self.user_id)
        self.turn_count = 0
        print(f"\n[OK] 新会话已创建 | session_id={self.session_id}")

    def reset(self):
        """重置会话"""
        old_id = self.session_id
        self.session_id = f"test_{int(time.time() * 1000)}"
        self._init_session()
        print(f"   原会话 {old_id} 已丢弃")

    def force_topic_shift(self):
        """强制话题切换：清空证据缓存等"""
        if self.session:
            self.session.evidence_cache = []
            self.session.evidence_cache_query = ""
            old_topic = self.session.current_topic
            self.session.current_topic = f"topic_{int(time.time())}"
            print(f"\n[切换] 强制话题切换 | 旧话题={old_topic} → 新话题={self.session.current_topic}")
            print(f"   证据缓存已清空 | evidence_cache=0")

    # ──────────────────────────── 记忆状态打印 ────────────────────────────

    def print_memory_state(self):
        """打印当前记忆模块的完整状态"""
        if not self.session:
            print("\n[WARN] 会话未初始化")
            return

        _print_section("记忆模块状态总览")

        # 1. Working Memory
        wm = self.session.working_memory
        print(f"\n  【工作记忆 WorkingMemory】")
        print(f"    max_turns={wm.max_turns} | 当前轮数={len(wm.turns)}")
        for i, turn in enumerate(wm.turns):
            print(f"\n    ── Turn {turn.turn_id} ──")
            print(f"    用户: {turn.user_message[:80]}{'...' if len(turn.user_message) > 80 else ''}")
            reply = turn.assistant_reply or "(空)"
            print(f"    助手: {reply[:80]}{'...' if len(reply) > 80 else ''}")
            print(f"    意图: {turn.intent} | 改写: {turn.rewritten_query[:60]}...")
            print(f"    证据分: {turn.evidence_score:.3f} | 检索chunks: {len(turn.retrieved_chunks)}")
            if turn.task_graph_snapshot:
                tasks = turn.task_graph_snapshot.get("tasks", [])
                print(f"    TaskGraph快照: {len(tasks)} 个任务")

        # 2. Compressed Memory
        cm_list = self.session.compressed_memories
        print(f"\n  【压缩记忆 CompressedMemory】")
        print(f"    共 {len(cm_list)} 块")
        for cm in cm_list:
            print(f"\n    ── Block {cm.memory_id} ──")
            print(f"    覆盖轮次: {cm.start_turn} ~ {cm.end_turn}")
            print(f"    摘要: {cm.summary[:150]}...")
            print(f"    关键事实: {cm.key_facts}")

        # 3. Long Term Memory
        lt = self.session.long_term
        print(f"\n  【长期记忆 LongTermMemory】")
        if lt:
            print(f"    user_id={lt.user_id}")
            print(f"    实体: {json.dumps(lt.entities, ensure_ascii=False) if lt.entities else '(空)'}")
            print(f"    偏好: {json.dumps(lt.preferences, ensure_ascii=False) if lt.preferences else '(空)'}")
            print(f"    话题标记: {lt.topic_flags}")
            print(f"    最后更新: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(lt.last_updated)) if lt.last_updated else 'N/A'}")
        else:
            print("    (未初始化)")

        # 4. Evidence Cache
        print(f"\n  【证据缓存 EvidenceCache】")
        print(f"    条目数: {len(self.session.evidence_cache)}")
        print(f"    对应查询: {self.session.evidence_cache_query or '(空)'}")
        for i, chunk in enumerate(self.session.evidence_cache[:5]):
            print(_format_chunk(chunk, i))
        if len(self.session.evidence_cache) > 5:
            print(f"    ... 还有 {len(self.session.evidence_cache) - 5} 条")

        # 5. Current Topic & Pending Clarification
        print(f"\n  【话题与澄清状态】")
        print(f"    当前话题: {self.session.current_topic}")
        pc = self.session.pending_clarification
        if pc:
            print(f"    待澄清意图: {pc.pending_intent}")
            print(f"    缺失槽位: {pc.missing_slots}")
            print(f"    澄清问题: {pc.clarification_question[:100]}")
            print(f"    创建轮次: {pc.created_turn_id} | 已解析: {pc.resolved_slots}")
        else:
            print(f"    待澄清: 无")

        # 6. Global Slots (session 级别)
        if hasattr(self.session, "global_slots") and self.session.global_slots:
            print(f"\n  【全局槽位 GlobalSlots】")
            for k, v in self.session.global_slots.items():
                print(f"    {k}: {v}")

    # ──────────────────────────── 单轮执行 ────────────────────────────

    async def run_turn(self, query: str) -> str:
        """
        执行单轮对话，打印所有中间过程。
        返回最终回复文本。
        """
        if not self.session:
            self._init_session()

        self.turn_count += 1
        turn_start = time.time()

        _print_section(
            f"Turn {self.turn_count} | Query: {query[:50]}{'...' if len(query) > 50 else ''}",
            width=90,
        )

        # ── 准备输入 ──
        request = ChatRequest(
            session_id=self.session_id,
            user_id=self.user_id,
            message=query,
        )
        resume_text = _get_resume_text()
        message_with_ocr = query  # 测试脚本暂不支持图片

        # ========================================
        # Step 1: Query 改写
        # ========================================
        _print_subsection("Step 1: Query 改写")
        rewriter = QueryRewriter()
        rewrite_result = await rewriter.rewrite(raw_query=message_with_ocr, session=self.session)

        print(f"\n    原始消息: {query}")
        print(f"    改写结果: {rewrite_result.rewritten_query}")
        print(f"    搜索关键词: {rewrite_result.search_keywords}")
        print(f"    是否追问: {rewrite_result.is_follow_up}")
        print(f"    追问类型: {rewrite_result.follow_up_type}")
        if rewrite_result.resolved_references:
            print(f"    消解指代: {rewrite_result.resolved_references}")

        # ========================================
        # Step 2: 意图识别
        # ========================================
        _print_subsection("Step 2: 意图识别 (LLMIntentRouter)")
        intent_router = LLMIntentRouter()
        multi_result = await intent_router.route_multi(
            rewrite_result=rewrite_result,
            session=self.session,
            attachments=[],
            raw_message=query,
        )
        intent_result = multi_intent_result_to_intent_result(multi_result)

        # 同步全局槽位
        if multi_result.global_slots:
            if not hasattr(self.session, "global_slots"):
                self.session.global_slots = {}
            for k, v in multi_result.global_slots.items():
                if v is not None:
                    self.session.global_slots[k] = v

        # 打印意图结果
        print(f"\n    主意图: {multi_result.primary_intent.value if multi_result.primary_intent else 'None'}")
        print(f"    需要澄清: {multi_result.needs_clarification}")
        if multi_result.clarification_reason:
            print(f"    澄清原因: {multi_result.clarification_reason}")

        print(f"\n    候选意图列表 ({len(multi_result.candidates)} 个):")
        for idx, cand in enumerate(multi_result.candidates):
            print(f"      [{idx}] {cand.intent_type.value} | confidence={cand.confidence:.2f} | source={cand.source}")
            print(f"          reason: {cand.reason[:100]}")
            if cand.slots:
                print(f"          slots: {json.dumps(cand.slots, ensure_ascii=False)}")
            if cand.missing_slots:
                print(f"          missing_slots: {cand.missing_slots}")
            print(f"          rule_agreement: {cand.rule_agreement} | can_parallel: {cand.can_parallel}")

        print(f"\n    全局槽位 (global_slots):")
        if multi_result.global_slots:
            for k, v in multi_result.global_slots.items():
                print(f"      {k}: {v}")
        else:
            print("      (空)")

        print(f"\n    执行拓扑 (execution_topology):")
        for layer_idx, layer in enumerate(multi_result.execution_topology):
            intent_names = [i.value for i in layer]
            print(f"      Layer {layer_idx}: {intent_names}")

        # ── 澄清场景：直接返回澄清问题，不走后续链路 ──
        if intent_result.needs_clarification:
            reply_text = intent_result.clarification_question or "抱歉，我没有完全理解您的意思，能再详细说明一下吗？"
            print(f"\n    [WARN] 触发澄清，跳过 Plan & ReAct")
            await self._finalize_turn(query, reply_text, rewrite_result, intent_result)
            return reply_text

        # ========================================
        # Step 3: Plan 模块
        # ========================================
        _print_subsection("Step 3: Plan 模块 (TaskGraphPlanner)")
        new_planner = TaskGraphPlanner()
        new_graph = await new_planner.create_graph(
            multi_result=multi_result,
            session=self.session,
            resume_text=resume_text,
            rewrite_result=rewrite_result,
            history_cache=[],
        )
        graph = convert_task_graph(new_graph)

        print(f"\n    新体系任务数: {len(new_graph.tasks)}")
        print(f"    转换后任务数: {len(graph.tasks)}")
        print(f"    并行组: {new_graph.execution_strategy.parallel_groups}")
        print(f"    关键路径: {new_graph.execution_strategy.critical_path}")
        print(f"    预估成本: {new_graph.execution_strategy.estimated_cost}")

        # 验证错误
        errors = graph.validate()
        if errors:
            print(f"\n    [WARN] TaskGraph 验证警告 ({len(errors)} 个):")
            for e in errors:
                print(f"      - {e}")
        else:
            print(f"\n    [OK] TaskGraph 验证通过")

        # 打印任务详情
        print(f"\n    任务列表:")
        for t in graph.tasks.values():
            deps = ", ".join(t.dependencies) if t.dependencies else "无"
            print(f"      {t.task_id}: [{t.task_type}] {t.tool_name or '-'} | deps=[{deps}]")
            print(f"        description: {t.description}")
            params_str = json.dumps(t.parameters, ensure_ascii=False, default=str)[:150]
            print(f"        params: {params_str}")

        # ========================================
        # Step 4: ReAct 执行
        # ========================================
        _print_subsection("Step 4: ReAct 执行 (ReActExecutor)")
        executor = ReActExecutor()
        graph = await executor.execute(graph, self.session)

        print(f"\n    执行状态: {graph.global_status}")
        print(f"    任务统计:")
        status_counts = {}
        for t in graph.tasks.values():
            status_counts[t.status] = status_counts.get(t.status, 0) + 1
        for status, count in status_counts.items():
            emoji = {"success": "[OK]", "failed": "[ERR]", "skipped": "[SKIP]", "aborted": "[ABORT]", "pending": "[PEND]"}.get(status, "[?]")
            print(f"      {emoji} {status}: {count}")

        # 详细打印每个工具的执行结果
        print(f"\n    各任务详细结果:")
        retrieved_chunks: List[dict] = []
        ranking_results: List[dict] = []

        for t in graph.tasks.values():
            if not t.tool_name:
                continue
            emoji = {"success": "[OK]", "failed": "[ERR]", "skipped": "[SKIP]", "aborted": "[ABORT]"}.get(t.status, "[?]")
            print(f"\n      {emoji} {t.task_id} | {t.tool_name} | status={t.status}")

            if t.result:
                result = t.result if isinstance(t.result, dict) else {"data": t.result}

                if t.tool_name == "kb_retrieve" and t.status == "success":
                    chunks = result.get("chunks", [])
                    total_found = result.get("total_found", len(chunks))
                    source = result.get("source", "unknown")
                    print(f"        召回数量: {len(chunks)} / 总命中 {total_found} | source={source}")
                    for i, chunk in enumerate(chunks[:8]):  # 最多显示8条
                        print(_format_chunk(chunk, i))
                    if len(chunks) > 8:
                        print(f"        ... 还有 {len(chunks) - 8} 个 chunk")
                    retrieved_chunks = chunks

                elif t.tool_name == "global_rank" and t.status == "success":
                    rankings = result.get("rankings", [])
                    print(f"        排序结果: 共 {len(rankings)} 个岗位")
                    for i, rank in enumerate(rankings[:8]):
                        print(_format_ranking(rank, i))
                    if len(rankings) > 8:
                        print(f"        ... 还有 {len(rankings) - 8} 个结果")
                    ranking_results = rankings

                elif t.tool_name == "match_analyze" and t.status == "success":
                    score = result.get("match_score", "N/A")
                    gaps = result.get("gaps", [])
                    suggestions = result.get("suggestions", [])
                    print(f"        匹配分数: {score}")
                    print(f"        短板: {gaps}")
                    print(f"        建议: {suggestions[:3]}")

                elif t.tool_name == "qa_synthesize" and t.status == "success":
                    answer = result.get("answer", "")
                    confidence = result.get("confidence", "N/A")
                    print(f"        置信度: {confidence}")
                    print(f"        回答: {answer[:200]}...")

                elif t.tool_name == "interview_gen" and t.status == "success":
                    questions = result.get("questions", [])
                    print(f"        生成面试题: {len(questions)} 道")
                    for q in questions[:3]:
                        q_text = q.get("question", q.get("q", ""))
                        print(f"          - {q_text[:80]}")

                elif t.tool_name == "external_search" and t.status == "success":
                    chunks = result.get("chunks", [])
                    print(f"        外部搜索结果: {len(chunks)} 条")
                    for i, c in enumerate(chunks[:3]):
                        title = c.get("title", "")
                        content = c.get("content", "")[:100]
                        print(f"          [{i}] {title}: {content}...")

                else:
                    # 通用打印
                    preview = json.dumps(result, ensure_ascii=False, default=str)[:300]
                    print(f"        result_preview: {preview}...")
            elif t.observation:
                print(f"        observation: {str(t.observation)[:200]}")
            else:
                print(f"        (无结果)")

        # 更新证据缓存（模拟 chat.py 中的逻辑）
        for t in graph.tasks.values():
            if t.tool_name == "kb_retrieve" and t.status == "success" and t.result:
                chunks = t.result.get("chunks", []) if isinstance(t.result, dict) else []
                if chunks:
                    self.session.evidence_cache = chunks[:settings.EVIDENCE_CACHE_MAX_SIZE]
                    self.session.evidence_cache_query = rewrite_result.rewritten_query
                    break

        # ========================================
        # Step 5: LLM 聚合生成最终回复
        # ========================================
        _print_subsection("Step 5: 最终回复生成 (LLM聚合)")

        # 收集工具输出
        tool_outputs = []
        for t in graph.tasks.values():
            if t.status == "success" and t.result:
                result = t.result if isinstance(t.result, dict) else {"data": t.result}
                # 精简长结果
                if t.tool_name == "kb_retrieve":
                    result = dict(result)
                    chunks = result.get("chunks", [])
                    result["chunks"] = [
                        {
                            "chunk_id": c.get("chunk_id", c.get("id", "")),
                            "content": c.get("content", "")[:300],
                            "metadata": {
                                "company": c.get("metadata", {}).get("company", ""),
                                "position": c.get("metadata", {}).get("position", ""),
                            },
                        }
                        for c in chunks[:6]
                    ]
                elif t.tool_name == "global_rank":
                    result = dict(result)
                    rankings = result.get("rankings", [])
                    result["rankings"] = [
                        {
                            "company": r.get("company", ""),
                            "position": r.get("position", ""),
                            "score": r.get("score", 0),
                            "reason": r.get("reason", "")[:200],
                        }
                        for r in rankings[:5]
                    ]
                tool_outputs.append({
                    "task_id": t.task_id,
                    "tool": t.tool_name,
                    "description": t.description,
                    "result": result,
                })

        system_prompt = (
            "你是一位专业的求职顾问。请基于以下工具执行结果，给用户一个清晰、有帮助的回复。\n"
            "要求：\n"
            "1. 直接回答用户的问题，不要重复用户的原话\n"
            "2. 如果有匹配分析结果，给出具体的分数和建议\n"
            "3. 如果有检索结果，基于事实回答，不要编造\n"
            "4. 语气友好、专业、结构化"
        )
        user_prompt = (
            f"【用户问题】\n{rewrite_result.rewritten_query}\n\n"
            f"【工具执行结果】\n{json.dumps(tool_outputs, ensure_ascii=False, default=str)[:3000]}\n\n"
            f"请生成回复："
        )

        reply_text = ""
        fallback_configs = [
            ("chat", TIMEOUT_HEAVY),
            ("core", TIMEOUT_STANDARD),
            ("planner", TIMEOUT_LIGHT),
            ("memory", TIMEOUT_LIGHT),
        ]
        used_model = None
        for model_name, model_timeout in fallback_configs:
            try:
                llm = LLMClient.from_config(model_name)
                reply_text = await llm.generate(
                    prompt=user_prompt,
                    system=system_prompt,
                    temperature=0.5,
                    max_tokens=1500,
                    timeout=model_timeout,
                )
                used_model = model_name
                break
            except Exception as e:
                logger.warning(f"聚合模型 {model_name} 失败: {e}")
        else:
            reply_text = "抱歉，我在处理您的请求时遇到了问题，请稍后重试。"

        reply_text = reply_text.strip()
        print(f"\n    使用模型: {used_model or 'fallback'}")
        print(f"    最终回复:\n{'─' * 70}")
        for line in reply_text.split("\n"):
            print(f"    {line}")
        print(f"{'─' * 70}")

        # ── 结束本轮 ──
        await self._finalize_turn(query, reply_text, rewrite_result, intent_result, graph, retrieved_chunks)

        elapsed = time.time() - turn_start
        print(f"\n  [耗时] 本轮耗时: {elapsed:.2f}s")
        return reply_text

    async def _finalize_turn(
        self,
        query: str,
        reply_text: str,
        rewrite_result: QueryRewriteResult,
        intent_result,
        graph=None,
        retrieved_chunks=None,
    ):
        """保存对话轮次并触发记忆轮转"""
        turn = DialogueTurn(
            turn_id=self.turn_count,
            user_message=query,
            assistant_reply=reply_text,
            intent=intent_result.demands[0].intent_type if intent_result.demands else "chat",
            rewritten_query=rewrite_result.rewritten_query,
            evidence_score=intent_result.demands[0].confidence if intent_result.demands else 0.0,
            retrieved_chunks=retrieved_chunks or [],
        )
        # 如果有 graph，保存快照
        if graph:
            try:
                turn.task_graph_snapshot = graph.to_dict()
            except Exception:
                pass

        self.session.working_memory.append(turn)

        # 记忆轮转
        try:
            mm = MemoryManager(llm_client=LLMClient.from_config("memory"))
            await mm.rotate_memory(self.session, turn)
        except Exception as e:
            logger.warning(f"记忆轮转失败: {e}")

        # 清理过期澄清状态
        if self.session.pending_clarification:
            current_turn_id = len(self.session.working_memory.turns)
            if self.session.pending_clarification.is_expired(current_turn_id, max_gap=2):
                logger.info(
                    f"澄清状态过期清理 | intent={self.session.pending_clarification.pending_intent}"
                )
                self.session.pending_clarification = None

    # ──────────────────────────── 交互循环 ────────────────────────────

    async def interactive_loop(self):
        """交互式测试循环"""
        print("\n" + "+" + "=" * 78 + "+")
        print("|" + " " * 20 + "多轮对话测试脚本" + " " * 42 + "|")
        print("|" + " " * 78 + "|")
        print("|  当前会话: " + f"{self.session_id:<67}" + "|")
        print("|  当前模式: " + f"{settings.AGENT_MODE:<67}" + "|")
        print("|" + " " * 78 + "|")
        print("|  命令: /memory  查看记忆  |  /reset  重置会话  |  /topic  切换话题      |")
        print("|        /turns   查看轮次  |  /chunks 查看缓存  |  /quit   退出         |")
        print("+" + "=" * 78 + "+")

        while True:
            try:
                user_input = input("\n[输入] > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n再见！")
                break

            if not user_input:
                continue

            # 命令处理
            if user_input == "/quit":
                print("\n再见！")
                break
            elif user_input == "/reset":
                self.reset()
                continue
            elif user_input == "/memory":
                self.print_memory_state()
                continue
            elif user_input == "/topic":
                self.force_topic_shift()
                continue
            elif user_input == "/turns":
                self._print_turns()
                continue
            elif user_input == "/chunks":
                self._print_evidence_cache()
                continue
            elif user_input.startswith("/"):
                print(f"未知命令: {user_input}")
                print("可用命令: /memory, /reset, /topic, /turns, /chunks, /quit")
                continue

            # 执行对话轮次
            try:
                await self.run_turn(user_input)
            except Exception as e:
                logger.exception("对话执行失败")
                print(f"\n[ERR] 错误: {e}")

    def _print_turns(self):
        """打印工作记忆中的所有轮次"""
        if not self.session or not self.session.working_memory.turns:
            print("\n  (无对话历史)")
            return
        print(f"\n  共 {len(self.session.working_memory.turns)} 轮:")
        for t in self.session.working_memory.turns:
            print(f"\n  Turn {t.turn_id}: [{t.intent}]")
            print(f"    用户: {t.user_message[:100]}")
            print(f"    助手: {t.assistant_reply[:100] if t.assistant_reply else '(空)'}")
            print(f"    改写: {t.rewritten_query[:80]}...")

    def _print_evidence_cache(self):
        """打印证据缓存详情"""
        if not self.session:
            print("\n  (无会话)")
            return
        print(f"\n  证据缓存查询: {self.session.evidence_cache_query or '(空)'}")
        print(f"  缓存条目: {len(self.session.evidence_cache)}")
        for i, chunk in enumerate(self.session.evidence_cache):
            print(_format_chunk(chunk, i))


# =======================================================
# 入口
# =======================================================

async def main():
    # 确认使用 LLM 路线
    effective_mode = settings.AGENT_MODE
    if effective_mode == "auto":
        effective_mode = settings.DEFAULT_AGENT_MODE
    print(f"\n当前 Agent 模式: {settings.AGENT_MODE} (effective={effective_mode})")
    if effective_mode != "llm":
        print("[WARN]  当前不是 LLM 模式，测试脚本将强制模拟 LLM 路线逻辑")
        print("   如需测试规则路线，请修改 settings.AGENT_MODE = 'llm' 后重试")

    tester = MultiTurnTester()
    await tester.interactive_loop()


if __name__ == "__main__":
    asyncio.run(main())
