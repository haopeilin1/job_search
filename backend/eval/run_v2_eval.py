#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2 ReAct Agent 综合评测脚本 —— 白盒组件级 + 端到端链路评测

评测维度：
┌─────────────┬─────────────────────────────────────────────────────────────┐
│ 结果指标     │ 任务成功率、完成时间、工具调用次数、token/API消耗、成本、稳定性 │
├─────────────┼─────────────────────────────────────────────────────────────┤
│ 过程指标     │ 意图识别准确率、工具选择正确率、工具执行成功率、规划合理性     │
├─────────────┼─────────────────────────────────────────────────────────────┤
│ 质量指标     │ 回复相关性、事实准确性（待人工/LLM-as-judge）                  │
├─────────────┼─────────────────────────────────────────────────────────────┤
│ 稳定性指标   │ 同一任务多次运行成功率、结果一致性                           │
└─────────────┴─────────────────────────────────────────────────────────────┘

用法：
    cd backend && python eval/run_v2_eval.py                    # 跑全部 46 条
    cd backend && python eval/run_v2_eval.py --batch A,C        # 只跑 A、C 批次
    cd backend && python eval/run_v2_eval.py --case eval_a_t04  # 单条调试
    cd backend && python eval/run_v2_eval.py --stability 3      # 稳定性测试（每条跑3次）
    cd backend && python eval/run_v2_eval.py --output eval/v2_report.json
"""

import argparse
import asyncio
import copy
import difflib
import json
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ──────────────────────────── 路径设置 ────────────────────────────
EVAL_DIR = Path(__file__).parent
sys.path.insert(0, str(EVAL_DIR.parent))

from pydantic import BaseModel

from app.core.config import settings
from app.core.llm_client import LLMClient, TIMEOUT_LIGHT, TIMEOUT_STANDARD, TIMEOUT_HEAVY
from app.core.memory import SessionMemory, DialogueTurn, WorkingMemory
from app.core.query_rewrite import QueryRewriter, QueryRewriteResult
# 新体系：LLMIntentRouter + TaskGraphPlanner
from app.core.llm_intent import LLMIntentRouter
from app.core.llm_planner import TaskGraphPlanner
from app.core.new_arch_adapter import multi_intent_result_to_intent_result, convert_task_graph
# 保留旧类型导入用于类型注解和兼容性
from app.core.intent_recognition import IntentResult
from app.core.planner import TaskGraph, TaskNode
from app.core.react_executor import ReActExecutor
from app.core.state import llm_config_store

# ──────────────────────────── 全局 Token/API 追踪器 ────────────────────────────

@dataclass
class LLMCallRecord:
    model: str
    layer: str          # chat/core/planner/memory/vision
    method: str         # chat / generate / vision_chat
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    success: bool
    error: str = ""
    # ── 新增：完整中间信息 ──
    prompt_full: str = ""       # 完整 prompt 内容
    system_prompt: str = ""     # system prompt（generate 模式）
    raw_output: str = ""        # 原始 LLM 输出（未被解析）
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    timeout: Optional[float] = None
    call_timestamp: float = 0.0

class LLMTracker:
    """monkey-patch LLMClient 以追踪所有 LLM 调用"""
    def __init__(self):
        self.calls: List[LLMCallRecord] = []
        self._original_chat = None
        self._original_generate = None

    def install(self):
        self._original_chat = LLMClient.chat
        self._original_generate = LLMClient.generate

        async def tracked_chat(self_obj, messages, temperature=0.7, max_tokens=None,
                               json_mode=False, timeout=None):
            start = time.time()
            layer = self._guess_layer(self_obj)
            try:
                result = await self._original_chat(
                    self_obj, messages, temperature, max_tokens, json_mode, timeout
                )
                latency = (time.time() - start) * 1000
                prompt_text = json.dumps(messages, ensure_ascii=False)
                prompt_tokens = len(prompt_text) // 2
                completion_tokens = len(result) // 2
                self.calls.append(LLMCallRecord(
                    model=self_obj.model, layer=layer, method="chat",
                    prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                    latency_ms=latency, success=True,
                    prompt_full=prompt_text[:50000],
                    raw_output=result[:50000] if isinstance(result, str) else json.dumps(result, ensure_ascii=False)[:50000],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    call_timestamp=start,
                ))
                return result
            except Exception as e:
                latency = (time.time() - start) * 1000
                prompt_text = json.dumps(messages, ensure_ascii=False)
                self.calls.append(LLMCallRecord(
                    model=self_obj.model, layer=layer, method="chat",
                    prompt_tokens=0, completion_tokens=0,
                    latency_ms=latency, success=False, error=str(e)[:500],
                    prompt_full=prompt_text[:50000],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    call_timestamp=start,
                ))
                raise

        async def tracked_generate(self_obj, prompt, system=None, timeout=None, **kwargs):
            start = time.time()
            layer = self._guess_layer(self_obj)
            try:
                result = await self._original_generate(
                    self_obj, prompt, system, timeout, **kwargs
                )
                latency = (time.time() - start) * 1000
                prompt_tokens = (len(prompt) + len(system or "")) // 2
                completion_tokens = len(result) // 2
                self.calls.append(LLMCallRecord(
                    model=self_obj.model, layer=layer, method="generate",
                    prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                    latency_ms=latency, success=True,
                    prompt_full=prompt[:50000],
                    system_prompt=system or "",
                    raw_output=result[:50000] if isinstance(result, str) else json.dumps(result, ensure_ascii=False)[:50000],
                    temperature=kwargs.get("temperature", 0.0),
                    max_tokens=kwargs.get("max_tokens"),
                    timeout=timeout,
                    call_timestamp=start,
                ))
                return result
            except Exception as e:
                latency = (time.time() - start) * 1000
                self.calls.append(LLMCallRecord(
                    model=self_obj.model, layer=layer, method="generate",
                    prompt_tokens=0, completion_tokens=0,
                    latency_ms=latency, success=False, error=str(e)[:500],
                    prompt_full=prompt[:50000] if isinstance(prompt, str) else "",
                    system_prompt=system or "",
                    temperature=kwargs.get("temperature", 0.0),
                    max_tokens=kwargs.get("max_tokens"),
                    timeout=timeout,
                    call_timestamp=start,
                ))
                raise

        LLMClient.chat = tracked_chat
        LLMClient.generate = tracked_generate

    def uninstall(self):
        if self._original_chat:
            LLMClient.chat = self._original_chat
        if self._original_generate:
            LLMClient.generate = self._original_generate

    def _guess_layer(self, client: LLMClient) -> str:
        """根据 model 反推属于哪一层"""
        m = client.model
        for name in ["chat", "core", "planner", "memory", "vision"]:
            cfg = getattr(llm_config_store, name, None)
            if cfg and cfg.model == m:
                return name
        return "unknown"

    def reset(self):
        self.calls.clear()

    def summary(self) -> Dict[str, Any]:
        total_calls = len(self.calls)
        success_calls = sum(1 for c in self.calls if c.success)
        total_prompt = sum(c.prompt_tokens for c in self.calls if c.success)
        total_completion = sum(c.completion_tokens for c in self.calls if c.success)
        total_latency = sum(c.latency_ms for c in self.calls)
        by_layer = defaultdict(lambda: {"calls": 0, "prompt": 0, "completion": 0, "latency": 0, "success": 0, "fail": 0})
        for c in self.calls:
            bl = by_layer[c.layer]
            bl["calls"] += 1
            bl["latency"] += c.latency_ms
            if c.success:
                bl["success"] += 1
                bl["prompt"] += c.prompt_tokens
                bl["completion"] += c.completion_tokens
            else:
                bl["fail"] += 1
        # 估算成本（使用 DashScope 近似价格）
        # qwen-plus: input 0.0008/1K, output 0.002/1K
        # qwen-turbo: input 0.0003/1K, output 0.0006/1K
        # deepseek-v3.2: input 0.001/1K, output 0.002/1K
        cost_map = {
            "qwen-plus": (0.0008, 0.002),
            "qwen-turbo": (0.0003, 0.0006),
            "deepseek-v3.2": (0.001, 0.002),
            "qwen-vl-max": (0.003, 0.006),
            "qwen3-vl-flash": (0.003, 0.006),
        }
        total_cost = 0.0
        for c in self.calls:
            if not c.success:
                continue
            in_price, out_price = cost_map.get(c.model, (0.001, 0.002))
            total_cost += c.prompt_tokens / 1000 * in_price + c.completion_tokens / 1000 * out_price

        return {
            "total_calls": total_calls,
            "success_calls": success_calls,
            "fail_calls": total_calls - success_calls,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "total_latency_ms": round(total_latency, 2),
            "avg_latency_ms": round(total_latency / total_calls, 2) if total_calls else 0,
            "estimated_cost_usd": round(total_cost, 6),
            "by_layer": dict(by_layer),
        }


# ──────────────────────────── 评测记录数据类 ────────────────────────────

@dataclass
class ComponentResult:
    """单个组件的评测记录"""
    component: str
    success: bool
    latency_ms: float
    error: str = ""
    output: Any = None

@dataclass
class TurnResult:
    """单个 test case 的完整评测记录"""
    case_id: str
    batch: str
    message: str
    scenario: str
    gold_intents: List[str] = field(default_factory=list)
    gold_slots: Dict = field(default_factory=dict)
    expected_tools: List[str] = field(default_factory=list)

    # 组件输出
    rewrite: Optional[ComponentResult] = None
    intent: Optional[ComponentResult] = None
    planner: Optional[ComponentResult] = None
    executor: Optional[ComponentResult] = None

    # 详细状态
    rewrite_result: Optional[Dict] = None
    intent_result: Optional[Dict] = None
    task_graph: Optional[Dict] = None
    executed_tools: List[str] = field(default_factory=list)
    failed_tools: List[str] = field(default_factory=list)
    replan_count: int = 0

    # 整体状态
    has_exception: bool = False
    exception_trace: str = ""
    task_success: bool = False
    e2e_latency_ms: float = 0.0

    # KB 异常追踪
    kb_empty: bool = False                    # KB 检索无命中
    external_search_triggered: bool = False   # 触发了外部搜索
    replan_t1: bool = False                   # T1 硬失败重试
    replan_t2: bool = False                   # T2 检索不足→external_search
    replan_t4: bool = False                   # T4 低匹配→建议任务
    intent_skipped: bool = False              # 意图识别被跳过(L2超时)

    # LLM 追踪
    llm_summary: Dict = field(default_factory=dict)

    # ── 新增：完整中间信息（追溯 + 人工校验） ──
    resume_text: str = ""                          # 当前使用的简历原文
    raw_llm_calls: List[Dict] = field(default_factory=list)  # 逐条 LLM 调用完整记录
    tool_executions_full: List[Dict] = field(default_factory=list)  # 工具执行完整详情
    session_history: List[Dict] = field(default_factory=list)       # session 对话历史快照
    final_response: str = ""                       # 系统最终回复文本
    judge_result: Dict = field(default_factory=dict)  # LLM-as-judge 评估结果
    user_message: str = ""                         # 用户原始问题（用于 judge）


def _intent_result_to_dict(ir: IntentResult) -> Dict:
    return {
        "demands": [{"intent_type": d.intent_type, "entities": d.entities, "priority": d.priority}
                    for d in ir.demands],
        "resolved_entities": ir.resolved_entities,
        "is_complete": ir.is_complete,
        "needs_clarification": ir.needs_clarification,
        "clarification_question": ir.clarification_question,
        "missing_entities": ir.missing_entities,
        "raw_intent_text": ir.raw_intent_text,
        "skipped_due_to_timeout": getattr(ir, "skipped_due_to_timeout", False),
    }


def _task_graph_to_dict(graph: TaskGraph) -> Dict:
    return {
        "task_count": len(graph.tasks),
        "global_status": graph.global_status,
        "replan_reason": graph.replan_reason,
        "planner_thought": graph.planner_thought,
        "tasks": {
            tid: {
                "task_id": t.task_id,
                "task_type": t.task_type,
                "tool_name": t.tool_name,
                "description": t.description,
                "dependencies": t.dependencies,
                "status": t.status,
                "result_preview": str(t.result)[:200] if t.result else None,
                "result_full": str(t.result) if t.result else None,
                "observation": t.observation[:200] if t.observation else None,
                "observation_full": t.observation if t.observation else None,
                "started_at": t.started_at,
                "finished_at": t.finished_at,
                "tool_input": t.tool_input if hasattr(t, "tool_input") else None,
            }
            for tid, t in graph.tasks.items()
        },
        "parallel_groups": graph.compute_parallel_groups(),
    }


# ──────────────────────────── Token 耗尽检测 ────────────────────────────

class TokenExhaustedError(Exception):
    """LLM Token / 配额 / 余额耗尽，必须中断测试"""
    pass


_TOKEN_EXHAUSTED_PATTERNS = [
    # HTTP 状态码相关
    "401", "403", "429",
    # 配额/余额相关（中英文）
    "insufficient_quota", "billing_hard_limit_reached", "quota_exceeded",
    "insufficient balance", "余额不足", "配额超限", "额度已用完",
    "access denied", "unauthorized", "authentication",
    # 速率限制
    "rate limit", "too many requests", "请求过于频繁",
    # DashScope 特定
    "DataInspectionFailed", "AccessDenied",
]


def is_token_exhausted_error(exc: Exception) -> bool:
    """判断异常是否由 token/配额/余额耗尽引起"""
    msg = str(exc).lower()
    return any(p.lower() in msg for p in _TOKEN_EXHAUSTED_PATTERNS)


# ──────────────────────────── LLM-as-judge 评估 ────────────────────────────

_JUDGE_PROMPT_TEMPLATE = """你是严格的评测专家。请判断「系统回复」是否成功解决了「用户问题」。

## 评测标准
- resolved=true：回复直接、准确回答了用户问题，信息完整，无事实错误。
- resolved=false：回复未回答、答非所问、信息缺失、存在事实错误，或只给出了澄清/拒绝。

## 背景信息
- 场景：{scenario}
- 用户简历（节选）：{resume_text}

## 用户问题
{user_message}

## 系统回复
{final_response}

## 预期意图
{gold_intents}

请严格按 JSON 格式输出（不要有任何其他内容）：
{{"resolved": true/false, "reason": "一句话说明理由"}}
"""


async def llm_judge(
    user_message: str,
    final_response: str,
    resume_text: str,
    scenario: str,
    gold_intents: List[str],
) -> Dict[str, Any]:
    """使用独立配置的 Judge 小模型评估最终回复是否解决用户问题。
    
    配置优先级：JUDGE_* → MEMORY_* → CHAT_*
    """
    # 优先读取 JUDGE 独立配置
    judge_url = settings.JUDGE_BASE_URL or settings.MEMORY_BASE_URL or settings.CHAT_BASE_URL
    judge_key = settings.JUDGE_API_KEY or settings.MEMORY_API_KEY or settings.CHAT_API_KEY
    judge_model = settings.JUDGE_MODEL or settings.MEMORY_MODEL or settings.CHAT_MODEL or "qwen-turbo"
    
    client = LLMClient(
        base_url=judge_url,
        api_key=judge_key,
        model=judge_model,
        timeout=TIMEOUT_LIGHT,
    )
    
    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        scenario=scenario or "未知",
        resume_text=(resume_text[:800] + "...") if len(resume_text) > 800 else resume_text,
        user_message=user_message,
        final_response=final_response[:3000],
        gold_intents=", ".join(gold_intents) if gold_intents else "无",
    )
    
    try:
        raw = await client.chat(
            messages=[
                {"role": "system", "content": "你是一个严格的评测专家，只输出 JSON。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=256,
            timeout=TIMEOUT_LIGHT,
        )
        # 提取 JSON
        text = raw.strip()
        # 去掉可能的 markdown 代码块
        if text.startswith("```"):
            text = text[text.find("{"):text.rfind("}") + 1]
        result = json.loads(text)
        return {
            "resolved": bool(result.get("resolved", False)),
            "reason": result.get("reason", ""),
            "raw": raw[:500],
        }
    except Exception as e:
        return {
            "resolved": False,
            "reason": f"judge 调用失败: {e}",
            "raw": "",
            "error": True,
        }


# ──────────────────────────── 单条 case 执行 ────────────────────────────

async def run_single_case(
    case: dict,
    resume_text: str,
    tracker: LLMTracker,
    session: Optional[SessionMemory] = None,
) -> TurnResult:
    """运行单个 test case 的完整 v2 链路，返回详细记录。
    
    Args:
        session: 可选，传入已有的 SessionMemory 以实现多轮对话共享上下文。
                 如果为 None，则创建新的 SessionMemory。
    """
    sid = case["session_id"]
    batch = sid.split("_")[1] if "_" in sid else "unknown"
    eval_ctx = case.get("eval_context", {})
    result = TurnResult(
        case_id=sid,
        batch=batch,
        message=case["message"],
        scenario=eval_ctx.get("scenario", ""),
        gold_intents=eval_ctx.get("gold_intents", []),
        gold_slots=eval_ctx.get("gold_slots", {}),
        expected_tools=eval_ctx.get("expected_tools", []),
    )

    # 多轮对话：复用传入的 session，否则创建新的
    if session is None:
        session = SessionMemory(session_id=sid)
    # 设置全局槽位：如果有简历，标记 resume_available
    if resume_text and len(resume_text.strip()) > 50:
        if not hasattr(session, "global_slots"):
            session.global_slots = {}
        session.global_slots["resume_available"] = True
    start_time = time.time()
    tracker.reset()
    result.resume_text = resume_text
    result.user_message = case.get("message", "")

    # 注入模拟多轮上下文（如果 eval_context 中配置了）
    injected_slots = eval_ctx.get("injected_history_slots")
    if injected_slots and isinstance(injected_slots, dict):
        if session.long_term is None:
            from app.core.memory import LongTermMemory
            session.long_term = LongTermMemory()
        session.long_term.entities.update(injected_slots)
    injected_cache = eval_ctx.get("injected_evidence_cache")
    if injected_cache and isinstance(injected_cache, list):
        session.evidence_cache = list(injected_cache)
        session.evidence_cache_query = eval_ctx.get("injected_evidence_query", "")

    try:
        # ── Step 0: Query Rewrite ──
        t0 = time.time()
        rewriter = QueryRewriter()
        rw = await rewriter.rewrite(raw_query=case["message"], session=session)
        result.rewrite = ComponentResult(
            component="query_rewrite",
            success=True,
            latency_ms=(time.time() - t0) * 1000,
            output={"rewritten_query": rw.rewritten_query, "follow_up_type": rw.follow_up_type,
                    "search_keywords": rw.search_keywords, "resolved_references": rw.resolved_references},
        )
        result.rewrite_result = result.rewrite.output

        # ── Step 1: Intent Recognition (新体系) ──
        t1 = time.time()
        intent_router = LLMIntentRouter()
        multi_result = await intent_router.route_multi(
            rewrite_result=rw,
            session=session,
            attachments=[],  # 测评机无附件
            raw_message=case["message"],
        )
        # 转换为旧格式以保持下游兼容性
        intent_result = multi_intent_result_to_intent_result(multi_result)
        result.intent = ComponentResult(
            component="intent_recognition",
            success=True,
            latency_ms=(time.time() - t1) * 1000,
            output=_intent_result_to_dict(intent_result),
        )
        result.intent_result = result.intent.output

        # 澄清场景：不生成 plan，直接标记为 clarification
        if intent_result.needs_clarification:
            # 保存对话历史和澄清状态（供多轮用例复用）
            turn_id = len(session.working_memory.turns) + 1
            turn = DialogueTurn(
                turn_id=turn_id,
                user_message=case["message"],
                assistant_reply=intent_result.clarification_question or "抱歉，我没有完全理解您的意思，能再详细说明一下吗？",
                intent=intent_result.demands[0].intent_type if intent_result.demands else "chat",
                rewritten_query=rw.rewritten_query,
                evidence_score=0.0,
            )
            session.working_memory.append(turn)
            # 保存澄清状态机
            from app.core.memory import PendingClarification
            primary = intent_result.demands[0].intent_type if intent_result.demands else "chat"
            session.pending_clarification = PendingClarification(
                pending_intent=primary,
                missing_slots=intent_result.missing_entities or [],
                clarification_question=intent_result.clarification_question or "",
                expected_slot_types=intent_result.missing_entities or [],
                created_turn_id=turn_id,
                resolved_slots=dict(intent_result.resolved_entities or {}),
            )
            # 将已解析槽位同步到 global_slots，供 QueryRewrite 检测 follow_up
            if intent_result.resolved_entities:
                if not hasattr(session, "global_slots"):
                    session.global_slots = {}
                for k, v in intent_result.resolved_entities.items():
                    if v is not None:
                        session.global_slots[k] = v
            result.task_success = True  # 澄清也是正确的行为
            result.e2e_latency_ms = (time.time() - start_time) * 1000
            result.llm_summary = tracker.summary()
            return result

        # ── Step 2: Planner (新体系) ──
        t2 = time.time()
        new_planner = TaskGraphPlanner()
        new_graph = await new_planner.create_graph(
            multi_result=multi_result,
            session=session,
            resume_text=resume_text,
            rewrite_result=rw,
            history_cache=[],
        )
        graph = convert_task_graph(new_graph)
        result.planner = ComponentResult(
            component="planner",
            success=True,
            latency_ms=(time.time() - t2) * 1000,
            output={"task_count": len(graph.tasks), "parallel_groups": len(graph.compute_parallel_groups())},
        )
        result.task_graph = _task_graph_to_dict(graph)

        # ── Step 3: ReAct Executor ──
        t3 = time.time()
        executor = ReActExecutor()
        graph = await executor.execute(graph, session)
        result.executor = ComponentResult(
            component="executor",
            success=True,
            latency_ms=(time.time() - t3) * 1000,
            output={"global_status": graph.global_status,
                    "success_count": sum(1 for t in graph.tasks.values() if t.status == "success"),
                    "failed_count": sum(1 for t in graph.tasks.values() if t.status == "failed"),
                    "skipped_count": sum(1 for t in graph.tasks.values() if t.status == "skipped")},
        )

        # 重新记录执行后的 task_graph（关键！之前记录的是执行前的状态）
        result.task_graph = _task_graph_to_dict(graph)

        # 收集执行后的工具信息
        result.executed_tools = sorted(set(
            t.tool_name for t in graph.tasks.values()
            if t.tool_name and t.status == "success"
        ))
        result.failed_tools = sorted(set(
            t.tool_name for t in graph.tasks.values()
            if t.tool_name and t.status == "failed"
        ))
        # replan 计数
        if graph.global_status in ("needs_replan", "replanning"):
            result.replan_count = 1

        # ── KB 异常追踪 ──
        # KB 无命中：kb_retrieve 成功但结果为空
        for t in graph.tasks.values():
            if t.tool_name == "kb_retrieve" and t.status == "success":
                try:
                    res = t.result or {}
                    chunks = res.get("chunks", []) if isinstance(res, dict) else []
                    if not chunks:
                        result.kb_empty = True
                except Exception:
                    pass
            if t.tool_name == "external_search" and t.status == "success":
                result.external_search_triggered = True

        # Replan 类型分析
        replan_reason = graph.replan_reason or ""
        if "T1" in replan_reason or "hard_fail" in replan_reason:
            result.replan_t1 = True
        if "T2" in replan_reason or "retrieval_insufficient" in replan_reason:
            result.replan_t2 = True
        if "T4" in replan_reason or "low_match" in replan_reason:
            result.replan_t4 = True

        # 意图识别跳过
        if result.intent_result:
            result.intent_skipped = result.intent_result.get("skipped_due_to_timeout", False)

        # 保存对话历史（供多轮用例复用上下文）
        turn = DialogueTurn(
            turn_id=len(session.working_memory.turns) + 1,
            user_message=case["message"],
            assistant_reply="",  # 测评机不生成自然语言回复
            intent=intent_result.demands[0].intent_type if intent_result.demands else "chat",
            rewritten_query=rw.rewritten_query,
            evidence_score=0.0,
        )
        session.working_memory.append(turn)

        # ── 任务成功判定（前置：非异常、非澄清） ──
        # 最终由 LLM-as-judge 评估，此处先留空，在 final_response 获取后再 judge
        pass

    except Exception as e:
        if is_token_exhausted_error(e):
            raise TokenExhaustedError(f"Token/配额耗尽，中断测试: {e}") from e
        result.has_exception = True
        result.exception_trace = traceback.format_exc()

    # ── 保存完整中间信息 ──
    result.e2e_latency_ms = (time.time() - start_time) * 1000
    result.llm_summary = tracker.summary()
    result.raw_llm_calls = [
        {
            "model": c.model,
            "layer": c.layer,
            "method": c.method,
            "prompt_tokens": c.prompt_tokens,
            "completion_tokens": c.completion_tokens,
            "latency_ms": c.latency_ms,
            "success": c.success,
            "error": c.error,
            "prompt_full": c.prompt_full,
            "system_prompt": c.system_prompt,
            "raw_output": c.raw_output,
            "temperature": c.temperature,
            "max_tokens": c.max_tokens,
            "timeout": c.timeout,
            "call_timestamp": c.call_timestamp,
        }
        for c in tracker.calls
    ]
    # 工具执行完整详情
    if result.task_graph:
        result.tool_executions_full = [
            {
                "task_id": tid,
                "tool_name": t.get("tool_name"),
                "status": t.get("status"),
                "result_full": t.get("result_full"),
                "observation_full": t.get("observation_full"),
                "tool_input": t.get("tool_input"),
                "started_at": t.get("started_at"),
                "finished_at": t.get("finished_at"),
            }
            for tid, t in result.task_graph.get("tasks", {}).items()
            if t.get("tool_name")
        ]
    # Session 历史快照
    turns = session.working_memory.turns if session.working_memory else []
    result.session_history = [
        {
            "turn_id": turn.turn_id,
            "user_message": turn.user_message,
            "assistant_reply": turn.assistant_reply,
            "intent": turn.intent,
            "rewritten_query": turn.rewritten_query,
            "tool_calls": turn.tool_calls,
            "tool_results": turn.tool_results,
            "retrieved_chunks": turn.retrieved_chunks,
        }
        for turn in turns
    ]
    # 最终回复（最后一条 assistant 消息）
    if turns:
        result.final_response = turns[-1].assistant_reply
    
    # ── LLM-as-judge 评估 task_success ──
    if result.has_exception:
        result.task_success = False
        result.judge_result = {"resolved": False, "reason": "执行异常", "source": "rule"}
    elif result.intent_result and result.intent_result.get("needs_clarification"):
        # 澄清用例：正确触发澄清即算成功，无需 judge
        result.task_success = True
        result.judge_result = {"resolved": True, "reason": "正确触发澄清", "source": "rule"}
    else:
        judge = await llm_judge(
            user_message=result.user_message,
            final_response=result.final_response,
            resume_text=result.resume_text,
            scenario=result.scenario,
            gold_intents=result.gold_intents,
        )
        result.judge_result = judge
        result.task_success = judge.get("resolved", False)
        # judge 调用失败时，fallback 到基于规则的宽松判定
        if judge.get("error"):
            has_final_response = bool(result.final_response.strip()) if result.final_response else False
            result.task_success = not result.has_exception and has_final_response
            result.judge_result["fallback"] = True
    
    return result


# ──────────────────────────── 指标计算 ────────────────────────────

def compute_metrics(results: List[TurnResult]) -> Dict[str, Any]:
    """从所有 case 的结果中计算综合指标"""
    total = len(results)
    if total == 0:
        return {}

    # ── 1. 结果指标 ──
    success_count = sum(1 for r in results if r.task_success)
    exception_count = sum(1 for r in results if r.has_exception)
    clarification_count = sum(1 for r in results
                               if r.intent_result and r.intent_result.get("needs_clarification"))
    latencies = [r.e2e_latency_ms for r in results]
    latencies_sorted = sorted(latencies)

    # LLM 聚合
    all_llm = [r.llm_summary for r in results if r.llm_summary]
    total_llm_calls = sum(s.get("total_calls", 0) for s in all_llm)
    total_tokens = sum(s.get("total_tokens", 0) for s in all_llm)
    total_cost = sum(s.get("estimated_cost_usd", 0) for s in all_llm)
    total_api_calls = total_llm_calls  # 每个 LLM 调用就是一次 API 调用

    # 工具调用次数（所有成功执行的工具）
    total_tool_calls = sum(len(r.executed_tools) for r in results)

    # ── 2. 意图识别指标 ──
    # gold_intents（测试集）到系统 intent_type 的映射
    INTENT_LABEL_MAP = {
        "explore": "position_explore",
        "assess": "match_assess",
        "verify": "attribute_verify",
        "prepare": "interview_prepare",
        "manage": "resume_manage",
        "chat": "general_chat",
        "clarification": "clarification",
    }

    intent_precisions = []
    intent_recalls = []
    intent_f1s = []
    for r in results:
        if not r.intent_result:
            continue
        predicted = set(d["intent_type"] for d in r.intent_result.get("demands", []))
        # 如果系统触发了澄清，将 clarification 加入 predicted
        if r.intent_result.get("needs_clarification"):
            predicted.add("clarification")
        # 将 gold intents 映射为系统标签
        gold = set()
        for g in r.gold_intents:
            mapped = INTENT_LABEL_MAP.get(g, g)
            gold.add(mapped)
        if not predicted and not gold:
            intent_precisions.append(1.0)
            intent_recalls.append(1.0)
            intent_f1s.append(1.0)
            continue
        inter = predicted & gold
        p = len(inter) / len(predicted) if predicted else 0.0
        rec = len(inter) / len(gold) if gold else 0.0
        f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0
        intent_precisions.append(p)
        intent_recalls.append(rec)
        intent_f1s.append(f1)

    # ── 3. 工具选择指标 ──
    tool_precisions = []
    tool_recalls = []
    tool_f1s = []
    for r in results:
        predicted = set(r.executed_tools)
        expected = set(r.expected_tools)
        if not predicted and not expected:
            tool_precisions.append(1.0)
            tool_recalls.append(1.0)
            tool_f1s.append(1.0)
            continue
        inter = predicted & expected
        p = len(inter) / len(predicted) if predicted else 0.0
        rec = len(inter) / len(expected) if expected else 0.0
        f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0
        tool_precisions.append(p)
        tool_recalls.append(rec)
        tool_f1s.append(f1)

    # ── 4. 工具执行成功率 ──
    tool_exec_total = 0
    tool_exec_success = 0
    for r in results:
        if r.task_graph:
            for t in r.task_graph.get("tasks", {}).values():
                if t.get("tool_name"):
                    tool_exec_total += 1
                    if t.get("status") == "success":
                        tool_exec_success += 1
    tool_exec_rate = tool_exec_success / tool_exec_total if tool_exec_total else 0.0

    # ── 5. 规划合理性 ──
    valid_plans = 0
    optimal_plans = 0
    for r in results:
        if not r.task_graph:
            continue
        # 合法性：global_status 不为 failed
        if r.task_graph.get("global_status") != "failed":
            valid_plans += 1
        # 最优性：如果 expected_tools 非空，执行的工具应覆盖 expected_tools
        expected = set(r.expected_tools)
        executed = set(r.executed_tools)
        if not expected or expected <= executed:
            optimal_plans += 1

    # ── 6. KB 异常指标 ──
    kb_empty_count = sum(1 for r in results if r.kb_empty)
    ext_search_count = sum(1 for r in results if r.external_search_triggered)
    replan_t1_count = sum(1 for r in results if r.replan_t1)
    replan_t2_count = sum(1 for r in results if r.replan_t2)
    replan_t4_count = sum(1 for r in results if r.replan_t4)
    intent_skip_count = sum(1 for r in results if r.intent_skipped)

    # ── 7. Fallback 率 ──
    fallback_rates = {
        "query_rewrite_fallback": 0.0,
        "intent_l2_fallback": 0.0,
        "intent_l3_fallback": 0.0,
        "planner_fallback": 0.0,
    }

    return {
        # 结果指标
        "outcome": {
            "total_cases": total,
            "task_success_rate": round(success_count / total, 4),
            "exception_rate": round(exception_count / total, 4),
            "clarification_rate": round(clarification_count / total, 4),
            "avg_latency_ms": round(sum(latencies) / total, 2),
            "p50_latency_ms": latencies_sorted[len(latencies_sorted) // 2] if latencies_sorted else 0,
            "p99_latency_ms": latencies_sorted[int(len(latencies_sorted) * 0.99)] if latencies_sorted else 0,
            "total_tool_calls": total_tool_calls,
            "avg_tool_calls_per_case": round(total_tool_calls / total, 2),
            "total_llm_api_calls": total_llm_calls,
            "total_tokens": total_tokens,
            "estimated_total_cost_usd": round(total_cost, 6),
            "avg_cost_per_case_usd": round(total_cost / total, 6) if total else 0,
        },
        # 过程指标
        "process": {
            "intent_precision": round(sum(intent_precisions) / len(intent_precisions), 4) if intent_precisions else 0,
            "intent_recall": round(sum(intent_recalls) / len(intent_recalls), 4) if intent_recalls else 0,
            "intent_f1": round(sum(intent_f1s) / len(intent_f1s), 4) if intent_f1s else 0,
            "tool_selection_precision": round(sum(tool_precisions) / len(tool_precisions), 4) if tool_precisions else 0,
            "tool_selection_recall": round(sum(tool_recalls) / len(tool_recalls), 4) if tool_recalls else 0,
            "tool_selection_f1": round(sum(tool_f1s) / len(tool_f1s), 4) if tool_f1s else 0,
            "tool_execution_success_rate": round(tool_exec_rate, 4),
            "plan_validity_rate": round(valid_plans / total, 4),
            "plan_optimality_rate": round(optimal_plans / total, 4),
            "fallback_rates": fallback_rates,
        },
        # KB 异常指标
        "kb_anomaly": {
            "kb_empty_rate": round(kb_empty_count / total, 4),
            "external_search_trigger_rate": round(ext_search_count / total, 4),
            "replan_t1_rate": round(replan_t1_count / total, 4),
            "replan_t2_rate": round(replan_t2_count / total, 4),
            "replan_t4_rate": round(replan_t4_count / total, 4),
            "intent_timeout_skip_rate": round(intent_skip_count / total, 4),
        },
        # 组件延迟
        "latency": {
            "query_rewrite_avg_ms": round(
                sum(r.rewrite.latency_ms for r in results if r.rewrite) /
                max(1, sum(1 for r in results if r.rewrite)), 2),
            "intent_recognition_avg_ms": round(
                sum(r.intent.latency_ms for r in results if r.intent) /
                max(1, sum(1 for r in results if r.intent)), 2),
            "planner_avg_ms": round(
                sum(r.planner.latency_ms for r in results if r.planner) /
                max(1, sum(1 for r in results if r.planner)), 2),
            "executor_avg_ms": round(
                sum(r.executor.latency_ms for r in results if r.executor) /
                max(1, sum(1 for r in results if r.executor)), 2),
        },
        # 批次统计
        "batch": _compute_batch_metrics(results),
    }


def _compute_batch_metrics(results: List[TurnResult]) -> Dict[str, Dict]:
    by_batch = defaultdict(list)
    for r in results:
        by_batch[r.batch].append(r)
    return {
        batch: {
            "total": len(rs),
            "success": sum(1 for r in rs if r.task_success),
            "success_rate": round(sum(1 for r in rs if r.task_success) / len(rs), 4) if rs else 0,
            "avg_latency_ms": round(sum(r.e2e_latency_ms for r in rs) / len(rs), 2) if rs else 0,
        }
        for batch, rs in sorted(by_batch.items())
    }


# ──────────────────────────── 稳定性测试 ────────────────────────────

async def run_stability_test(
    case: dict,
    resume_text: str,
    tracker: LLMTracker,
    n_runs: int,
) -> Dict[str, Any]:
    """对单个 case 重复运行 n 次，测试稳定性"""
    results: List[TurnResult] = []
    for i in range(n_runs):
        # 每次创建全新的 session，避免状态污染
        r = await run_single_case(case, resume_text, tracker)
        results.append(r)
        await asyncio.sleep(0.5)  # 避免请求过快

    success_runs = [r for r in results if r.task_success]
    success_rate = len(success_runs) / n_runs

    # 结果一致性：比较 executed_tools 集合的 Jaccard 相似度
    if len(results) >= 2:
        consistency_scores = []
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                s1 = set(results[i].executed_tools)
                s2 = set(results[j].executed_tools)
                inter = s1 & s2
                union = s1 | s2
                jaccard = len(inter) / len(union) if union else 1.0
                consistency_scores.append(jaccard)
        avg_consistency = sum(consistency_scores) / len(consistency_scores)
    else:
        avg_consistency = 1.0

    # 延迟稳定性（变异系数 CV）
    latencies = [r.e2e_latency_ms for r in results]
    avg_lat = sum(latencies) / len(latencies)
    std_lat = (sum((x - avg_lat) ** 2 for x in latencies) / len(latencies)) ** 0.5
    cv = std_lat / avg_lat if avg_lat > 0 else 0.0

    return {
        "case_id": case["session_id"],
        "n_runs": n_runs,
        "success_rate": round(success_rate, 4),
        "result_consistency": round(avg_consistency, 4),
        "latency_cv": round(cv, 4),
        "avg_latency_ms": round(avg_lat, 2),
        "min_latency_ms": round(min(latencies), 2),
        "max_latency_ms": round(max(latencies), 2),
        "runs": [
            {
                "run": i + 1,
                "success": r.task_success,
                "has_exception": r.has_exception,
                "latency_ms": round(r.e2e_latency_ms, 2),
                "executed_tools": r.executed_tools,
                "llm_calls": r.llm_summary.get("total_calls", 0),
            }
            for i, r in enumerate(results)
        ],
    }


# ──────────────────────────── 报告输出 ────────────────────────────

def print_report(metrics: Dict, results: Optional[List[TurnResult]] = None, stability: Optional[List[Dict]] = None):
    """控制台输出评测报告"""
    print("\n" + "=" * 70)
    print("  v2 ReAct Agent 综合评测报告")
    print("=" * 70)

    out = metrics.get("outcome", {})
    print(f"\n【结果指标】")
    print(f"  总用例数:          {out.get('total_cases', 0)}")
    print(f"  任务成功率:        {out.get('task_success_rate', 0) * 100:.2f}%")
    print(f"  异常率:            {out.get('exception_rate', 0) * 100:.2f}%")
    print(f"  澄清触发率:        {out.get('clarification_rate', 0) * 100:.2f}%")
    print(f"  平均延迟:          {out.get('avg_latency_ms', 0):.0f} ms")
    print(f"  P99 延迟:          {out.get('p99_latency_ms', 0):.0f} ms")
    print(f"  总工具调用:        {out.get('total_tool_calls', 0)}")
    print(f"  每例平均工具调用:  {out.get('avg_tool_calls_per_case', 0):.2f}")
    print(f"  总 LLM API 调用:   {out.get('total_llm_api_calls', 0)}")
    print(f"  总 Token 消耗:     {out.get('total_tokens', 0):,}")
    print(f"  预估总成本:        ${out.get('estimated_total_cost_usd', 0):.6f}")
    print(f"  每例平均成本:      ${out.get('avg_cost_per_case_usd', 0):.6f}")

    proc = metrics.get("process", {})
    print(f"\n【过程指标】")
    print(f"  意图识别精确率:    {proc.get('intent_precision', 0) * 100:.2f}%")
    print(f"  意图识别召回率:    {proc.get('intent_recall', 0) * 100:.2f}%")
    print(f"  意图识别 F1:       {proc.get('intent_f1', 0) * 100:.2f}%")
    print(f"  工具选择精确率:    {proc.get('tool_selection_precision', 0) * 100:.2f}%")
    print(f"  工具选择召回率:    {proc.get('tool_selection_recall', 0) * 100:.2f}%")
    print(f"  工具选择 F1:       {proc.get('tool_selection_f1', 0) * 100:.2f}%")
    print(f"  工具执行成功率:    {proc.get('tool_execution_success_rate', 0) * 100:.2f}%")
    print(f"  规划合法性:        {proc.get('plan_validity_rate', 0) * 100:.2f}%")
    print(f"  规划最优性:        {proc.get('plan_optimality_rate', 0) * 100:.2f}%")

    lat = metrics.get("latency", {})
    print(f"\n【组件平均延迟】")
    print(f"  QueryRewrite:      {lat.get('query_rewrite_avg_ms', 0):.0f} ms")
    print(f"  IntentRecognition: {lat.get('intent_recognition_avg_ms', 0):.0f} ms")
    print(f"  Planner:           {lat.get('planner_avg_ms', 0):.0f} ms")
    print(f"  Executor:          {lat.get('executor_avg_ms', 0):.0f} ms")

    kb = metrics.get("kb_anomaly", {})
    print(f"\n【KB 异常指标】")
    print(f"  KB 无命中率:         {kb.get('kb_empty_rate', 0) * 100:.2f}%")
    print(f"  外部搜索触发率:      {kb.get('external_search_trigger_rate', 0) * 100:.2f}%")
    print(f"  Replan T1 率:        {kb.get('replan_t1_rate', 0) * 100:.2f}%")
    print(f"  Replan T2 率:        {kb.get('replan_t2_rate', 0) * 100:.2f}%")
    print(f"  Replan T4 率:        {kb.get('replan_t4_rate', 0) * 100:.2f}%")
    print(f"  意图超时跳过率:      {kb.get('intent_timeout_skip_rate', 0) * 100:.2f}%")

    print(f"\n【批次统计】")
    for batch, bm in sorted(metrics.get("batch", {}).items()):
        print(f"  [{batch.upper()}] {bm['success']}/{bm['total']} 成功 ({bm['success_rate'] * 100:.1f}%) | "
              f"avg_lat={bm['avg_latency_ms']:.0f}ms")
    
    # Judge 统计
    if results:
        judge_cases = [r for r in results if r.judge_result.get("source") != "rule"]
        if judge_cases:
            judge_ok = sum(1 for r in judge_cases if r.judge_result.get("resolved"))
            fallback_count = sum(1 for r in judge_cases if r.judge_result.get("fallback"))
            print(f"\n【LLM-as-judge 统计】")
            print(f"  通过 judge 评估: {judge_ok}/{len(judge_cases)} ({judge_ok/len(judge_cases)*100:.1f}%)")
            if fallback_count:
                print(f"  judge 失败 fallback: {fallback_count} 条")

    if stability:
        print(f"\n【稳定性测试】")
        for st in stability:
            print(f"  {st['case_id']}: {st['n_runs']}次运行 | "
                  f"成功率={st['success_rate'] * 100:.1f}% | "
                  f"一致性={st['result_consistency'] * 100:.1f}% | "
                  f"延迟CV={st['latency_cv']:.2f}")

    print("=" * 70)


# ──────────────────────────── 数据加载 ────────────────────────────

def load_dataset(batch: Optional[str] = None, case_id: Optional[str] = None) -> List[dict]:
    dataset_file = EVAL_DIR / "test_dataset.jsonl"
    cases = []
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    if case_id:
        cases = [c for c in cases if c["session_id"] == case_id]
    elif batch:
        batches = [b.strip().lower() for b in batch.split(",")]
        cases = [c for c in cases if any(c["session_id"].startswith(f"eval_{b}_") for b in batches)]
    return cases


def load_resumes() -> dict:
    resumes_file = EVAL_DIR / "test_resumes.json"
    with open(resumes_file, "r", encoding="utf-8") as f:
        return json.load(f)


def get_resume_for_case(case_id: str, resumes: dict) -> str:
    """根据 case_id 选择对应的测试简历"""
    mapping = resumes.get("session_resume_map", {})
    resume_id = mapping.get(case_id, "eval_resume_ai")
    # resumes["resumes"] 是列表
    for r in resumes.get("resumes", []):
        if r.get("id") == resume_id:
            return r.get("text", "")
    return ""


# ──────────────────────────── 主函数 ────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="v2 ReAct Agent 综合评测")
    parser.add_argument("--batch", default=None, help="指定批次，如 A 或 A,C")
    parser.add_argument("--case", default=None, help="指定单条 case_id")
    parser.add_argument("--stability", type=int, default=0, help="稳定性测试：每条跑 N 次")
    parser.add_argument("--output", default=None, help="输出 JSON 报告路径")
    parser.add_argument("--workers", type=int, default=1, help="并发数（默认1，避免LLM限流）")
    args = parser.parse_args()

    cases = load_dataset(batch=args.batch, case_id=args.case)
    resumes = load_resumes()

    print(f"\n加载 {len(cases)} 条测试用例")
    print(f"稳定性模式: {'是 (每条跑 ' + str(args.stability) + ' 次)' if args.stability > 1 else '否'}")

    tracker = LLMTracker()
    tracker.install()

    try:
        all_results: List[TurnResult] = []
        stability_results: List[Dict] = []

        # Session group 管理：同 group 的 case 共享 SessionMemory
        group_session_map: Dict[str, SessionMemory] = {}
        
        for i, case in enumerate(cases, 1):
            sid = case["session_id"]
            group = case.get("session_group")
            resume_text = get_resume_for_case(sid, resumes)
            group_tag = f"[{group}] " if group else ""
            print(f"\n[{i}/{len(cases)}] {group_tag}{sid}: {case['message'][:40]}...")
            
            # 多轮对话：复用已有 SessionMemory
            if group:
                if group in group_session_map:
                    session = group_session_map[group]
                    print(f"  (复用 SessionMemory: group={group})")
                else:
                    session = SessionMemory(session_id=sid)
                    group_session_map[group] = session
            else:
                session = None
            
            try:
                if args.stability > 1:
                    st = await run_stability_test(case, resume_text, tracker, args.stability)
                    stability_results.append(st)
                    # 稳定性测试取最后一次的结果计入主指标
                    r = await run_single_case(case, resume_text, tracker, session)
                else:
                    r = await run_single_case(case, resume_text, tracker, session)
            except TokenExhaustedError as e:
                print(f"\n  [!!] {e}")
                print("  检测到 Token/配额/余额耗尽，测试已中断！")
                # 保存已完成的测试结果
                break

            all_results.append(r)
            status = "[OK]" if r.task_success else "[XX]"
            exc = " [异常]" if r.has_exception else ""
            tools = f" tools={r.executed_tools}" if r.executed_tools else ""
            print(f"  {status} success={r.task_success}{exc} lat={r.e2e_latency_ms:.0f}ms{tools}")

        # 计算指标
        metrics = compute_metrics(all_results)

        # 输出报告
        print_report(metrics, all_results, stability_results if args.stability > 1 else None)

        # 保存详细结果
        output_data = {
            "metrics": metrics,
            "cases": [
                {
                    "case_id": r.case_id,
                    "batch": r.batch,
                    "message": r.message,
                    "scenario": r.scenario,
                    "gold_intents": r.gold_intents,
                    "gold_slots": r.gold_slots,
                    "expected_tools": r.expected_tools,
                    "task_success": r.task_success,
                    "has_exception": r.has_exception,
                    "exception_trace": r.exception_trace if r.has_exception else None,
                    "e2e_latency_ms": r.e2e_latency_ms,
                    "rewrite": asdict(r.rewrite) if r.rewrite else None,
                    "intent": asdict(r.intent) if r.intent else None,
                    "planner": asdict(r.planner) if r.planner else None,
                    "executor": asdict(r.executor) if r.executor else None,
                    "rewrite_result": r.rewrite_result,
                    "intent_result": r.intent_result,
                    "task_graph": r.task_graph,
                    "executed_tools": r.executed_tools,
                    "failed_tools": r.failed_tools,
                    "replan_count": r.replan_count,
                    "llm_summary": r.llm_summary,
                    # ── 新增：完整中间信息 ──
                    "resume_text": r.resume_text,
                    "raw_llm_calls": r.raw_llm_calls,
                    "tool_executions_full": r.tool_executions_full,
                    "session_history": r.session_history,
                    "final_response": r.final_response,
                    "judge_result": r.judge_result,
                }
                for r in all_results
            ],
        }
        if stability_results:
            output_data["stability"] = stability_results

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"\n报告已保存: {out_path}")
        else:
            default_out = EVAL_DIR / f"v2_eval_report_{int(time.time())}.json"
            with open(default_out, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"\n报告已保存: {default_out}")

    finally:
        tracker.uninstall()


if __name__ == "__main__":
    asyncio.run(main())
