#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v3 评测脚本 —— 严格对齐评测方法

评测原则（与用户对齐）：
1. 全部使用新体系（LLM 路线 v2）
2. 输出全部中间信息，便于人工校对及回溯
3. 意图识别：多意图必须全部命中才算命中；多轮对话涉及澄清需正确处理
4. 工具调用：每个意图有主要工具，命中主要工具即算正确；但需考虑知识库覆盖（如外部搜索替代）
5. 全工具链命中率作为辅助指标
6. 任务完成率由 Judge LLM 判断
7. Token 消耗、端到端时间、三次运行稳定性
8. 额外：小模型输入标注信息与运行链路比对，输出符合情况与符合率

用法：
    cd backend && python eval/run_eval_v3.py                    # 跑全部
    cd backend && python eval/run_eval_v3.py --batch A,C        # 指定批次
    cd backend && python eval/run_eval_v3.py --case eval_a_t04  # 单条调试
    cd backend && python eval/run_eval_v3.py --stability 3      # 稳定性测试
    cd backend && python eval/run_eval_v3.py --output eval/v3_report.json
"""

import argparse
import asyncio
import copy
import json
import sys
import time
import logging
import traceback
from collections import defaultdict

logger = logging.getLogger(__name__)
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

EVAL_DIR = Path(__file__).parent
sys.path.insert(0, str(EVAL_DIR.parent))

# v2 judge 后处理（多维度评分）
from judge_postprocess import judge_single_case, _build_case_context  # noqa: E402

from pydantic import BaseModel

from app.core.config import settings
from app.core.llm_client import LLMClient, TIMEOUT_LIGHT, TIMEOUT_STANDARD, TIMEOUT_HEAVY

# 评测模式下临时加大 Ollama 超时（qwen2.5:14b 响应极慢）
TIMEOUT_LIGHT = 180.0
TIMEOUT_STANDARD = 300.0
TIMEOUT_HEAVY = 600.0

# Monkey-patch：让所有引用原始超时的模块使用新值
import app.core.llm_client as _llm_client_mod
_llm_client_mod.TIMEOUT_LIGHT = TIMEOUT_LIGHT
_llm_client_mod.TIMEOUT_STANDARD = TIMEOUT_STANDARD
_llm_client_mod.TIMEOUT_HEAVY = TIMEOUT_HEAVY

import httpx

BASE_URL = "http://127.0.0.1:8002"
CHAT_URL = f"{BASE_URL}/api/v1/chat"

from app.core.memory import SessionMemory, DialogueTurn, WorkingMemory
from app.core.query_rewrite import QueryRewriter, QueryRewriteResult
from app.core.llm_intent import LLMIntentRouter
from app.core.llm_planner import TaskGraphPlanner
from app.core.new_arch_adapter import multi_intent_result_to_intent_result, convert_task_graph
from app.core.intent_recognition import IntentResult
from app.core.planner import TaskGraph, TaskNode
from app.core.plan_executor import PlanExecutor
from app.core.state import llm_config_store


# ═══════════════════════════════════════════════════════
# 0. 全局 LLM 调用追踪器（monkey-patch）
# ═══════════════════════════════════════════════════════

@dataclass
class LLMCallRecord:
    model: str
    layer: str
    method: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    success: bool
    error: str = ""
    prompt_full: str = ""
    system_prompt: str = ""
    raw_output: str = ""
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    timeout: Optional[float] = None
    call_timestamp: float = 0.0


class LLMTracker:
    def __init__(self):
        self.calls: List[LLMCallRecord] = []
        self._original_chat = None
        self._original_generate = None
        self._original_embed = None
        self._original_rerank = None
        self.embedding_calls: List[Dict] = []
        self.reranker_calls: List[Dict] = []

    def install(self):
        self._original_chat = LLMClient.chat
        self._original_generate = LLMClient.generate

        # ── monkey-patch EmbeddingClient._api_embed ──
        from app.core.embedding import EmbeddingClient
        self._original_embed = EmbeddingClient._api_embed

        async def tracked_embed(self_obj, texts: List[str]):
            start = time.time()
            try:
                result = await self._original_embed(self_obj, texts)
                latency = (time.time() - start) * 1000
                self.embedding_calls.append({
                    "model": self_obj.model,
                    "texts_count": len(texts),
                    "total_chars": sum(len(t) for t in texts),
                    "latency_ms": latency,
                    "success": True,
                    "timestamp": start,
                })
                return result
            except Exception as e:
                latency = (time.time() - start) * 1000
                self.embedding_calls.append({
                    "model": self_obj.model,
                    "texts_count": len(texts),
                    "total_chars": sum(len(t) for t in texts),
                    "latency_ms": latency,
                    "success": False,
                    "error": str(e)[:200],
                    "timestamp": start,
                })
                raise

        EmbeddingClient._api_embed = tracked_embed

        # ── monkey-patch reranker.rerank ──
        from app.core import reranker as reranker_mod
        self._original_rerank = reranker_mod.rerank

        async def tracked_rerank(query, candidates, top_k=10, max_length=512, batch_size=8):
            start = time.time()
            try:
                result = await self._original_rerank(query, candidates, top_k, max_length, batch_size)
                latency = (time.time() - start) * 1000
                self.reranker_calls.append({
                    "query_len": len(query),
                    "candidates": len(candidates),
                    "top_k": top_k,
                    "latency_ms": latency,
                    "success": True,
                    "timestamp": start,
                })
                return result
            except Exception as e:
                latency = (time.time() - start) * 1000
                self.reranker_calls.append({
                    "query_len": len(query),
                    "candidates": len(candidates),
                    "top_k": top_k,
                    "latency_ms": latency,
                    "success": False,
                    "error": str(e)[:200],
                    "timestamp": start,
                })
                raise

        reranker_mod.rerank = tracked_rerank

        async def tracked_chat(self_obj, messages, temperature=0.7, max_tokens=None,
                               json_mode=False, timeout=None):
            start = time.time()
            layer = self._guess_layer(self_obj)
            try:
                result = await self._original_chat(self_obj, messages, temperature, max_tokens, json_mode, timeout)
                latency = (time.time() - start) * 1000
                prompt_text = json.dumps(messages, ensure_ascii=False)
                usage = getattr(self_obj, "last_usage", {}) or {}
                prompt_tokens = usage.get("prompt_tokens", len(prompt_text) // 2)
                completion_tokens = usage.get("completion_tokens", len(result) // 2)
                self.calls.append(LLMCallRecord(
                    model=self_obj.model, layer=layer, method="chat",
                    prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                    latency_ms=latency, success=True,
                    prompt_full=prompt_text[:50000],
                    raw_output=result[:50000] if isinstance(result, str) else json.dumps(result, ensure_ascii=False)[:50000],
                    temperature=temperature, max_tokens=max_tokens, timeout=timeout,
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
                    temperature=temperature, max_tokens=max_tokens, timeout=timeout,
                    call_timestamp=start,
                ))
                raise

        async def tracked_generate(self_obj, prompt, system=None, timeout=None, **kwargs):
            start = time.time()
            layer = self._guess_layer(self_obj)
            try:
                result = await self._original_generate(self_obj, prompt, system, timeout, **kwargs)
                latency = (time.time() - start) * 1000
                usage = getattr(self_obj, "last_usage", {}) or {}
                prompt_tokens = usage.get("prompt_tokens", (len(prompt) + len(system or "")) // 2)
                completion_tokens = usage.get("completion_tokens", len(result) // 2)
                self.calls.append(LLMCallRecord(
                    model=self_obj.model, layer=layer, method="generate",
                    prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                    latency_ms=latency, success=True,
                    prompt_full=prompt[:50000], system_prompt=system or "",
                    raw_output=result[:50000] if isinstance(result, str) else json.dumps(result, ensure_ascii=False)[:50000],
                    temperature=kwargs.get("temperature", 0.0),
                    max_tokens=kwargs.get("max_tokens"), timeout=timeout,
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
                    max_tokens=kwargs.get("max_tokens"), timeout=timeout,
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
        if self._original_embed:
            from app.core.embedding import EmbeddingClient
            EmbeddingClient._api_embed = self._original_embed
        if self._original_rerank:
            from app.core import reranker as reranker_mod
            reranker_mod.rerank = self._original_rerank

    def _guess_layer(self, client: LLMClient) -> str:
        m = client.model
        for name in ["chat", "core", "planner", "memory", "vision"]:
            cfg = getattr(llm_config_store, name, None)
            if cfg and cfg.model == m:
                return name
        return "unknown"

    def reset(self):
        self.calls.clear()
        self.embedding_calls.clear()
        self.reranker_calls.clear()

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

        # embedding / reranker 汇总
        emb_calls = len(self.embedding_calls)
        emb_success = sum(1 for c in self.embedding_calls if c.get("success"))
        emb_latency = sum(c.get("latency_ms", 0) for c in self.embedding_calls)
        emb_chars = sum(c.get("total_chars", 0) for c in self.embedding_calls)

        rer_calls = len(self.reranker_calls)
        rer_success = sum(1 for c in self.reranker_calls if c.get("success"))
        rer_latency = sum(c.get("latency_ms", 0) for c in self.reranker_calls)
        rer_candidates = sum(c.get("candidates", 0) for c in self.reranker_calls)

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
            "embedding": {
                "calls": emb_calls,
                "success": emb_success,
                "fail": emb_calls - emb_success,
                "total_chars": emb_chars,
                "total_latency_ms": round(emb_latency, 2),
                "avg_latency_ms": round(emb_latency / emb_calls, 2) if emb_calls else 0,
            },
            "reranker": {
                "calls": rer_calls,
                "success": rer_success,
                "fail": rer_calls - rer_success,
                "total_candidates": rer_candidates,
                "total_latency_ms": round(rer_latency, 2),
                "avg_latency_ms": round(rer_latency / rer_calls, 2) if rer_calls else 0,
            },
        }


# ═══════════════════════════════════════════════════════
# 1. 评测记录数据类（扩展版）
# ═══════════════════════════════════════════════════════

@dataclass
class ComponentResult:
    component: str
    success: bool
    latency_ms: float
    error: str = ""
    output: Any = None


@dataclass
class TurnResult:
    case_id: str
    batch: str
    message: str
    scenario: str
    gold_intents: List[str] = field(default_factory=list)
    gold_slots: Dict = field(default_factory=dict)
    expected_tools: List[str] = field(default_factory=list)

    rewrite: Optional[ComponentResult] = None
    intent: Optional[ComponentResult] = None
    planner: Optional[ComponentResult] = None
    executor: Optional[ComponentResult] = None

    rewrite_result: Optional[Dict] = None
    intent_result: Optional[Dict] = None
    task_graph: Optional[Dict] = None
    executed_tools: List[str] = field(default_factory=list)
    failed_tools: List[str] = field(default_factory=list)
    replan_count: int = 0

    has_exception: bool = False
    exception_trace: str = ""
    task_success: bool = False
    e2e_latency_ms: float = 0.0

    kb_empty: bool = False
    external_search_triggered: bool = False
    replan_t1: bool = False
    replan_t2: bool = False
    replan_t4: bool = False
    intent_skipped: bool = False

    llm_summary: Dict = field(default_factory=dict)
    resume_text: str = ""
    raw_llm_calls: List[Dict] = field(default_factory=list)
    tool_executions_full: List[Dict] = field(default_factory=list)
    session_history: List[Dict] = field(default_factory=list)
    final_response: str = ""
    judge_result: Dict = field(default_factory=dict)
    user_message: str = ""

    session_id: str = ""
    turn_id: int = 0

    # ── v3 新增：严格判定结果 ──
    intent_strict_hit: bool = False           # 意图严格命中（多意图全部命中）
    intent_hit_reason: str = ""               # 意图命中/未命中原因
    tool_primary_hit: bool = False            # 主要工具命中
    tool_primary_tools: List[str] = field(default_factory=list)  # 动态调整后的主要工具
    tool_full_chain_hit: bool = False         # 全工具链命中
    tool_hit_reason: str = ""                 # 工具命中/未命中原因
    process_quality: Dict = field(default_factory=dict)  # 小模型过程质量评估

    # ── v3 新增：embedding / reranker 埋点 ──
    embedding_calls: List[Dict] = field(default_factory=list)
    reranker_calls: List[Dict] = field(default_factory=list)

    # ── 反思模块埋点 ──
    reflection_result: Dict = field(default_factory=dict)


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


# ═══════════════════════════════════════════════════════
# 2. Token 耗尽检测
# ═══════════════════════════════════════════════════════

class TokenExhaustedError(Exception):
    pass


_TOKEN_EXHAUSTED_PATTERNS = [
    "401", "403", "429",
    "insufficient_quota", "billing_hard_limit_reached", "quota_exceeded",
    "insufficient balance", "余额不足", "配额超限", "额度已用完",
    "access denied", "unauthorized", "authentication",
    "rate limit", "too many requests", "请求过于频繁",
    "DataInspectionFailed", "AccessDenied",
]


def is_token_exhausted_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(p.lower() in msg for p in _TOKEN_EXHAUSTED_PATTERNS)


# ═══════════════════════════════════════════════════════
# 3. Judge LLM：任务完成率判断
# ═══════════════════════════════════════════════════════

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
        text = raw.strip()
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


# ═══════════════════════════════════════════════════════
# 4. 过程质量 Judge：小模型比对标注与运行链路
# ═══════════════════════════════════════════════════════

_PROCESS_QUALITY_PROMPT = """你是一位严格的评测专家。请比对以下「标注信息」和「实际运行结果」，逐条判断每个步骤是否符合预期。

## 标注信息
- 标注意图：{gold_intents}
- 标注槽位：{gold_slots}
- 标注预期工具：{expected_tools}
- 场景说明：{scenario}

## 实际运行结果
- QueryRewrite 改写结果：{rewrite_result}
- 意图识别结果：{intent_result}
- 执行工具列表：{executed_tools}
- 是否触发澄清：{is_clarification}
- 任务图状态：{graph_status}

## 判断标准
1. QueryRewrite：改写是否保留了原意，追问类型判断是否准确
2. 意图识别：是否命中了所有标注意图（多意图需全部命中，澄清场景按场景预期判断）
3. 槽位提取：提取的 company/position/attributes 是否与标注一致
4. 工具调用：是否命中了主要工具（如 kb_retrieve/qa_synthesize/match_analyze 等）
5. 澄清判断：是否按场景预期正确触发/不触发澄清

请严格按 JSON 输出，不要有任何其他内容：
{{
  "overall_match": true/false,
  "match_rate": 0.0-1.0,
  "details": {{
    "query_rewrite": {{"match": true/false, "reason": "..."}},
    "intent_recognition": {{"match": true/false, "reason": "..."}},
    "slot_extraction": {{"match": true/false, "reason": "..."}},
    "tool_invocation": {{"match": true/false, "reason": "..."}},
    "clarification": {{"match": true/false, "reason": "..."}}
  }}
}}
"""


async def process_quality_judge(
    case: dict,
    result: TurnResult,
) -> Dict[str, Any]:
    """小模型比对标注信息和实际运行链路，输出符合情况和符合率"""
    eval_ctx = case.get("eval_context", {})

    judge_url = settings.JUDGE_BASE_URL or settings.MEMORY_BASE_URL or settings.CHAT_BASE_URL
    judge_key = settings.JUDGE_API_KEY or settings.MEMORY_API_KEY or settings.CHAT_API_KEY
    judge_model = settings.JUDGE_MODEL or settings.MEMORY_MODEL or settings.CHAT_MODEL or "qwen-turbo"

    client = LLMClient(
        base_url=judge_url,
        api_key=judge_key,
        model=judge_model,
        timeout=TIMEOUT_STANDARD,
    )

    graph_status = "unknown"
    if result.task_graph:
        graph_status = result.task_graph.get("global_status", "unknown")

    prompt = _PROCESS_QUALITY_PROMPT.format(
        gold_intents=json.dumps(eval_ctx.get("gold_intents", []), ensure_ascii=False),
        gold_slots=json.dumps(eval_ctx.get("gold_slots", {}), ensure_ascii=False),
        expected_tools=json.dumps(eval_ctx.get("expected_tools", []), ensure_ascii=False),
        scenario=eval_ctx.get("scenario", "未知"),
        rewrite_result=json.dumps(result.rewrite_result, ensure_ascii=False) if result.rewrite_result else "无",
        intent_result=json.dumps(result.intent_result, ensure_ascii=False) if result.intent_result else "无",
        executed_tools=json.dumps(result.executed_tools, ensure_ascii=False),
        is_clarification=str(result.intent_result.get("needs_clarification", False)) if result.intent_result else "False",
        graph_status=graph_status,
    )

    try:
        raw = await client.chat(
            messages=[
                {"role": "system", "content": "你是严格的评测专家，只输出合法 JSON。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=512,
            timeout=TIMEOUT_STANDARD,
        )
        text = raw.strip()
        if text.startswith("```"):
            text = text[text.find("{"):text.rfind("}") + 1]
        parsed = json.loads(text)
        return {
            "overall_match": bool(parsed.get("overall_match", False)),
            "match_rate": float(parsed.get("match_rate", 0.0)),
            "details": parsed.get("details", {}),
            "raw": raw[:800],
        }
    except Exception as e:
        return {
            "overall_match": False,
            "match_rate": 0.0,
            "details": {},
            "error": str(e),
            "raw": raw[:500] if 'raw' in dir() else "",
        }


async def _run_v2_judge(case: dict, result: TurnResult, tracker: LLMTracker) -> None:
    """调用 judge_postprocess v3 多维度评分，结果直接写回 result。
    
    v3 变更：
    - 评分从 4维0-10分 → 10维0-5分
    - code_override 不覆盖 LLM 结论，双输出供人工参考
    - 传入更丰富的过程信息
    """
    result.llm_summary = tracker.summary()

    default_scores = {
        "intent_accuracy": 0, "slot_accuracy": 0, "tool_correctness": 0,
        "tool_execution": 0, "response_accuracy": 0, "response_completeness": 0,
        "citation_quality": 0, "coherence": 0, "tone": 0, "efficiency": 0,
        "faithfulness": 0, "answer_relevance": 0,
    }

    if result.has_exception:
        result.task_success = False
        result.judge_result = {
            "llm_resolved": False,
            "llm_reason": "执行异常",
            "code_resolved": False,
            "code_reason": "执行异常，code直接否决",
            "resolved": False,
            "reason": "执行异常",
            "scores": default_scores,
            "source": "rule",
            "needs_rag": "kb_retrieve" in result.expected_tools,
        }
        result.process_quality = {"overall_match": False, "match_rate": 0.0, "details": {}, "source": "rule"}
        return

    if result.intent_result and result.intent_result.get("needs_clarification"):
        # 只有当标注期望也是澄清时，才直接判定为正确触发澄清
        gold_intents = result.gold_intents or []
        if "clarification" in gold_intents:
            result.task_success = True
            result.judge_result = {
                "llm_resolved": True,
                "llm_reason": "正确触发澄清",
                "code_resolved": True,
                "code_reason": "gold_intents包含clarification，规则直接通过",
                "resolved": True,
                "reason": "正确触发澄清",
                "scores": {**default_scores, "intent_accuracy": 5, "response_accuracy": 5, "response_completeness": 5},
                "source": "rule",
            }
            result.process_quality = {"overall_match": True, "match_rate": 1.0, "details": {}, "source": "rule"}
            return
        # 否则继续走正常 judge 流程，由 LLM judge 评判澄清是否合理

    # 构造 judge_single_case 需要的 dict（传入更多过程信息）
    pred_intents = []
    pred_slots = {}
    if result.intent_result:
        for d in result.intent_result.get("demands", []):
            if isinstance(d, dict) and d.get("intent"):
                pred_intents.append(d["intent"])
                if d.get("entities"):
                    pred_slots.update(d["entities"])

    # 构建工具调用详情
    tool_details = []
    if result.tool_executions_full:
        for t in result.tool_executions_full:
            if isinstance(t, dict):
                tool_details.append({
                    "tool": t.get("tool_name", ""),
                    "status": t.get("status", ""),
                    "input": str(t.get("input", ""))[:200],
                    "output_preview": str(t.get("output", ""))[:300],
                })

    case_result = {
        "case_id": result.case_id,
        "message": result.user_message,
        "gold_intents": result.gold_intents,
        "reply": result.final_response or "",
        "tools_called": result.executed_tools or [],
        "pred_intents": pred_intents,
        "pred_slots": pred_slots,
        "tool_details": tool_details,
        "tool_executions_full": result.tool_executions_full or [],
        "failed_tools": result.failed_tools or [],
        "replan_count": result.replan_count,
        "replan_t1": result.replan_t1,
        "replan_t2": result.replan_t2,
        "replan_t4": result.replan_t4,
        "external_search_triggered": result.external_search_triggered,
        "kb_empty": result.kb_empty,
        "session_history": result.session_history or [],
    }

    ctx = _build_case_context(result.case_id, case_result)
    # 补充 resume_text（_build_case_context 从文件加载，可能缺失）
    if result.resume_text and ctx.get("resume_info") == "无简历":
        ctx["resume_info"] = result.resume_text[:800]
    # 补充过程信息到 ctx
    ctx["tool_details"] = tool_details
    ctx["replan_info"] = f"replan_count={result.replan_count}, T1={result.replan_t1}, T2={result.replan_t2}, T4={result.replan_t4}"
    ctx["external_search"] = result.external_search_triggered
    ctx["pred_slots"] = pred_slots

    try:
        jr = await judge_single_case(case_result, ctx)
    except Exception as e:
        jr = {
            "resolved": False,
            "scores": default_scores,
            "reason": f"judge 调用失败: {e}",
            "case_id": result.case_id,
            "rule_hit": None,
            "veto": False,
        }

    scores = jr.get("scores", default_scores)
    judge_resolved = jr.get("resolved", False)
    judge_reason = jr.get("reason", "")

    # ── code 辅助判断（不覆盖 LLM，双输出）──
    intent_acc = scores.get("intent_accuracy", 0)
    resp_acc = scores.get("response_accuracy", 0)
    resp_comp = scores.get("response_completeness", 0)

    # 否决项检查
    veto_reasons = []
    if resp_acc <= 1:
        veto_reasons.append("response_accuracy<=1: 严重编造或事实错误")
    if jr.get("rule_hit") == "empty_reply":
        veto_reasons.append("空回复或极短回复")
    if jr.get("rule_hit") == "fake_no_resume":
        veto_reasons.append("系统声称缺少简历但实际已传入")

    # 关键维度是否及格
    key_ok = intent_acc >= 3 and resp_acc >= 3 and resp_comp >= 3
    code_pass = key_ok and len(veto_reasons) == 0

    if veto_reasons:
        code_reason = f"否决项: {'; '.join(veto_reasons)} | 关键维度: intent={intent_acc}, resp_acc={resp_acc}, resp_comp={resp_comp}"
    else:
        code_reason = f"关键维度全部≥3 (intent={intent_acc}, resp_acc={resp_acc}, resp_comp={resp_comp})，无否决项"

    # 最终输出：保留 LLM 原始判断 + code 判断，默认用 LLM 的（人工可覆盖）
    result.judge_result = {
        # LLM 原始判断（保留）
        "llm_resolved": judge_resolved,
        "llm_reason": judge_reason,
        # Code 判断（新增）
        "code_resolved": code_pass,
        "code_reason": code_reason,
        # 默认用 LLM 的结论（人工可覆盖）
        "resolved": judge_resolved,
        "reason": judge_reason,
        "scores": scores,
        "rule_hit": jr.get("rule_hit"),
        "veto": jr.get("veto", False),
        "source": "llm",
        "judge_prompt": jr.get("judge_prompt", ""),
        "raw_output": jr.get("raw_output", ""),
    }
    result.task_success = judge_resolved

    # 过程质量：用关键维度综合估算
    result.process_quality = {
        "overall_match": result.task_success,
        "match_rate": (intent_acc + resp_acc + resp_comp) / 15.0 if (intent_acc or resp_acc or resp_comp) else 0.0,
        "details": scores,
        "source": "llm",
    }

    # 保存 embedding / reranker 埋点
    result.embedding_calls = tracker.embedding_calls.copy()
    result.reranker_calls = tracker.reranker_calls.copy()


# ═══════════════════════════════════════════════════════
# 5. 严格指标计算
# ═══════════════════════════════════════════════════════

# 新体系意图 → 旧体系标签映射（用于与标注 gold_intents 比对）
INTENT_LABEL_MAP = {
    "explore": "position_explore",
    "assess": "match_assess",
    "verify": "attribute_verify",
    "prepare": "interview_prepare",
    "manage": "resume_manage",
    "chat": "general_chat",
    "clarification": "clarification",
}

# 反向映射：旧标签 → 新体系意图
REVERSE_INTENT_MAP = {v: k for k, v in INTENT_LABEL_MAP.items()}


def compute_intent_strict_hit(result: TurnResult) -> Tuple[bool, str]:
    """
    严格意图命中判定：多意图必须全部命中才算命中。

    规则：
    1. 系统将 predicted_intents（demands 中的 intent_type）与 gold_intents 比对
    2. 多意图场景：predicted 集合必须 **等于** gold 集合（不考虑顺序）
    3. 澄清场景：
       - 若标注为 clarification，系统触发澄清 → 命中
       - 若标注为非 clarification，系统触发澄清 → 未命中（除非场景允许）
    """
    if not result.intent_result:
        return False, "意图识别结果为空"

    # 获取系统预测的意图集合
    predicted = set()
    for d in result.intent_result.get("demands", []):
        it = d.get("intent_type") or d.get("intent") or ""
        # 统一映射为新体系名称
        rev = REVERSE_INTENT_MAP.get(it, it)
        predicted.add(rev)

    # 如果系统触发澄清
    if result.intent_result.get("needs_clarification"):
        predicted.add("clarification")

    # 获取标注的 gold 意图集合
    gold = set(result.gold_intents)

    # 严格判定：集合必须相等
    if predicted == gold:
        return True, f"完全命中: predicted={predicted}, gold={gold}"

    # 未命中，分析差异
    missing = gold - predicted
    extra = predicted - gold
    reasons = []
    if missing:
        reasons.append(f"漏识: {missing}")
    if extra:
        reasons.append(f"误识: {extra}")
    return False, "; ".join(reasons)


def compute_tool_hit(result: TurnResult, expected_tools: List[str]) -> Tuple[bool, bool, List[str], str]:
    """
    工具调用命中判定。

    返回: (primary_hit, full_chain_hit, primary_tools, reason)

    规则：
    1. 主要工具命中：系统执行的工具中，命中了至少一个主要工具即算正确
    2. 全工具链命中：系统执行的工具集合 **包含** 所有主要工具
    3. 动态调整：
       - 若 kb_retrieve 结果不足（T2触发）或 rerank 分数极低，external_search 可替代 kb_retrieve
       - 若知识库无命中（kb_empty），external_search 成为主要工具
    """
    executed = set(result.executed_tools)
    failed = set(result.failed_tools)

    # 动态调整主要工具列表
    primary_tools = set(expected_tools)

    # 知识库检索不足时，external_search 可替代 kb_retrieve
    if result.replan_t2 or result.external_search_triggered or result.kb_empty:
        if "kb_retrieve" in primary_tools:
            primary_tools.add("external_search")

    # 主要工具命中判断
    primary_hit = bool(executed & primary_tools)

    # 全工具链命中判断：所有主要工具都被执行（允许额外工具）
    full_chain_hit = primary_tools <= executed

    # 生成原因
    if primary_hit:
        hit_tools = executed & primary_tools
        if full_chain_hit:
            reason = f"主要工具全部命中: {hit_tools}"
        else:
            missing = primary_tools - executed
            reason = f"主要工具部分命中: 命中={hit_tools}, 缺失={missing}"
    else:
        reason = f"主要工具全部未命中: 预期={primary_tools}, 实际执行={executed}, 失败={failed}"

    return primary_hit, full_chain_hit, list(primary_tools), reason


# ═══════════════════════════════════════════════════════
# 6. 单条 case 执行
# ═══════════════════════════════════════════════════════

def _extract_result_from_response(
    data: dict,
    case: dict,
    result: TurnResult,
    start_time: float,
) -> TurnResult:
    """从 chat_endpoint 返回的 dict 中提取评测信息（HTTP 和本地 direct 共用）。"""
    result.e2e_latency_ms = (time.time() - start_time) * 1000
    result.session_id = data.get("session_id", result.case_id)

    # 3. 从返回中提取信息
    debug = data.get("debug_info", {})
    session_history = debug.get("session_history", [])
    result.turn_id = session_history[-1].get("turn_id", 0) if session_history else 0

    route_meta = data.get("route_meta", {})
    llm_agent = data.get("llm_agent", {})
    is_clarification = data.get("is_clarification", False)
    reply = data.get("reply", {})

    # rewrite
    result.rewrite_result = {
        "rewritten_query": llm_agent.get("rewritten_query", ""),
        "follow_up_type": llm_agent.get("follow_up_type", ""),
        "search_keywords": llm_agent.get("search_keywords", ""),
        "resolved_references": llm_agent.get("global_slots", {}),
    }
    result.rewrite = ComponentResult(
        component="query_rewrite",
        success=True,
        latency_ms=0,
        output=result.rewrite_result,
    )

    # intent
    demands = route_meta.get("demands", [])
    intent_debug = debug.get("intent", {})
    result.intent_result = {
        "demands": demands,
        "needs_clarification": is_clarification,
        "clarification_question": intent_debug.get("clarification_question", ""),
        "missing_entities": intent_debug.get("missing_entities", []),
        "resolved_entities": intent_debug.get("resolved_entities", {}),
        "skipped_due_to_timeout": False,
    }
    result.intent = ComponentResult(
        component="intent_recognition",
        success=True,
        latency_ms=0,
        output=result.intent_result,
    )

    # 严格意图命中判定
    result.intent_strict_hit, result.intent_hit_reason = compute_intent_strict_hit(result)

    # 澄清场景
    if is_clarification:
        result.tool_primary_hit = False
        result.tool_full_chain_hit = False
        result.tool_hit_reason = "澄清场景，未执行工具"
        result.tool_primary_tools = result.expected_tools
        result.task_success = True
        return result

    # task_graph
    task_graph_debug = debug.get("task_graph", {})
    result.task_graph = task_graph_debug

    # 工具执行信息
    tasks = task_graph_debug.get("tasks", {})
    result.executed_tools = sorted(set(
        t.get("tool_name") for t in tasks.values()
        if t.get("tool_name") and t.get("status") == "success"
    ))
    result.failed_tools = sorted(set(
        t.get("tool_name") for t in tasks.values()
        if t.get("tool_name") and t.get("status") == "failed"
    ))
    # 【修复】提取完整工具执行记录到 tool_executions_full
    result.tool_executions_full = [
        {
            "task_id": tid,
            "tool_name": t.get("tool_name"),
            "status": t.get("status"),
            "input": t.get("input"),
            "output": t.get("output"),
            "latency_ms": t.get("latency_ms"),
            "error": t.get("error"),
        }
        for tid, t in tasks.items()
        if t.get("tool_name")
    ]

    # replan
    global_status = task_graph_debug.get("global_status", "")
    if global_status in ("needs_replan", "replanning"):
        result.replan_count = 1
    replan_reason = task_graph_debug.get("replan_reason", "")
    if "T1" in replan_reason or "hard_fail" in replan_reason:
        result.replan_t1 = True
    if "T2" in replan_reason or "retrieval_insufficient" in replan_reason:
        result.replan_t2 = True
    if "T4" in replan_reason or "low_match" in replan_reason:
        result.replan_t4 = True

    # KB 异常
    for t in tasks.values():
        if t.get("tool_name") == "kb_retrieve" and t.get("status") == "success":
            try:
                res = t.get("result", {}) or {}
                chunks = res.get("chunks", []) if isinstance(res, dict) else []
                if not chunks:
                    result.kb_empty = True
            except Exception:
                pass
        if t.get("tool_name") == "external_search" and t.get("status") == "success":
            result.external_search_triggered = True

    # 工具命中判定
    result.tool_primary_hit, result.tool_full_chain_hit, result.tool_primary_tools, result.tool_hit_reason = \
        compute_tool_hit(result, result.expected_tools)

    # 最终回复
    result.final_response = (reply.get("content") or reply.get("text", "")) if isinstance(reply, dict) else str(reply)
    result.session_history = debug.get("session_history", [])

    # 反思模块结果
    result.reflection_result = debug.get("reflection_result", {})

    return result


async def run_single_case(
    case: dict,
    resume_text: str,
    tracker: LLMTracker,
    session: Optional[SessionMemory] = None,
    reset_session: bool = False,
) -> TurnResult:
    sid = case["session_id"]
    batch = sid.split("_")[1] if "_" in sid else "unknown"
    eval_ctx = case.get("eval_context", {})
    # 多轮对话复用：若传了 session 对象，使用其 session_id 保持 HTTP session 一致
    http_session_id = session.session_id if session else sid
    result = TurnResult(
        case_id=sid,
        batch=batch,
        message=case["message"],
        scenario=eval_ctx.get("scenario", ""),
        gold_intents=eval_ctx.get("gold_intents", []),
        gold_slots=eval_ctx.get("gold_slots", {}),
        expected_tools=eval_ctx.get("expected_tools", []),
        session_id=sid,
        turn_id=eval_ctx.get("turn_id", 0),
    )

    start_time = time.time()
    tracker.reset()
    result.resume_text = resume_text
    result.user_message = case.get("message", "")

    try:
        # 1. 切换 active resume
        resume_id = case.get("resume_id")
        if resume_id:
            import httpx
            async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
                resp = await client.put(f"{BASE_URL}/api/v1/resumes/{resume_id}/activate")
                if resp.status_code != 200:
                    logger.warning(f"[Eval] 切换 resume 失败: {resp.status_code}")

        # 2. 调用全链路 HTTP API
        merged_eval_ctx = dict(eval_ctx)
        if reset_session:
            merged_eval_ctx["reset_session"] = True
        payload = {
            "session_id": http_session_id,
            "message": case["message"],
            "eval_context": merged_eval_ctx,
        }
        import httpx
        async with httpx.AsyncClient(timeout=600.0, trust_env=False) as client:
            resp = await client.post(
                CHAT_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()

        result = _extract_result_from_response(data, case, result, start_time)
    except Exception as e:
        if is_token_exhausted_error(e):
            raise TokenExhaustedError(f"Token/配额耗尽，中断测试: {e}") from e
        result.has_exception = True
        result.exception_trace = traceback.format_exc()

    # v2 多维度 Judge
    await _run_v2_judge(case, result, tracker)
    return result


async def run_single_case_direct(
    case: dict,
    resume_text: str,
    tracker: LLMTracker,
    reset_session: bool = False,
) -> TurnResult:
    """本地直接调用 chat_endpoint（不走 HTTP），用于单轮评测。"""
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
        session_id=sid,
        turn_id=eval_ctx.get("turn_id", 0),
    )

    start_time = time.time()
    tracker.reset()
    result.resume_text = resume_text
    result.user_message = case.get("message", "")

    try:
        # 1. 切换 active resume（直接调用函数）
        resume_id = case.get("resume_id")
        if resume_id:
            from app.routers.resumes import activate_resume
            try:
                await activate_resume(resume_id)
            except Exception as e:
                logger.warning(f"[Eval] 本地切换 resume 失败: {e}")

        # 2. 本地直接调用 chat_endpoint
        from app.routers.chat import chat_endpoint
        from app.services.handlers import ChatRequest

        merged_eval_ctx = dict(eval_ctx)
        if reset_session:
            merged_eval_ctx["reset_session"] = True

        request = ChatRequest(
            session_id=sid,
            message=case["message"],
            eval_context=merged_eval_ctx,
        )
        data = await chat_endpoint(request)

        result = _extract_result_from_response(data, case, result, start_time)
    except Exception as e:
        if is_token_exhausted_error(e):
            raise TokenExhaustedError(f"Token/配额耗尽，中断测试: {e}") from e
        result.has_exception = True
        result.exception_trace = traceback.format_exc()

    # 注入 LLM 调用追踪记录
    _inject_tracker_data(result, tracker)

    # v2 多维度 Judge
    await _run_v2_judge(case, result, tracker)
    return result


def _inject_tracker_data(result: TurnResult, tracker: LLMTracker):
    """将 LLMTracker 的调用记录注入到 TurnResult"""
    llm_calls = []
    for c in tracker.calls:
        llm_calls.append({
            "model": c.model,
            "layer": c.layer,
            "method": c.method,
            "prompt_tokens": c.prompt_tokens,
            "completion_tokens": c.completion_tokens,
            "latency_ms": c.latency_ms,
            "success": c.success,
            "error": c.error,
            "prompt_preview": c.prompt_full[:2000] if c.prompt_full else "",
            "system_preview": c.system_prompt[:500] if c.system_prompt else "",
            "output_preview": c.raw_output[:2000] if c.raw_output else "",
            "temperature": c.temperature,
            "max_tokens": c.max_tokens,
            "timestamp": c.call_timestamp,
        })
    for c in tracker.embedding_calls:
        llm_calls.append({
            "model": c.get("model", "embedding"),
            "layer": "embedding",
            "method": "embed",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_ms": c.get("latency_ms", 0),
            "success": c.get("success", False),
            "error": c.get("error", ""),
            "prompt_preview": f"texts_count={c.get('texts_count')}, total_chars={c.get('total_chars')}",
            "timestamp": c.get("timestamp", 0),
        })
    for c in tracker.reranker_calls:
        llm_calls.append({
            "model": "reranker",
            "layer": "reranker",
            "method": "rerank",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_ms": c.get("latency_ms", 0),
            "success": c.get("success", False),
            "error": c.get("error", ""),
            "prompt_preview": f"candidates={c.get('candidates')}, top_k={c.get('top_k')}",
            "timestamp": c.get("timestamp", 0),
        })
    result.raw_llm_calls = llm_calls


# ═══════════════════════════════════════════════════════
# 7. 指标汇总
# ═══════════════════════════════════════════════════════

def _is_clarification_case(r: TurnResult) -> bool:
    """判断是否为标注了 clarification 的特殊边界用例"""
    return "clarification" in (r.gold_intents or [])


def compute_metrics(results: List[TurnResult]) -> Dict[str, Any]:
    total = len(results)
    if total == 0:
        return {}

    # ── 分离常规用例与 clarification 边界用例 ──
    normal_results = [r for r in results if not _is_clarification_case(r)]
    clarification_results = [r for r in results if _is_clarification_case(r)]
    normal_total = len(normal_results)
    clarification_total = len(clarification_results)

    # ── 常规用例指标（不含 clarification）──
    success_count = sum(1 for r in normal_results if r.task_success)
    exception_count = sum(1 for r in results if r.has_exception)
    latencies = [r.e2e_latency_ms for r in results]
    latencies_sorted = sorted(latencies)

    all_llm = [r.llm_summary for r in results if r.llm_summary]
    total_llm_calls = sum(s.get("total_calls", 0) for s in all_llm)
    total_tokens = sum(s.get("total_tokens", 0) for s in all_llm)
    total_cost = sum(s.get("estimated_cost_usd", 0) for s in all_llm)
    total_tool_calls = sum(len(r.executed_tools) for r in normal_results)

    # ── 严格意图命中指标（仅常规用例参与 rate 计算，但按意图类型仍统计全部）──
    intent_strict_hits = sum(1 for r in results if r.intent_strict_hit)
    intent_strict_rate_normal = sum(1 for r in normal_results if r.intent_strict_hit) / normal_total if normal_total else 0.0

    intent_by_gold = defaultdict(lambda: {"total": 0, "hit": 0})
    for r in results:
        for g in r.gold_intents:
            intent_by_gold[g]["total"] += 1
            if r.intent_strict_hit:
                intent_by_gold[g]["hit"] += 1

    # ── 工具调用指标（仅常规用例）──
    tool_primary_hits = sum(1 for r in normal_results if r.tool_primary_hit)
    tool_full_chain_hits = sum(1 for r in normal_results if r.tool_full_chain_hit)
    tool_primary_rate = tool_primary_hits / normal_total if normal_total else 0.0
    tool_full_chain_rate = tool_full_chain_hits / normal_total if normal_total else 0.0

    tool_by_expected = defaultdict(lambda: {"total": 0, "primary_hit": 0, "full_hit": 0})
    for r in normal_results:
        for t in r.expected_tools:
            tool_by_expected[t]["total"] += 1
            if r.tool_primary_hit:
                tool_by_expected[t]["primary_hit"] += 1
            if r.tool_full_chain_hit:
                tool_by_expected[t]["full_hit"] += 1

    # ── 过程质量指标（仅常规用例）──
    process_quality_ok = sum(1 for r in normal_results if r.process_quality.get("overall_match", False))
    process_quality_rate = process_quality_ok / normal_total if normal_total else 0.0
    avg_match_rate = sum(r.process_quality.get("match_rate", 0.0) for r in normal_results) / normal_total if normal_total else 0.0

    # ── 工具执行成功率（仅常规用例，clarification 无 task_graph）──
    tool_exec_total = 0
    tool_exec_success = 0
    for r in normal_results:
        if r.task_graph:
            for t in r.task_graph.get("tasks", {}).values():
                if t.get("tool_name"):
                    tool_exec_total += 1
                    if t.get("status") == "success":
                        tool_exec_success += 1
    tool_exec_rate = tool_exec_success / tool_exec_total if tool_exec_total else 0.0

    # ── 规划合理性（仅常规用例）──
    valid_plans = sum(1 for r in normal_results if r.task_graph and r.task_graph.get("global_status") != "failed")

    # ── KB 异常（全部统计，但 rate 基于常规用例）──
    kb_empty_count = sum(1 for r in normal_results if r.kb_empty)
    ext_search_count = sum(1 for r in normal_results if r.external_search_triggered)
    replan_t1_count = sum(1 for r in normal_results if r.replan_t1)
    replan_t2_count = sum(1 for r in normal_results if r.replan_t2)
    replan_t4_count = sum(1 for r in normal_results if r.replan_t4)
    intent_skip_count = sum(1 for r in normal_results if r.intent_skipped)

    # ── clarification 边界用例单独统计 ──
    clarification_stats = {
        "total": clarification_total,
        "correct_trigger": 0,   # gold=clarification 且系统触发
        "missed_trigger": 0,    # gold=clarification 但系统未触发
        "false_trigger": 0,     # gold≠clarification 但系统触发（从全部结果中统计）
    }
    for r in clarification_results:
        if r.intent_result and r.intent_result.get("needs_clarification"):
            clarification_stats["correct_trigger"] += 1
        else:
            clarification_stats["missed_trigger"] += 1
    # 误触发：从全部结果中找 gold 不含 clarification 但系统触发的
    for r in results:
        if not _is_clarification_case(r):
            if r.intent_result and r.intent_result.get("needs_clarification"):
                clarification_stats["false_trigger"] += 1

    return {
        "outcome": {
            "total_cases": total,
            "normal_cases": normal_total,
            "clarification_cases": clarification_total,
            "task_success_rate": round(success_count / normal_total, 4) if normal_total else 0.0,
            "exception_rate": round(exception_count / total, 4),
            "avg_latency_ms": round(sum(latencies) / total, 2),
            "p50_latency_ms": latencies_sorted[len(latencies_sorted) // 2] if latencies_sorted else 0,
            "p99_latency_ms": latencies_sorted[int(len(latencies_sorted) * 0.99)] if latencies_sorted else 0,
            "total_tool_calls": total_tool_calls,
            "avg_tool_calls_per_case": round(total_tool_calls / normal_total, 2) if normal_total else 0.0,
            "total_llm_api_calls": total_llm_calls,
            "total_tokens": total_tokens,
            "estimated_total_cost_usd": round(total_cost, 6),
            "avg_cost_per_case_usd": round(total_cost / total, 6) if total else 0,
        },
        "intent": {
            "strict_hit_count": sum(1 for r in normal_results if r.intent_strict_hit),
            "strict_hit_rate": round(intent_strict_rate_normal, 4),
            "by_intent_type": {
                k: {"total": v["total"], "hit": v["hit"], "rate": round(v["hit"] / v["total"], 4) if v["total"] else 0}
                for k, v in sorted(intent_by_gold.items())
            },
        },
        "tool": {
            "primary_hit_count": tool_primary_hits,
            "primary_hit_rate": round(tool_primary_rate, 4),
            "full_chain_hit_count": tool_full_chain_hits,
            "full_chain_hit_rate": round(tool_full_chain_rate, 4),
            "by_tool_type": {
                k: {"total": v["total"], "primary_hit": v["primary_hit"], "full_hit": v["full_hit"],
                    "primary_rate": round(v["primary_hit"] / v["total"], 4) if v["total"] else 0,
                    "full_rate": round(v["full_hit"] / v["total"], 4) if v["total"] else 0}
                for k, v in sorted(tool_by_expected.items())
            },
            "execution_success_rate": round(tool_exec_rate, 4),
        },
        "process_quality": {
            "overall_match_count": process_quality_ok,
            "overall_match_rate": round(process_quality_rate, 4),
            "avg_match_rate": round(avg_match_rate, 4),
        },
        "kb_anomaly": {
            "kb_empty_rate": round(kb_empty_count / normal_total, 4) if normal_total else 0.0,
            "external_search_trigger_rate": round(ext_search_count / normal_total, 4) if normal_total else 0.0,
            "replan_t1_rate": round(replan_t1_count / normal_total, 4) if normal_total else 0.0,
            "replan_t2_rate": round(replan_t2_count / normal_total, 4) if normal_total else 0.0,
            "replan_t4_rate": round(replan_t4_count / normal_total, 4) if normal_total else 0.0,
            "intent_timeout_skip_rate": round(intent_skip_count / normal_total, 4) if normal_total else 0.0,
        },
        "latency": {
            "query_rewrite_avg_ms": round(
                sum(r.rewrite.latency_ms for r in results if r.rewrite) /
                max(1, sum(1 for r in results if r.rewrite)), 2),
            "intent_recognition_avg_ms": round(
                sum(r.intent.latency_ms for r in results if r.intent) /
                max(1, sum(1 for r in results if r.intent)), 2),
            "planner_avg_ms": round(
                sum(r.planner.latency_ms for r in normal_results if r.planner) /
                max(1, sum(1 for r in normal_results if r.planner)), 2),
            "executor_avg_ms": round(
                sum(r.executor.latency_ms for r in normal_results if r.executor) /
                max(1, sum(1 for r in normal_results if r.executor)), 2),
        },
        "clarification": clarification_stats,
        "batch": _compute_batch_metrics(results),
    }


def _compute_batch_metrics(results: List[TurnResult]) -> Dict[str, Dict]:
    by_batch = defaultdict(list)
    for r in results:
        by_batch[r.batch].append(r)
    metrics = {}
    for batch, rs in sorted(by_batch.items()):
        normal_rs = [r for r in rs if not _is_clarification_case(r)]
        clar_rs = [r for r in rs if _is_clarification_case(r)]
        normal_total = len(normal_rs)
        metrics[batch] = {
            "total": len(rs),
            "normal_cases": normal_total,
            "clarification_cases": len(clar_rs),
            "success": sum(1 for r in normal_rs if r.task_success),
            "success_rate": round(sum(1 for r in normal_rs if r.task_success) / normal_total, 4) if normal_total else 0,
            "intent_strict_hit": sum(1 for r in normal_rs if r.intent_strict_hit),
            "intent_strict_rate": round(sum(1 for r in normal_rs if r.intent_strict_hit) / normal_total, 4) if normal_total else 0,
            "tool_primary_hit": sum(1 for r in normal_rs if r.tool_primary_hit),
            "tool_primary_rate": round(sum(1 for r in normal_rs if r.tool_primary_hit) / normal_total, 4) if normal_total else 0,
            "avg_latency_ms": round(sum(r.e2e_latency_ms for r in rs) / len(rs), 2) if rs else 0,
        }
        # 批次内的 clarification 统计
        if clar_rs:
            metrics[batch]["clarification"] = {
                "total": len(clar_rs),
                "correct_trigger": sum(1 for r in clar_rs if r.intent_result and r.intent_result.get("needs_clarification")),
                "missed_trigger": sum(1 for r in clar_rs if not (r.intent_result and r.intent_result.get("needs_clarification"))),
            }
    return metrics


# ═══════════════════════════════════════════════════════
# 8. 稳定性测试
# ═══════════════════════════════════════════════════════

async def run_stability_test(
    case: dict,
    resume_text: str,
    tracker: LLMTracker,
    n_runs: int,
) -> Dict[str, Any]:
    results: List[TurnResult] = []
    for i in range(n_runs):
        r = await run_single_case(case, resume_text, tracker, reset_session=True)
        results.append(r)
        await asyncio.sleep(0.5)

    success_runs = [r for r in results if r.task_success]
    success_rate = len(success_runs) / n_runs

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
                "intent_strict_hit": r.intent_strict_hit,
                "tool_primary_hit": r.tool_primary_hit,
                "latency_ms": round(r.e2e_latency_ms, 2),
                "executed_tools": r.executed_tools,
                "llm_calls": r.llm_summary.get("total_calls", 0),
            }
            for i, r in enumerate(results)
        ],
    }


# ═══════════════════════════════════════════════════════
# 9. 报告输出
# ═══════════════════════════════════════════════════════

def print_report(metrics: Dict, results: Optional[List[TurnResult]] = None, stability: Optional[List[Dict]] = None):
    print("\n" + "=" * 70)
    print("  v3 ReAct Agent 综合评测报告（严格对齐版）")
    print("=" * 70)

    out = metrics.get("outcome", {})
    normal_total = out.get("normal_cases", 0)
    clar_total = out.get("clarification_cases", 0)
    total_cases = out.get("total_cases", 0)

    print(f"\n【结果指标】")
    print(f"  总用例数:          {total_cases} (常规={normal_total}, 澄清边界={clar_total})")
    print(f"  任务成功率:        {out.get('task_success_rate', 0) * 100:.2f}% (仅常规用例)")
    print(f"  异常率:            {out.get('exception_rate', 0) * 100:.2f}%")
    print(f"  平均延迟:          {out.get('avg_latency_ms', 0):.0f} ms")
    print(f"  P99 延迟:          {out.get('p99_latency_ms', 0):.0f} ms")
    print(f"  总工具调用:        {out.get('total_tool_calls', 0)} (仅常规用例)")
    print(f"  每例平均工具调用:  {out.get('avg_tool_calls_per_case', 0):.2f} (仅常规用例)")
    print(f"  总 LLM API 调用:   {out.get('total_llm_api_calls', 0)}")
    print(f"  总 Token 消耗:     {out.get('total_tokens', 0):,}")
    print(f"  预估总成本:        ${out.get('estimated_total_cost_usd', 0):.6f}")
    print(f"  每例平均成本:      ${out.get('avg_cost_per_case_usd', 0):.6f}")

    # ── clarification 边界用例单独统计 ──
    clar = metrics.get("clarification", {})
    if clar.get("total", 0) > 0:
        print(f"\n【澄清边界用例统计（单独说明）】")
        print(f"  总数:              {clar.get('total', 0)}")
        print(f"  正确触发:          {clar.get('correct_trigger', 0)} (gold=clarification 且系统触发)")
        print(f"  漏触发:            {clar.get('missed_trigger', 0)} (gold=clarification 但系统未触发)")
        print(f"  误触发:            {clar.get('false_trigger', 0)} (gold≠clarification 但系统触发)")
        print(f"  识别准确率:        {clar.get('correct_trigger', 0) / clar.get('total', 1) * 100:.1f}%")

    intent = metrics.get("intent", {})
    print(f"\n【意图识别指标（严格判定）】")
    print(f"  严格命中数:        {intent.get('strict_hit_count', 0)}/{normal_total} (仅常规用例)")
    print(f"  严格命中率:        {intent.get('strict_hit_rate', 0) * 100:.2f}%")
    print(f"  （规则：多意图必须全部命中，澄清场景按预期判断）")
    for itype, im in sorted(intent.get("by_intent_type", {}).items()):
        print(f"    [{itype}] {im['hit']}/{im['total']} ({im['rate'] * 100:.1f}%)")

    tool = metrics.get("tool", {})
    print(f"\n【工具调用指标】")
    print(f"  主要工具命中数:    {tool.get('primary_hit_count', 0)}/{normal_total} (仅常规用例)")
    print(f"  主要工具命中率:    {tool.get('primary_hit_rate', 0) * 100:.2f}%")
    print(f"  （规则：命中至少一个主要工具即算正确，考虑 kb 覆盖动态调整）")
    print(f"  全工具链命中数:    {tool.get('full_chain_hit_count', 0)}/{normal_total} (仅常规用例)")
    print(f"  全工具链命中率:    {tool.get('full_chain_hit_rate', 0) * 100:.2f}%")
    print(f"  （辅助指标：所有主要工具都被执行）")
    for ttype, tm in sorted(tool.get("by_tool_type", {}).items()):
        print(f"    [{ttype}] 主要命中={tm['primary_hit']}/{tm['total']} ({tm['primary_rate'] * 100:.1f}%), 全链命中={tm['full_hit']}/{tm['total']} ({tm['full_rate'] * 100:.1f}%)")
    print(f"  工具执行成功率:    {tool.get('execution_success_rate', 0) * 100:.2f}%")

    pq = metrics.get("process_quality", {})
    print(f"\n【过程质量评估（小模型比对）】")
    print(f"  整体符合数:        {pq.get('overall_match_count', 0)}/{normal_total} (仅常规用例)")
    print(f"  整体符合率:        {pq.get('overall_match_rate', 0) * 100:.2f}%")
    print(f"  平均符合率:        {pq.get('avg_match_rate', 0) * 100:.2f}%")

    kb = metrics.get("kb_anomaly", {})
    print(f"\n【KB 异常指标】")
    print(f"  KB 无命中率:         {kb.get('kb_empty_rate', 0) * 100:.2f}%")
    print(f"  外部搜索触发率:      {kb.get('external_search_trigger_rate', 0) * 100:.2f}%")
    print(f"  Replan T1 率:        {kb.get('replan_t1_rate', 0) * 100:.2f}%")
    print(f"  Replan T2 率:        {kb.get('replan_t2_rate', 0) * 100:.2f}%")
    print(f"  Replan T4 率:        {kb.get('replan_t4_rate', 0) * 100:.2f}%")
    print(f"  意图超时跳过率:      {kb.get('intent_timeout_skip_rate', 0) * 100:.2f}%")

    lat = metrics.get("latency", {})
    print(f"\n【组件平均延迟】")
    print(f"  QueryRewrite:      {lat.get('query_rewrite_avg_ms', 0):.0f} ms")
    print(f"  IntentRecognition: {lat.get('intent_recognition_avg_ms', 0):.0f} ms")
    print(f"  Planner:           {lat.get('planner_avg_ms', 0):.0f} ms")
    print(f"  Executor:          {lat.get('executor_avg_ms', 0):.0f} ms")

    print(f"\n【批次统计】")
    for batch, bm in sorted(metrics.get("batch", {}).items()):
        print(f"  [{batch.upper()}] {bm['success']}/{bm['total']} 成功 ({bm['success_rate'] * 100:.1f}%) | "
              f"意图严格={bm['intent_strict_hit']}/{bm['total']} ({bm['intent_strict_rate'] * 100:.1f}%) | "
              f"工具主要={bm['tool_primary_hit']}/{bm['total']} ({bm['tool_primary_rate'] * 100:.1f}%) | "
              f"avg_lat={bm['avg_latency_ms']:.0f}ms")

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


# ═══════════════════════════════════════════════════════
# 10. 数据加载
# ═══════════════════════════════════════════════════════

def load_dataset(batch: Optional[str] = None, case_id: Optional[str] = None, dataset_file: Optional[Path] = None) -> List[dict]:
    if dataset_file is None:
        dataset_file = EVAL_DIR / "test_dataset.jsonl"
    cases = []
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    if case_id:
        case_ids = [cid.strip() for cid in case_id.split(",")]
        cases = [c for c in cases if c["session_id"] in case_ids]
    elif batch:
        batches = [b.strip().lower() for b in batch.split(",")]
        cases = [c for c in cases if any(c["session_id"].startswith(f"eval_{b}_") for b in batches)]
    return cases


def load_resumes() -> dict:
    resumes_file = EVAL_DIR / "test_resumes.json"
    with open(resumes_file, "r", encoding="utf-8") as f:
        return json.load(f)


def get_resume_for_case(case: dict, resumes: dict) -> str:
    # 优先使用 test_dataset.jsonl 中的 resume_id 字段
    resume_id = case.get("resume_id")
    if not resume_id:
        # fallback：兼容旧版 session_resume_map
        mapping = resumes.get("session_resume_map", {})
        resume_id = mapping.get(case.get("session_id", ""), "")
    for r in resumes.get("resumes", []):
        if r.get("id") == resume_id:
            return r.get("text", "")
    return ""


# ═══════════════════════════════════════════════════════
# 11. 主函数
# ═══════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(description="v3 ReAct Agent 严格对齐评测")
    parser.add_argument("--batch", default=None, help="指定批次，如 A 或 A,C")
    parser.add_argument("--case", default=None, help="指定单条 case_id")
    parser.add_argument("--stability", type=int, default=0, help="稳定性测试：每条跑 N 次")
    parser.add_argument("--output", default=None, help="输出 JSON 报告路径")
    parser.add_argument("--workers", type=int, default=1, help="并发数（默认1）")
    parser.add_argument("--dataset", default=None, help="指定数据集文件路径（默认 test_dataset.jsonl）")
    args = parser.parse_args()

    dataset_path = Path(args.dataset) if args.dataset else None
    cases = load_dataset(batch=args.batch, case_id=args.case, dataset_file=dataset_path)
    resumes = load_resumes()

    print(f"\n加载 {len(cases)} 条测试用例")
    print(f"稳定性模式: {'是 (每条跑 ' + str(args.stability) + ' 次)' if args.stability > 1 else '否'}")

    tracker = LLMTracker()
    tracker.install()

    try:
        all_results: List[TurnResult] = []
        stability_results: List[Dict] = []
        group_session_map: Dict[str, str] = {}  # group -> session_id

        for i, case in enumerate(cases, 1):
            sid = case["session_id"]
            group = case.get("session_group")
            resume_text = get_resume_for_case(case, resumes)
            group_tag = f"[{group}] " if group else ""
            print(f"\n[{i}/{len(cases)}] {group_tag}{sid}: {case['message'][:40]}...")

            # 单轮始终重置；多轮第一轮重置，后续复用
            reset_session = True
            session = None
            if group:
                if group in group_session_map:
                    reset_session = False
                    sid = group_session_map[group]
                    session = SessionMemory(session_id=sid)
                    print(f"  (复用 session: group={group} sid={sid})")
                else:
                    group_session_map[group] = sid
                    session = SessionMemory(session_id=sid)

            try:
                if args.stability > 1:
                    st = await run_stability_test(case, resume_text, tracker, args.stability)
                    stability_results.append(st)
                    r = await run_single_case(case, resume_text, tracker, session, reset_session=reset_session)
                elif group:
                    # 多轮对话：走 HTTP，保持服务端 session 状态
                    r = await run_single_case(case, resume_text, tracker, session, reset_session=reset_session)
                else:
                    # 单轮对话：本地直接调用，跳过 HTTP 开销
                    r = await run_single_case_direct(case, resume_text, tracker, reset_session=reset_session)
            except TokenExhaustedError as e:
                print(f"\n  [!!] {e}")
                print("  检测到 Token/配额/余额耗尽，测试已中断！")
                break

            all_results.append(r)
            status = "[OK]" if r.task_success else "[XX]"
            exc = " [异常]" if r.has_exception else ""
            intent_hit = " I" if r.intent_strict_hit else " i"
            tool_hit = " T" if r.tool_primary_hit else " t"
            print(f"  {status} success={r.task_success}{exc} intent_strict={r.intent_strict_hit} tool_primary={r.tool_primary_hit} lat={r.e2e_latency_ms:.0f}ms")

            # 【关键】每跑完一条就追加保存，防止超时丢失
            case_record = {
                "case_id": r.case_id,
                "batch": r.batch,
                "session_id": r.session_id,
                "turn_id": r.turn_id,
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
                "intent_strict_hit": r.intent_strict_hit,
                "intent_hit_reason": r.intent_hit_reason,
                "tool_primary_hit": r.tool_primary_hit,
                "tool_primary_tools": r.tool_primary_tools,
                "tool_full_chain_hit": r.tool_full_chain_hit,
                "tool_hit_reason": r.tool_hit_reason,
                "process_quality": r.process_quality,
                "resume_text": r.resume_text,
                "raw_llm_calls": r.raw_llm_calls,
                "tool_executions_full": r.tool_executions_full,
                "session_history": r.session_history,
                "final_response": r.final_response,
                "judge_result": r.judge_result,
                "embedding_calls": r.embedding_calls,
                "reranker_calls": r.reranker_calls,
            }
            # 追加写入 cases.jsonl（断点续跑的关键）
            cases_jsonl_path = (Path(args.output).parent / "cases.jsonl") if args.output else (EVAL_DIR / f"v3_eval_cases_{int(time.time())}.jsonl")
            cases_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cases_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(case_record, ensure_ascii=False) + "\n")

        metrics = compute_metrics(all_results)
        print_report(metrics, all_results, stability_results if args.stability > 1 else None)

        output_data = {
            "metrics": metrics,
            "cases": [
                {
                    "case_id": r.case_id,
                    "batch": r.batch,
                    "session_id": r.session_id,
                    "turn_id": r.turn_id,
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
                    "intent_strict_hit": r.intent_strict_hit,
                    "intent_hit_reason": r.intent_hit_reason,
                    "tool_primary_hit": r.tool_primary_hit,
                    "tool_primary_tools": r.tool_primary_tools,
                    "tool_full_chain_hit": r.tool_full_chain_hit,
                    "tool_hit_reason": r.tool_hit_reason,
                    "process_quality": r.process_quality,
                    "resume_text": r.resume_text,
                    "raw_llm_calls": r.raw_llm_calls,
                    "tool_executions_full": r.tool_executions_full,
                    "session_history": r.session_history,
                    "final_response": r.final_response,
                    "judge_result": r.judge_result,
                    "embedding_calls": r.embedding_calls,
                    "reranker_calls": r.reranker_calls,
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
            print(f"逐条结果已追加: {cases_jsonl_path}")
        else:
            default_out = EVAL_DIR / f"v3_eval_report_{int(time.time())}.json"
            with open(default_out, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"\n报告已保存: {default_out}")
            print(f"逐条结果已追加: {cases_jsonl_path}")

    finally:
        tracker.uninstall()


if __name__ == "__main__":
    asyncio.run(main())
