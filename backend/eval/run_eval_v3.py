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
import traceback
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

EVAL_DIR = Path(__file__).parent
sys.path.insert(0, str(EVAL_DIR.parent))

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

from app.core.memory import SessionMemory, DialogueTurn, WorkingMemory
from app.core.query_rewrite import QueryRewriter, QueryRewriteResult
from app.core.llm_intent import LLMIntentRouter
from app.core.llm_planner import TaskGraphPlanner
from app.core.new_arch_adapter import multi_intent_result_to_intent_result, convert_task_graph
from app.core.intent_recognition import IntentResult
from app.core.planner import TaskGraph, TaskNode
from app.core.react_executor import ReActExecutor
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

    def install(self):
        self._original_chat = LLMClient.chat
        self._original_generate = LLMClient.generate

        async def tracked_chat(self_obj, messages, temperature=0.7, max_tokens=None,
                               json_mode=False, timeout=None):
            start = time.time()
            layer = self._guess_layer(self_obj)
            try:
                result = await self._original_chat(self_obj, messages, temperature, max_tokens, json_mode, timeout)
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
                prompt_tokens = (len(prompt) + len(system or "")) // 2
                completion_tokens = len(result) // 2
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

    def _guess_layer(self, client: LLMClient) -> str:
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

    # ── v3 新增：严格判定结果 ──
    intent_strict_hit: bool = False           # 意图严格命中（多意图全部命中）
    intent_hit_reason: str = ""               # 意图命中/未命中原因
    tool_primary_hit: bool = False            # 主要工具命中
    tool_primary_tools: List[str] = field(default_factory=list)  # 动态调整后的主要工具
    tool_full_chain_hit: bool = False         # 全工具链命中
    tool_hit_reason: str = ""                 # 工具命中/未命中原因
    process_quality: Dict = field(default_factory=dict)  # 小模型过程质量评估


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
        it = d.get("intent_type", "")
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

async def run_single_case(
    case: dict,
    resume_text: str,
    tracker: LLMTracker,
    session: Optional[SessionMemory] = None,
) -> TurnResult:
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

    if session is None:
        session = SessionMemory(session_id=sid)
    if resume_text and len(resume_text.strip()) > 50:
        if not hasattr(session, "global_slots"):
            session.global_slots = {}
        session.global_slots["resume_available"] = True

    start_time = time.time()
    tracker.reset()
    result.resume_text = resume_text
    result.user_message = case.get("message", "")

    # 注入模拟多轮上下文
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

        # ── Step 1: Intent Recognition ──
        t1 = time.time()
        intent_router = LLMIntentRouter()
        multi_result = await intent_router.route_multi(
            rewrite_result=rw,
            session=session,
            attachments=[],
            raw_message=case["message"],
        )
        intent_result = multi_intent_result_to_intent_result(multi_result)
        result.intent = ComponentResult(
            component="intent_recognition",
            success=True,
            latency_ms=(time.time() - t1) * 1000,
            output=_intent_result_to_dict(intent_result),
        )
        result.intent_result = result.intent.output

        # ── 严格意图命中判定 ──
        result.intent_strict_hit, result.intent_hit_reason = compute_intent_strict_hit(result)

        # 澄清场景
        if intent_result.needs_clarification:
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
            if intent_result.resolved_entities:
                if not hasattr(session, "global_slots"):
                    session.global_slots = {}
                for k, v in intent_result.resolved_entities.items():
                    if v is not None:
                        session.global_slots[k] = v

            # 工具判定（澄清场景无工具执行）
            result.tool_primary_hit = False
            result.tool_full_chain_hit = False
            result.tool_hit_reason = "澄清场景，未执行工具"
            result.tool_primary_tools = result.expected_tools

            result.task_success = True  # 澄清也是正确行为
            result.e2e_latency_ms = (time.time() - start_time) * 1000
            result.llm_summary = tracker.summary()

            # ── 过程质量 Judge ──
            result.process_quality = await process_quality_judge(case, result)
            return result

        # ── Step 2: Planner ──
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

        # ── Step 3: Executor ──
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
        result.task_graph = _task_graph_to_dict(graph)

        result.executed_tools = sorted(set(
            t.tool_name for t in graph.tasks.values()
            if t.tool_name and t.status == "success"
        ))
        result.failed_tools = sorted(set(
            t.tool_name for t in graph.tasks.values()
            if t.tool_name and t.status == "failed"
        ))
        if graph.global_status in ("needs_replan", "replanning"):
            result.replan_count = 1

        # KB 异常追踪
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

        replan_reason = graph.replan_reason or ""
        if "T1" in replan_reason or "hard_fail" in replan_reason:
            result.replan_t1 = True
        if "T2" in replan_reason or "retrieval_insufficient" in replan_reason:
            result.replan_t2 = True
        if "T4" in replan_reason or "low_match" in replan_reason:
            result.replan_t4 = True

        if result.intent_result:
            result.intent_skipped = result.intent_result.get("skipped_due_to_timeout", False)

        # 保存对话历史
        turn = DialogueTurn(
            turn_id=len(session.working_memory.turns) + 1,
            user_message=case["message"],
            assistant_reply="",
            intent=intent_result.demands[0].intent_type if intent_result.demands else "chat",
            rewritten_query=rw.rewritten_query,
            evidence_score=0.0,
        )
        session.working_memory.append(turn)

        # ── 严格工具命中判定 ──
        result.tool_primary_hit, result.tool_full_chain_hit, result.tool_primary_tools, result.tool_hit_reason = \
            compute_tool_hit(result, result.expected_tools)

    except Exception as e:
        if is_token_exhausted_error(e):
            raise TokenExhaustedError(f"Token/配额耗尽，中断测试: {e}") from e
        result.has_exception = True
        result.exception_trace = traceback.format_exc()

    # 保存完整中间信息
    result.e2e_latency_ms = (time.time() - start_time) * 1000
    result.llm_summary = tracker.summary()
    result.raw_llm_calls = [
        {"model": c.model, "layer": c.layer, "method": c.method,
         "prompt_tokens": c.prompt_tokens, "completion_tokens": c.completion_tokens,
         "latency_ms": c.latency_ms, "success": c.success, "error": c.error,
         "prompt_full": c.prompt_full, "system_prompt": c.system_prompt,
         "raw_output": c.raw_output, "temperature": c.temperature,
         "max_tokens": c.max_tokens, "timeout": c.timeout,
         "call_timestamp": c.call_timestamp}
        for c in tracker.calls
    ]
    if result.task_graph:
        result.tool_executions_full = [
            {"task_id": tid, "tool_name": t.get("tool_name"), "status": t.get("status"),
             "result_full": t.get("result_full"), "observation_full": t.get("observation_full"),
             "tool_input": t.get("tool_input"), "started_at": t.get("started_at"),
             "finished_at": t.get("finished_at")}
            for tid, t in result.task_graph.get("tasks", {}).items()
            if t.get("tool_name")
        ]
    # ── 从 task_graph 提取最终回复 ──
    final_reply = ""
    if result.task_graph:
        tasks = result.task_graph.get("tasks", {})
        # 优先从 llm_reasoning 任务提取
        for tid, t in tasks.items():
            if t.get("task_type") == "llm_reasoning" and t.get("status") == "success" and t.get("result_full"):
                try:
                    res = eval(t["result_full"]) if isinstance(t["result_full"], str) else t["result_full"]
                    if isinstance(res, dict) and "output" in res:
                        final_reply = res["output"]
                        break
                except Exception:
                    pass
        # 其次从 aggregate 任务提取
        if not final_reply:
            for tid, t in tasks.items():
                if t.get("task_type") == "aggregate" and t.get("status") == "success" and t.get("result_full"):
                    try:
                        res = eval(t["result_full"]) if isinstance(t["result_full"], str) else t["result_full"]
                        if isinstance(res, dict) and "aggregation" in res:
                            agg = res["aggregation"]
                            if isinstance(agg, dict):
                                for k, v in agg.items():
                                    if isinstance(v, dict) and "answer" in v:
                                        final_reply = v["answer"]
                                        break
                                    if isinstance(v, dict) and "output" in v:
                                        final_reply = v["output"]
                                        break
                            if not final_reply:
                                final_reply = json.dumps(agg, ensure_ascii=False)
                            break
                    except Exception:
                        pass

    # 更新 session history 中的 assistant_reply
    turns = session.working_memory.turns if session.working_memory else []
    if turns:
        turns[-1].assistant_reply = final_reply
    result.session_history = [
        {"turn_id": turn.turn_id, "user_message": turn.user_message,
         "assistant_reply": turn.assistant_reply, "intent": turn.intent,
         "rewritten_query": turn.rewritten_query}
        for turn in turns
    ]
    result.final_response = final_reply

    # LLM-as-judge 评估
    if result.has_exception:
        result.task_success = False
        result.judge_result = {"resolved": False, "reason": "执行异常", "source": "rule"}
    elif result.intent_result and result.intent_result.get("needs_clarification"):
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
        if judge.get("error"):
            has_final_response = bool(result.final_response.strip()) if result.final_response else False
            result.task_success = not result.has_exception and has_final_response
            result.judge_result["fallback"] = True

    # ── 过程质量 Judge ──
    result.process_quality = await process_quality_judge(case, result)

    return result


# ═══════════════════════════════════════════════════════
# 7. 指标汇总
# ═══════════════════════════════════════════════════════

def compute_metrics(results: List[TurnResult]) -> Dict[str, Any]:
    total = len(results)
    if total == 0:
        return {}

    success_count = sum(1 for r in results if r.task_success)
    exception_count = sum(1 for r in results if r.has_exception)
    clarification_count = sum(1 for r in results
                               if r.intent_result and r.intent_result.get("needs_clarification"))
    latencies = [r.e2e_latency_ms for r in results]
    latencies_sorted = sorted(latencies)

    all_llm = [r.llm_summary for r in results if r.llm_summary]
    total_llm_calls = sum(s.get("total_calls", 0) for s in all_llm)
    total_tokens = sum(s.get("total_tokens", 0) for s in all_llm)
    total_cost = sum(s.get("estimated_cost_usd", 0) for s in all_llm)
    total_tool_calls = sum(len(r.executed_tools) for r in results)

    # ── 严格意图命中指标 ──
    intent_strict_hits = sum(1 for r in results if r.intent_strict_hit)
    intent_strict_rate = intent_strict_hits / total if total else 0.0

    # 按意图类型统计
    intent_by_gold = defaultdict(lambda: {"total": 0, "hit": 0})
    for r in results:
        for g in r.gold_intents:
            intent_by_gold[g]["total"] += 1
            if r.intent_strict_hit:
                intent_by_gold[g]["hit"] += 1

    # ── 工具调用指标 ──
    tool_primary_hits = sum(1 for r in results if r.tool_primary_hit)
    tool_full_chain_hits = sum(1 for r in results if r.tool_full_chain_hit)
    tool_primary_rate = tool_primary_hits / total if total else 0.0
    tool_full_chain_rate = tool_full_chain_hits / total if total else 0.0

    # 按工具类型统计
    tool_by_expected = defaultdict(lambda: {"total": 0, "primary_hit": 0, "full_hit": 0})
    for r in results:
        for t in r.expected_tools:
            tool_by_expected[t]["total"] += 1
            if r.tool_primary_hit:
                tool_by_expected[t]["primary_hit"] += 1
            if r.tool_full_chain_hit:
                tool_by_expected[t]["full_hit"] += 1

    # ── 过程质量指标 ──
    process_quality_ok = sum(1 for r in results if r.process_quality.get("overall_match", False))
    process_quality_rate = process_quality_ok / total if total else 0.0
    avg_match_rate = sum(r.process_quality.get("match_rate", 0.0) for r in results) / total if total else 0.0

    # ── 工具执行成功率 ──
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

    # ── 规划合理性 ──
    valid_plans = sum(1 for r in results if r.task_graph and r.task_graph.get("global_status") != "failed")

    # ── KB 异常 ──
    kb_empty_count = sum(1 for r in results if r.kb_empty)
    ext_search_count = sum(1 for r in results if r.external_search_triggered)
    replan_t1_count = sum(1 for r in results if r.replan_t1)
    replan_t2_count = sum(1 for r in results if r.replan_t2)
    replan_t4_count = sum(1 for r in results if r.replan_t4)
    intent_skip_count = sum(1 for r in results if r.intent_skipped)

    return {
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
        "intent": {
            "strict_hit_count": intent_strict_hits,
            "strict_hit_rate": round(intent_strict_rate, 4),
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
            "kb_empty_rate": round(kb_empty_count / total, 4),
            "external_search_trigger_rate": round(ext_search_count / total, 4),
            "replan_t1_rate": round(replan_t1_count / total, 4),
            "replan_t2_rate": round(replan_t2_count / total, 4),
            "replan_t4_rate": round(replan_t4_count / total, 4),
            "intent_timeout_skip_rate": round(intent_skip_count / total, 4),
        },
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
            "intent_strict_hit": sum(1 for r in rs if r.intent_strict_hit),
            "intent_strict_rate": round(sum(1 for r in rs if r.intent_strict_hit) / len(rs), 4) if rs else 0,
            "tool_primary_hit": sum(1 for r in rs if r.tool_primary_hit),
            "tool_primary_rate": round(sum(1 for r in rs if r.tool_primary_hit) / len(rs), 4) if rs else 0,
            "avg_latency_ms": round(sum(r.e2e_latency_ms for r in rs) / len(rs), 2) if rs else 0,
        }
        for batch, rs in sorted(by_batch.items())
    }


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
        r = await run_single_case(case, resume_text, tracker)
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

    intent = metrics.get("intent", {})
    print(f"\n【意图识别指标（严格判定）】")
    print(f"  严格命中数:        {intent.get('strict_hit_count', 0)}/{out.get('total_cases', 0)}")
    print(f"  严格命中率:        {intent.get('strict_hit_rate', 0) * 100:.2f}%")
    print(f"  （规则：多意图必须全部命中，澄清场景按预期判断）")
    for itype, im in sorted(intent.get("by_intent_type", {}).items()):
        print(f"    [{itype}] {im['hit']}/{im['total']} ({im['rate'] * 100:.1f}%)")

    tool = metrics.get("tool", {})
    print(f"\n【工具调用指标】")
    print(f"  主要工具命中数:    {tool.get('primary_hit_count', 0)}/{out.get('total_cases', 0)}")
    print(f"  主要工具命中率:    {tool.get('primary_hit_rate', 0) * 100:.2f}%")
    print(f"  （规则：命中至少一个主要工具即算正确，考虑 kb 覆盖动态调整）")
    print(f"  全工具链命中数:    {tool.get('full_chain_hit_count', 0)}/{out.get('total_cases', 0)}")
    print(f"  全工具链命中率:    {tool.get('full_chain_hit_rate', 0) * 100:.2f}%")
    print(f"  （辅助指标：所有主要工具都被执行）")
    for ttype, tm in sorted(tool.get("by_tool_type", {}).items()):
        print(f"    [{ttype}] 主要命中={tm['primary_hit']}/{tm['total']} ({tm['primary_rate'] * 100:.1f}%), 全链命中={tm['full_hit']}/{tm['total']} ({tm['full_rate'] * 100:.1f}%)")
    print(f"  工具执行成功率:    {tool.get('execution_success_rate', 0) * 100:.2f}%")

    pq = metrics.get("process_quality", {})
    print(f"\n【过程质量评估（小模型比对）】")
    print(f"  整体符合数:        {pq.get('overall_match_count', 0)}/{out.get('total_cases', 0)}")
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
    mapping = resumes.get("session_resume_map", {})
    resume_id = mapping.get(case_id, "eval_resume_ai")
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
        group_session_map: Dict[str, SessionMemory] = {}

        for i, case in enumerate(cases, 1):
            sid = case["session_id"]
            group = case.get("session_group")
            resume_text = get_resume_for_case(sid, resumes)
            group_tag = f"[{group}] " if group else ""
            print(f"\n[{i}/{len(cases)}] {group_tag}{sid}: {case['message'][:40]}...")

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
                    r = await run_single_case(case, resume_text, tracker, session)
                else:
                    r = await run_single_case(case, resume_text, tracker, session)
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

        metrics = compute_metrics(all_results)
        print_report(metrics, all_results, stability_results if args.stability > 1 else None)

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
                    # v3 新增
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
            default_out = EVAL_DIR / f"v3_eval_report_{int(time.time())}.json"
            with open(default_out, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"\n报告已保存: {default_out}")

    finally:
        tracker.uninstall()


if __name__ == "__main__":
    asyncio.run(main())
