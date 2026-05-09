"""
新体系适配层 —— 将 llm_intent.py + llm_planner.py 的输出适配为旧体系格式

职责：
1. 意图名称映射（新体系小写 → 旧体系带前缀）
2. MultiIntentResult → IntentResult 转换
3. task_graph.TaskGraph → planner.TaskGraph 转换

目标：让 chat.py 在不改变 ReActExecutor / 评测脚本的前提下接入新体系
"""

import logging
from typing import List, Dict, Any, Optional

from app.core.intent_recognition import IntentResult, Demand
from app.core.llm_intent import MultiIntentResult, LLMIntentType, IntentCandidate

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# 1. 意图名称映射
# ═══════════════════════════════════════════════════════

LLM_TO_OLD_INTENT_MAP: Dict[str, str] = {
    "explore": "position_explore",
    "assess": "match_assess",
    "verify": "attribute_verify",
    "prepare": "interview_prepare",
    "manage": "resume_manage",
    "chat": "general_chat",
}

OLD_TO_LLM_INTENT_MAP: Dict[str, str] = {v: k for k, v in LLM_TO_OLD_INTENT_MAP.items()}


def map_intent_name_to_old(llm_intent: str) -> str:
    """新体系意图名 → 旧体系意图名"""
    return LLM_TO_OLD_INTENT_MAP.get(llm_intent, llm_intent)


def map_intent_name_to_new(old_intent: str) -> str:
    """旧体系意图名 → 新体系意图名"""
    return OLD_TO_LLM_INTENT_MAP.get(old_intent, old_intent)


# ═══════════════════════════════════════════════════════
# 2. MultiIntentResult → IntentResult 转换
# ═══════════════════════════════════════════════════════

def multi_intent_result_to_intent_result(multi: MultiIntentResult) -> IntentResult:
    """
    将 llm_intent.py 的 MultiIntentResult 转换为 intent_recognition.py 的 IntentResult。

    用于：chat.py 中 clarification 判断、planner fallback、SSE 输出、评测埋点等
    仍需要旧 IntentResult 格式的场景。
    """
    demands: List[Demand] = []
    missing_entities: List[str] = []

    for idx, cand in enumerate(multi.candidates):
        demands.append(
            Demand(
                intent_type=map_intent_name_to_old(cand.intent_type.value),
                entities=cand.slots or {},
                confidence=cand.confidence,
                priority=1 if cand.intent_type == multi.primary_intent else 2,
            )
        )
        if cand.missing_slots:
            missing_entities.extend(cand.missing_slots)

    # 去重 missing_entities
    missing_entities = list(set(missing_entities))

    return IntentResult(
        demands=demands,
        resolved_entities=multi.global_slots or {},
        is_complete=not multi.needs_clarification,
        needs_clarification=multi.needs_clarification,
        clarification_question=multi.clarification_reason or "",
        missing_entities=missing_entities,
        raw_intent_text=", ".join(
            map_intent_name_to_old(c.intent_type.value) for c in multi.candidates
        ),
    )


# ═══════════════════════════════════════════════════════
# 3. task_graph.TaskGraph → planner.TaskGraph 转换
# ═══════════════════════════════════════════════════════

def convert_task_graph(new_graph: "NewTaskGraph") -> "OldTaskGraph":
    """
    将 llm_planner.py 输出的 task_graph.TaskGraph 转换为 planner.py 的 TaskGraph。

    注意：
    - resolved_params 在旧体系中由 planner._fill_static_slots 预填充，
      新体系中不存在该步骤，因此转换后 resolved_params 为空，
      由 ReActExecutor 在运行时通过 _resolve_dynamic_params 动态解析。
    - fallback / params_hash / reused_from_history 等新字段在旧体系中不存储，
      因此会丢失（ReActExecutor 不依赖它们）。
    """
    from app.core.planner import TaskGraph as OldTaskGraph, TaskNode as OldTaskNode

    old_graph = OldTaskGraph()
    for t in new_graph.tasks:
        old_task = OldTaskNode(
            task_id=t.task_id,
            task_type=t.task_type,
            tool_name=t.tool_name,
            description=t.description,
            parameters=t.parameters,
            dependencies=t.dependencies,
            # resolved_params 留空，由执行器运行时填充
        )
        old_graph.add_task(old_task)

    # 复制 planner_thought（用于调试）
    old_graph.planner_thought = getattr(new_graph, "planner_thought", "")

    logger.info(
        f"[Adapter] TaskGraph 转换完成 | 新体系任务数={len(new_graph.tasks)} "
        f"| 旧体系任务数={len(old_graph.tasks)}"
    )
    return old_graph


# 类型引用（避免循环导入）
from app.core.task_graph import TaskGraph as NewTaskGraph  # noqa: E402
