"""
Agent 反思模块 (Reflection Module) —— 简化版

设计原则：
  - 100% 确定的问题 → 硬规则（0ms）
  - 其余所有情况 → 轻量 LLM 一次判断（50-200ms）
  - 不层层递进，不搞工作流

职责：
  1. 硬规则兜底（结果为空、全部失败）
  2. 轻量 LLM 反思：完整性、冲突、时效性、缓存复用质量
  3. 输出建议动作：pass / re_retrieve / note_uncertainty / replan_tools

调用时机：
  PlanExecutor.execute() 完成后 → 聚合/直出前
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.llm_client import LLMClient, TIMEOUT_LIGHT
from app.core.memory import SessionMemory
from app.core.planner import TaskGraph, TaskNode

logger = logging.getLogger(__name__)


@dataclass
class ReflectionResult:
    """反思结果"""
    is_complete: bool = True
    has_conflict: bool = False
    missing_info: List[str] = field(default_factory=list)
    suggested_action: str = "pass"     # pass | re_retrieve | note_uncertainty | replan_tools
    confidence: float = 1.0
    reason: str = ""
    problematic_task: str = ""         # 问题出在哪个 task_id（如 "T0"）
    details: Dict[str, Any] = field(default_factory=dict)


class AgentReflector:
    """
    Agent 执行后反思器 —— 简化版。

    只分两条路：
      1. 硬规则（100% 确定的问题）→ 直接返回
      2. 轻量 LLM（其余所有情况）→ 一次判断
    """

    # ── 硬规则触发条件（100% 确定） ──
    @classmethod
    def _hard_rule_check(
        cls, graph: TaskGraph, user_query: str
    ) -> Optional[ReflectionResult]:
        """
        硬规则快速兜底。仅在结论100%确定时返回非None。
        """
        _task_iter = graph.tasks.values() if hasattr(graph.tasks, 'values') else graph.tasks
        success_tasks = [t for t in _task_iter if t.status == "success"]
        failed_tasks = [t for t in _task_iter if t.status == "failed"]

        # 规则1：没有任何任务成功
        if not success_tasks:
            failed_ids = [t.task_id for t in failed_tasks]
            return ReflectionResult(
                is_complete=False,
                missing_info=["所有任务均未成功执行"],
                suggested_action="replan_tools",
                confidence=1.0,
                reason="HARD_RULE: 无成功任务",
                problematic_task=failed_ids[0] if failed_ids else "",
            )

        # 规则2：检索类任务成功但结果为空
        for task in success_tasks:
            if task.tool_name in ("kb_retrieve", "external_search"):
                res = task.result if isinstance(task.result, dict) else {}
                chunks = res.get("chunks", []) if isinstance(res, dict) else []
                if not chunks:
                    return ReflectionResult(
                        is_complete=False,
                        missing_info=[f"{task.tool_name} 检索结果为空"],
                        suggested_action="re_retrieve",
                        confidence=1.0,
                        reason=f"HARD_RULE: {task.tool_name} 空结果",
                        problematic_task=task.task_id,
                    )

        # 规则3：关键任务失败（非外部搜索）
        critical_failed = [t for t in failed_tasks if t.tool_name != "external_search"]
        if critical_failed:
            return ReflectionResult(
                is_complete=False,
                missing_info=[f"关键任务失败: {[t.tool_name for t in critical_failed]}"],
                suggested_action="replan_tools",
                confidence=1.0,
                reason=f"HARD_RULE: 关键任务失败 {critical_failed[0].tool_name}",
                problematic_task=critical_failed[0].task_id,
            )

        return None

    # ── 轻量 LLM 反思 ──
    @classmethod
    async def _llm_reflect(
        cls,
        graph: TaskGraph,
        session: SessionMemory,
        user_query: str,
    ) -> ReflectionResult:
        """
        调用轻量 LLM 一次性反思。

        Prompt 中明确告知 LLM：
          1. 用户原始问题
          2. 各工具执行结果摘要
          3. 反思维度（完整性、冲突、时效性、缓存复用）
          4. 输出格式
        """
        # 构建任务摘要
        task_summaries = []
        _tasks = graph.tasks.values() if hasattr(graph.tasks, 'values') else graph.tasks
        for task in _tasks:
            status_emoji = "✅" if task.status == "success" else "❌" if task.status == "failed" else "⏭"
            res_preview = ""
            if task.result and isinstance(task.result, dict):
                # 提取关键信息
                chunks = task.result.get("chunks", [])
                if chunks:
                    res_preview = f"chunks={len(chunks)}"
                else:
                    res_preview = str(task.result.get("data", task.result))[:150]
            elif task.observation:
                res_preview = str(task.observation)[:150]
            task_summaries.append(
                f"{status_emoji} {task.task_id} | {task.tool_name or task.task_type} | {res_preview}"
            )

        # 证据缓存复用信息
        if session and session.evidence_cache:
            cache_info = f"evidence_cache={len(session.evidence_cache)}条 (query='{session.evidence_cache_query[:40]}...')"
        else:
            cache_info = "evidence_cache=空（本轮未复用缓存）"

        system_prompt = (
            "你是Agent质量检查员。请根据用户问题和工具执行结果，判断是否存在以下问题。"
            "你只检查工具执行结果的质量（检索有无内容、任务有无失败、来源有无冲突、缓存复用是否相关），"
            "不检查聚合LLM、qa_synthesize或general_chat生成的文本质量。\n"
            "\n"
            "【反思维度】\n"
            "1. 完整性：工具结果是否完整回答了用户问题？\n"
            "   - 检查工具输出的内容是否覆盖了用户问题的所有方面\n"
            "   - 例如用户问'薪资'但检索结果只有'岗位要求' → 不完整\n"
            "2. 来源冲突：kb_retrieve（本地知识库）和 external_search（外部搜索）的结果是否矛盾？\n"
            "3. 时效性：用户问题含'最近/最新/目前'等词时，结果是否标注了时间范围？\n"
            "4. 缓存复用质量（仅在 evidence_cache 非空时检查）：如果使用了 evidence_cache 复用，复用的内容是否与当前问题真正相关？\n"
            "   - 如果 evidence_cache 为空（本轮未复用缓存），跳过此项检查\n"
            "5. 工具输出质量：match_analyze是否返回了分数/标签/差距？interview_gen是否返回了题目？qa_synthesize是否返回了答案？\n"
            "\n"
            "【输出格式】严格JSON，不要markdown代码块：\n"
            "{\n"
            '  "is_complete": true/false,       // 结果是否完整回答了用户问题\n'
            '  "has_conflict": true/false,       // 是否存在来源冲突\n'
            '  "missing_info": ["..."],          // 缺失信息列表（如无则空数组）\n'
            '  "suggested_action": "pass" | "re_retrieve" | "note_uncertainty" | "replan_tools",\n'
            '  "problematic_task": "T0" | "",    // 【问题定位】哪个任务出了问题（task_id），如无则空字符串\n'
            '  "confidence": 0.0-1.0,            // 你的判断置信度\n'
            '  "reason": "简要理由"\n'
            "}\n"
            "\n"
            "【问题定位说明】\n"
            "- 如果某个任务失败导致整体无法回答，problematic_task 填写该任务的 task_id\n"
            "- 如果是检索结果缺失关键信息（如缺少薪资），填写对应的检索任务 task_id\n"
            "- 如果是缓存复用导致答非所问，填写复用的 kb_retrieve 任务 task_id\n"
            "- 如果无法确定或没有问题，填写空字符串\"\"\n"
            "\n"
            "【action 说明】\n"
            "- pass：结果完整，无问题\n"
            "- re_retrieve：信息缺失或检索结果不足，建议补充检索（会触发外部搜索或重新检索）\n"
            "- note_uncertainty：结果基本可用但存在冲突或不确定性，需要在回复中标注\n"
            "- replan_tools：严重问题（如关键任务失败），需要重新规划任务\n"
        )

        user_prompt = (
            f"【用户问题】{user_query}\n"
            f"【证据缓存】{cache_info}\n"
            f"【工具执行结果】\n" + "\n".join(task_summaries[:10]) + "\n"
            "\n请进行反思检查，输出JSON："
        )

        try:
            llm = LLMClient.from_config("memory")
            raw = await llm.generate(
                prompt=user_prompt,
                system=system_prompt,
                timeout=TIMEOUT_LIGHT,
                max_tokens=300,
                temperature=0.1,
            )

            import json
            text = raw.strip()
            if text.startswith("```"):
                text = text.strip("`").replace("json", "").strip()

            data = json.loads(text)

            result = ReflectionResult(
                is_complete=bool(data.get("is_complete", True)),
                has_conflict=bool(data.get("has_conflict", False)),
                missing_info=data.get("missing_info", []) if isinstance(data.get("missing_info"), list) else [],
                suggested_action=data.get("suggested_action", "pass"),
                confidence=float(data.get("confidence", 0.8)),
                reason=data.get("reason", ""),
                problematic_task=data.get("problematic_task", ""),
            )

            # action 校验
            if result.suggested_action not in ("pass", "re_retrieve", "note_uncertainty", "replan_tools"):
                result.suggested_action = "note_uncertainty"

            return result

        except Exception as e:
            logger.warning(f"[Reflection] LLM 反思失败: {e}")
            # fallback：保守返回 note_uncertainty
            return ReflectionResult(
                suggested_action="note_uncertainty",
                reason=f"LLM反思失败: {e}",
                confidence=0.5,
            )

    # ── 主入口 ──
    @classmethod
    async def reflect(
        cls,
        graph: TaskGraph,
        session: SessionMemory,
        user_query: str,
    ) -> ReflectionResult:
        """主入口：先硬规则，再LLM反思。"""
        if not getattr(settings, "REFLECTION_ENABLED", True):
            return ReflectionResult(reason="反思模块已禁用")

        # 1. 硬规则兜底（100%确定的问题）
        hard_result = cls._hard_rule_check(graph, user_query)
        if hard_result:
            logger.info(f"[Reflection] 硬规则触发 | action={hard_result.suggested_action} | reason={hard_result.reason}")
            return hard_result

        # 2. 轻量 LLM 反思（其余所有情况）
        result = await cls._llm_reflect(graph, session, user_query)
        logger.info(
            f"[Reflection] LLM反思完成 | action={result.suggested_action} | "
            f"complete={result.is_complete} | conflict={result.has_conflict} | "
            f"confidence={result.confidence:.2f} | reason={result.reason[:80]}"
        )
        return result
