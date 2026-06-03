"""
Agent 反思模块 (Reflection Module) —— v2

设计原则：
  - 100% 确定的问题 → 硬规则（0ms）
  - 其余所有情况 → 轻量 LLM 一次判断（50-200ms）
  - 不层层递进，不搞工作流

新增反思维度（v2）：
  6. 语义相关性：检索结果的 company/position/属性 是否与用户 query 真正匹配
  7. 多来源证据冲突分析：kb_retrieve 与 external_search 矛盾时的深度分析

职责：
  1. 硬规则兜底（结果为空、全部失败）
  2. 轻量 LLM 反思：完整性、冲突、时效性、缓存复用质量、语义相关性、来源冲突分析
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


# ═══════════════════════════════════════════════════════
# 1. 数据模型
# ═══════════════════════════════════════════════════════

@dataclass
class SemanticRelevanceResult:
    """语义相关性检查结果"""
    is_relevant: bool = True
    issue: str = ""                    # 问题描述，如"岗位不匹配"、"公司不匹配"
    suggested_new_query: str = ""      # 建议的新检索 query（用于 replan）


@dataclass
class SourceConflictAnalysis:
    """多来源证据冲突分析"""
    has_conflict: bool = False
    analysis: str = ""                 # LLM 深度分析文本
    recommendation: str = ""           # 给最终回复的建议（如"标注两者差异，不替用户裁决"）
    severity: str = "low"              # low / medium / high


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
    # v2 新增
    semantic_relevance: Optional[SemanticRelevanceResult] = None
    source_conflict_analysis: Optional[SourceConflictAnalysis] = None


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
          2. 各工具执行结果摘要（含 chunks 的 metadata）
          3. 反思维度（完整性、冲突、时效性、缓存复用、语义相关性、来源冲突分析）
          4. 输出格式
        """
        # 构建任务摘要
        task_summaries = []
        retrieval_results = []   # 用于传给 LLM 做语义相关性判断
        _tasks = graph.tasks.values() if hasattr(graph.tasks, 'values') else graph.tasks
        for task in _tasks:
            status_emoji = "✅" if task.status == "success" else "❌" if task.status == "failed" else "⏭"
            res_preview = ""
            chunk_metadata = []
            if task.result and isinstance(task.result, dict):
                chunks = task.result.get("chunks", [])
                if chunks:
                    res_preview = f"chunks={len(chunks)}"
                    # 提取前3条 chunk 的 metadata 用于语义相关性判断
                    for i, c in enumerate(chunks[:3]):
                        if isinstance(c, dict):
                            meta = c.get("metadata", {})
                            m_str = f"[{i+1}] company={meta.get('company','?')}, position={meta.get('position','?')}, score={c.get('hybrid_score','?')}"
                            chunk_metadata.append(m_str)
                else:
                    res_preview = str(task.result.get("data", task.result))[:150]
            elif task.observation:
                res_preview = str(task.observation)[:150]
            
            summary = f"{status_emoji} {task.task_id} | {task.tool_name or task.task_type} | {res_preview}"
            if chunk_metadata:
                summary += "\n    " + "\n    ".join(chunk_metadata)
            task_summaries.append(summary)

        # 证据缓存复用信息
        if session and session.evidence_cache:
            cache_info = f"evidence_cache={len(session.evidence_cache)}条 (query='{session.evidence_cache_query[:40]}...')"
        else:
            cache_info = "evidence_cache=空（本轮未复用缓存）"

        system_prompt = (
            "你是Agent质量检查员。请根据用户问题和工具执行结果，判断是否存在以下问题。"
            "你只检查工具执行结果的质量（检索有无内容、任务有无失败、来源有无冲突、缓存复用是否相关、检索语义是否匹配），"
            "不检查聚合LLM、qa_synthesize或general_chat生成的文本质量。\n"
            "\n"
            "【反思维度】\n"
            "1. 完整性：工具结果是否完整回答了用户问题？\n"
            "   - 检查工具输出的内容是否覆盖了用户问题的所有方面\n"
            "   - 例如用户问'薪资'但检索结果只有'岗位要求' → 不完整\n"
            "2. 来源冲突（仅当 kb_retrieve 和 external_search 同时成功返回内容时检查）：\n"
            "   - 比较两者在相同属性（如薪资、要求、福利）上的数值/描述是否矛盾\n"
            "   - 分析各来源的可信度：知识库是内部结构化JD，外部搜索是网络公开信息，两者权威性不同\n"
            "   - ⚠️ 你只分析冲突并给出建议，不做最终裁决，不要把一方的说法当作唯一正确答案\n"
            "   - 如果冲突严重（severity=high）且可能影响用户核心决策，建议 replan_tools 追加验证检索\n"
            "3. 时效性：用户问题含'最近/最新/目前'等词时，结果是否标注了时间范围？\n"
            "4. 缓存复用质量（仅在 evidence_cache 非空时检查）：如果使用了 evidence_cache 复用，复用的内容是否与当前问题真正相关？\n"
            "   - 如果 evidence_cache 为空（本轮未复用缓存），跳过此项检查\n"
            "5. 工具输出质量：match_analyze是否返回了分数/标签/差距？interview_gen是否返回了题目？qa_synthesize是否返回了答案？\n"
            "6. 语义相关性（关键维度）：\n"
            "   - 检查检索结果的 metadata 中的 company/position 是否与用户 query 中的目标真正匹配\n"
            "   - 同义词、简称、岗位名变体（如'算法工程师'vs'算法岗'）不算不匹配\n"
            "   - 但'后端开发'vs'产品经理'、'字节跳动'vs'百度'算严重不匹配\n"
            "   - 如果检索结果与用户意图无关（is_relevant=false），必须给出 suggested_new_query，让系统重新检索\n"
            "   - 语义不相关时，suggested_action 应为 replan_tools（不是 re_retrieve，因为原 query 方向可能错了）\n"
            "\n"
            "【输出格式】严格JSON，不要markdown代码块：\n"
            "{\n"
            '  "is_complete": true/false,       // 结果是否完整回答了用户问题\n'
            '  "has_conflict": true/false,       // 是否存在来源冲突\n'
            '  "missing_info": ["..."],          // 缺失信息列表（如无则空数组）\n'
            '  "suggested_action": "pass" | "re_retrieve" | "note_uncertainty" | "replan_tools",\n'
            '  "problematic_task": "T0" | "",    // 【问题定位】哪个任务出了问题（task_id），如无则空字符串\n'
            '  "confidence": 0.0-1.0,            // 你的判断置信度\n'
            '  "reason": "简要理由",\n'
            '  "semantic_relevance": {           // 【语义相关性分析】\n'
            '    "is_relevant": true/false,\n'
            '    "issue": "问题描述，如岗位不匹配",\n'
            '    "suggested_new_query": "建议的新检索query"\n'
            '  },\n'
            '  "source_conflict_analysis": {     // 【多来源冲突分析，仅当两者都有结果时】\n'
            '    "has_conflict": true/false,\n'
            '    "analysis": "深度分析文本",\n'
            '    "recommendation": "给最终回复的建议，如同时呈现并标注差异",\n'
            '    "severity": "low" | "medium" | "high"\n'
            '  }\n'
            "}\n"
            "\n"
            "【问题定位说明】\n"
            "- 如果某个任务失败导致整体无法回答，problematic_task 填写该任务的 task_id\n"
            "- 如果是检索结果缺失关键信息（如缺少薪资），填写对应的检索任务 task_id\n"
            "- 如果是检索结果语义不相关（如搜到了错误岗位），填写对应的 kb_retrieve 任务 task_id\n"
            "- 如果是缓存复用导致答非所问，填写复用的 kb_retrieve 任务 task_id\n"
            "- 如果无法确定或没有问题，填写空字符串\"\"\n"
            "\n"
            "【action 说明】\n"
            "- pass：结果完整，无问题\n"
            "- re_retrieve：信息缺失或检索结果不足，建议补充检索（会触发外部搜索或重新检索）\n"
            "- note_uncertainty：结果基本可用但存在冲突或不确定性，需要在回复中标注。注意：来源冲突时优先用此选项，把冲突分析写入回复即可\n"
            "- replan_tools：严重问题（如关键任务失败、检索结果语义完全不相关、冲突严重需追加验证），需要重新规划任务\n"
            "\n"
            "【重要约束】\n"
            "- 来源冲突分析中，你只提供分析和建议，不替用户做最终决策\n"
            "- 语义不相关时，suggested_new_query 必须具体且准确，不要泛泛而谈\n"
            "- 如果仅有轻微冲突（severity=low），用 note_uncertainty 即可，不需要 replan_tools\n"
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
                max_tokens=800,
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

            # 解析 semantic_relevance
            sr_raw = data.get("semantic_relevance")
            if isinstance(sr_raw, dict):
                result.semantic_relevance = SemanticRelevanceResult(
                    is_relevant=bool(sr_raw.get("is_relevant", True)),
                    issue=str(sr_raw.get("issue", "")),
                    suggested_new_query=str(sr_raw.get("suggested_new_query", "")),
                )
                # 自动修正 action：如果语义不相关且当前不是 replan_tools，升级为 replan_tools
                if not result.semantic_relevance.is_relevant and result.suggested_action != "replan_tools":
                    result.suggested_action = "replan_tools"
                    if not result.problematic_task:
                        # 尝试定位到 kb_retrieve 任务
                        for t in _tasks:
                            if t.tool_name == "kb_retrieve" and t.status == "success":
                                result.problematic_task = t.task_id
                                break
                    result.reason += f" | 语义不相关: {result.semantic_relevance.issue}"
                    logger.info(f"[Reflection] 语义不相关自动升级 action=replan_tools | issue={result.semantic_relevance.issue}")

            # 解析 source_conflict_analysis
            sc_raw = data.get("source_conflict_analysis")
            if isinstance(sc_raw, dict):
                result.source_conflict_analysis = SourceConflictAnalysis(
                    has_conflict=bool(sc_raw.get("has_conflict", False)),
                    analysis=str(sc_raw.get("analysis", "")),
                    recommendation=str(sc_raw.get("recommendation", "")),
                    severity=str(sc_raw.get("severity", "low")).lower(),
                )
                # 冲突严重时，若当前是 note_uncertainty，保持；若当前是 pass，升级为 note_uncertainty
                if result.source_conflict_analysis.has_conflict:
                    if result.source_conflict_analysis.severity == "high" and result.suggested_action in ("pass", "re_retrieve"):
                        result.suggested_action = "replan_tools"
                        result.reason += f" | 来源冲突严重({result.source_conflict_analysis.severity}): {result.source_conflict_analysis.analysis[:60]}"
                        logger.info(f"[Reflection] 来源冲突严重自动升级 action=replan_tools")
                    elif result.suggested_action == "pass":
                        result.has_conflict = True
                        result.suggested_action = "note_uncertainty"
                        result.reason += f" | 来源冲突({result.source_conflict_analysis.severity}): {result.source_conflict_analysis.analysis[:60]}"

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
        if result.semantic_relevance:
            logger.info(
                f"[Reflection] 语义相关性 | is_relevant={result.semantic_relevance.is_relevant} | "
                f"issue={result.semantic_relevance.issue} | "
                f"new_query={result.semantic_relevance.suggested_new_query[:40]}..."
            )
        if result.source_conflict_analysis:
            logger.info(
                f"[Reflection] 来源冲突分析 | has_conflict={result.source_conflict_analysis.has_conflict} | "
                f"severity={result.source_conflict_analysis.severity} | "
                f"analysis={result.source_conflict_analysis.analysis[:80]}..."
            )
        return result
