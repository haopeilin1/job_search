"""
ReAct执行器 v2 —— 按依赖拓扑执行任务图，支持动态Replan

核心循环：
  while 还有未完成任务:
    1. 获取所有可执行任务（依赖已到达终态）
    2. 并行执行本批任务
    3. 观察结果
    4. 判断是否需要Replan（T1-T5触发条件）
    5. 若需要Replan → 局部扩展或全局重构任务图
    6. 若触发ask_user/abort → 提前退出

与v1的区别：
- Fallback由执行器动态决定，不依赖任务节点的预设fallback
- 支持运行时Replan（插入新任务）
- 工具异常时优先寻找替代方案
"""

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from app.core.llm_client import LLMClient, TIMEOUT_STANDARD
from app.core.memory import SessionMemory
from app.core.mcp_search import ExternalSearchTool
from app.core.planner import TaskGraph, TaskNode
from app.core.tool_registry import ToolRegistry, create_tool_registry

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# 1. Replan触发条件定义
# ═══════════════════════════════════════════════════════

class Replanner:
    """
    Replan判断器。
    根据任务执行结果，判断是否需要调整任务图。
    """

    @staticmethod
    def should_replan(task: TaskNode, graph: TaskGraph) -> Optional[str]:
        """
        判断是否需要Replan，返回触发原因或None。
        
        T1: 工具硬失败
        T2: 检索结果不足
        T3: 检索结果与query不相关
        T4: 业务规则触发补充任务（如匹配度过低需生成建议）
        T5: 发现新信息缺口（预留扩展点）
        """
        # T1: 工具硬失败
        if task.status == "failed":
            return "T1_tool_failed"

        # T2: 检索结果不足（仅针对检索类工具）
        if task.tool_name in ("kb_retrieve", "external_search") and task.status == "success":
            result_data = task.result if isinstance(task.result, dict) else {}
            chunks = result_data.get("chunks", [])
            if len(chunks) < 2:
                return "T2_insufficient_results"
            # 检查最高分
            scores = [c.get("hybrid_score", 0) for c in chunks if isinstance(c, dict)]
            if scores and max(scores) < 0.3:
                return "T2_low_relevance"

        # T3: 检索结果不相关（轻量判断）
        # 注：T3的实现成本较高，需要额外LLM调用，当前版本暂不自动触发
        # 可在未来版本中通过轻量模型判断结果相关性

        # T4: 业务规则触发
        if task.tool_name == "match_analyze" and task.status == "success":
            result_data = task.result if isinstance(task.result, dict) else {}
            score = result_data.get("match_score", 0)
            if score < 50:
                # 匹配度过低，可能需要补充建议或推荐其他岗位
                return "T4_low_match_score"

        # T5: 预留扩展
        return None

    @staticmethod
    async def replan(graph: TaskGraph, trigger: str, failed_task: TaskNode) -> TaskGraph:
        """
        根据触发原因调整任务图。
        
        L1: 节点级修复（retry / 替代方案）
        L2: 局部扩展（在当前节点后插入新任务）
        L3: 全局重构（重新生成任务图，当前版本暂不支持）
        """
        if trigger == "T1_tool_failed":
            return await Replanner._handle_tool_failure(graph, failed_task)

        if trigger == "T2_insufficient_results":
            return Replanner._handle_insufficient_results(graph, failed_task)

        if trigger == "T2_low_relevance":
            return Replanner._handle_low_relevance(graph, failed_task)

        if trigger == "T4_low_match_score":
            return Replanner._handle_low_match(graph, failed_task)

        return graph

    @staticmethod
    async def _handle_tool_failure(graph: TaskGraph, task: TaskNode) -> TaskGraph:
        """处理工具失败：尝试替代方案"""
        error_msg = str(task.observation).lower()

        # 依赖缺失（如jieba未安装）→ 尝试替代方案
        if "jieba" in error_msg or "no module named" in error_msg:
            if task.tool_name == "kb_retrieve":
                # 替代方案：禁用BM25，只用向量检索
                logger.info(f"[Replan] {task.task_id} 依赖缺失，尝试替代方案：纯向量检索")
                task.parameters["_disable_bm25"] = True
                task.status = "pending"
                task.observation = ""
                return graph

        # 网络超时 → retry（由执行器在主循环中处理）
        if "timeout" in error_msg or "超时" in error_msg:
            task.status = "pending"
            task.observation = ""
            return graph

        # 其他失败 → 无法自动修复，标记为失败
        return graph

    @staticmethod
    def _handle_insufficient_results(graph: TaskGraph, task: TaskNode) -> TaskGraph:
        """处理检索结果不足：插入外部搜索任务"""
        if task.tool_name == "kb_retrieve":
            # 在kb_retrieve后插入external_search
            new_task_id = f"T{len(graph.tasks)}"
            search_query = task.resolved_params.get("query", "")
            new_task = TaskNode(
                task_id=new_task_id,
                task_type="tool_call",
                tool_name="external_search",
                description="补充外部搜索（知识库召回不足）",
                parameters={"query": search_query, "count": 5},
                dependencies=[task.task_id],
                is_critical=False,
            )
            graph.add_task(new_task)

            # 更新下游任务的依赖（如果下游依赖kb_retrieve，现在可以也依赖external_search）
            # 简化处理：下游任务仍只依赖kb_retrieve，external_search的结果通过参数传递时按需合并
            logger.info(f"[Replan] 插入外部搜索任务 {new_task_id} | query={search_query}")

        return graph

    @staticmethod
    def _handle_low_relevance(graph: TaskGraph, task: TaskNode) -> TaskGraph:
        """处理检索结果相关度过低：同样插入外部搜索"""
        return Replanner._handle_insufficient_results(graph, task)

    @staticmethod
    def _handle_low_match(graph: TaskGraph, task: TaskNode) -> TaskGraph:
        """处理匹配度过低：插入建议生成任务"""
        new_task_id = f"T{len(graph.tasks)}"
        new_task = TaskNode(
            task_id=new_task_id,
            task_type="llm_reasoning",
            tool_name=None,
            description="基于低匹配度结果生成提升建议",
            parameters={
                "system_prompt": "你是一位职业顾问。用户与目标岗位匹配度较低，请给出针对性的提升建议。",
                "prompt": "{{T_match.output}}",  # 由执行器动态解析
            },
            dependencies=[task.task_id],
            is_critical=False,
        )
        graph.add_task(new_task)
        logger.info(f"[Replan] 插入建议生成任务 {new_task_id}")
        return graph


# ═══════════════════════════════════════════════════════
# 2. 执行器核心类
# ═══════════════════════════════════════════════════════

class ReActExecutor:
    """
    ReAct执行器。
    
    职责：
    - 拓扑执行（尊重依赖关系，同层并行）
    - 动态槽位填充（解析跨任务引用）
    - 运行时Replan（结果不足/失败时扩展任务图）
    - 失败处理（替代方案/报告用户/降级跳过）
    """

    MAX_STEPS = 10  # 防止无限循环（含Replan扩展）
    RETRY_MAX = 3
    RETRY_BACKOFF_BASE = 1.0  # 秒

    def __init__(self, tool_registry: Optional[ToolRegistry] = None):
        self.registry = tool_registry or create_tool_registry()
        self.external_search = ExternalSearchTool()
        self.llm = None

    async def execute(self, graph: TaskGraph, session: SessionMemory) -> TaskGraph:
        """
        执行TaskGraph直到完成。
        """
        logger.info(f"[ReActExecutor] 开始执行 | tasks={len(graph.tasks)}")
        graph.global_status = "running"

        iteration = 0
        while iteration < self.MAX_STEPS:
            iteration += 1

            # 1. 获取可执行任务
            ready = graph.get_ready_tasks()
            if not ready:
                pending = [t.task_id for t in graph.tasks.values() if t.status == "pending"]
                if pending:
                    logger.warning(f"[ReActExecutor] 死锁检测 | pending={pending}")
                break

            # 2. 执行本批任务（并行）
            logger.info(f"[ReActExecutor] 第{iteration}轮 | ready={len(ready)} | tasks={[t.task_id for t in ready]}")
            results = await asyncio.gather(
                *[self._execute_single(t, graph, session) for t in ready],
                return_exceptions=True,
            )

            # 3. 处理结果 + Replan判断
            needs_replan = False
            for task, result in zip(ready, results):
                if isinstance(result, Exception):
                    logger.error(f"[ReActExecutor] {task.task_id} 异常: {result}")
                    task.status = "failed"
                    task.observation = f"执行异常: {result}"

                # 判断是否需要Replan
                replan_trigger = Replanner.should_replan(task, graph)
                if replan_trigger:
                    logger.info(f"[ReActExecutor] {task.task_id} 触发Replan: {replan_trigger}")
                    graph = await Replanner.replan(graph, replan_trigger, task)
                    needs_replan = True

            # 4. 检查全局状态
            if graph.global_status in ("needs_clarification", "failed"):
                logger.info(f"[ReActExecutor] 全局状态={graph.global_status}，提前退出")
                break

            # 如果有Replan插入了新任务，继续循环
            if needs_replan:
                continue

            # 检查是否全部完成
            all_terminal = all(
                t.status in ("success", "failed", "skipped", "aborted")
                for t in graph.tasks.values()
            )
            if all_terminal:
                break

        # 5. 标记所有未完成状态
        for t in graph.tasks.values():
            if t.status in ("pending", "running"):
                t.status = "skipped"
                t.observation = "执行器退出时未执行"

        # 6. 汇总
        success = sum(1 for t in graph.tasks.values() if t.status == "success")
        failed = sum(1 for t in graph.tasks.values() if t.status == "failed")
        skipped = sum(1 for t in graph.tasks.values() if t.status == "skipped")

        if graph.global_status not in ("needs_clarification", "failed"):
            graph.global_status = "success" if success > 0 else "failed"

        logger.info(
            f"[ReActExecutor] 执行完成 | total={len(graph.tasks)} | "
            f"success={success} | failed={failed} | skipped={skipped} | "
            f"status={graph.global_status}"
        )
        return graph

    async def _execute_single(self, task: TaskNode, graph: TaskGraph, session: SessionMemory) -> None:
        """执行单个任务"""
        if task.status != "pending":
            return

        task.status = "running"
        task.started_at = time.time()

        try:
            # 1. 动态槽位填充（解析跨任务引用）
            # 优先使用 planner 已静态解析的参数，避免覆盖 entity.xxx 等静态占位符
            base_params = task.resolved_params if task.resolved_params else task.parameters
            resolved_params = self._resolve_dynamic_params(base_params, graph, session)

            # 对 match_analyze 任务：若 jd_text 来自 chunks[N] 但 company/position 不匹配，自动修正
            if task.tool_name == "match_analyze":
                company = resolved_params.get("company")
                position = resolved_params.get("position")
                jd_text = resolved_params.get("jd_text")
                if (company or position) and isinstance(jd_text, dict):
                    meta = jd_text.get("metadata", {})
                    c_company = meta.get("company", "")
                    c_position = meta.get("position", "")
                    is_match = (not company or company in c_company) and (not position or position in c_position)
                    if not is_match:
                        # 从 T0 结果中筛选匹配的 chunk
                        t0 = graph.get_task("T0")
                        if t0 and t0.status == "success" and isinstance(t0.result, dict):
                            chunks = t0.result.get("chunks", [])
                            filtered = []
                            for c in chunks:
                                if isinstance(c, dict):
                                    m = c.get("metadata", {})
                                    cc = m.get("company", "")
                                    cp = m.get("position", "")
                                    if (not company or company in cc) and (not position or position in cp):
                                        filtered.append(c)
                            if filtered:
                                resolved_params["jd_text"] = filtered[0]
                                logger.info(f"[ReActExecutor] match_analyze jd_text 自动修正: {company}/{position} | 从 {len(chunks)} 个 chunk 中命中 {len(filtered)} 个")
                            else:
                                logger.warning(f"[ReActExecutor] match_analyze 未找到 {company}/{position} 匹配的 chunk，保持原样")

            task.resolved_params = resolved_params

            # 2. 按任务类型执行
            if task.task_type == "tool_call":
                if task.tool_name == "external_search":
                    result = await self.external_search.execute(resolved_params)
                else:
                    result = await self._execute_tool_call(task, resolved_params, session)
            elif task.task_type == "llm_reasoning":
                result = await self._execute_llm_reasoning(task, resolved_params)
            elif task.task_type == "aggregate":
                result = await self._execute_aggregate(task, resolved_params)
            else:
                result = {"success": True, "data": {"note": f"未知任务类型: {task.task_type}"}}

            # 3. 更新状态
            if isinstance(result, dict) and result.get("success"):
                task.status = "success"
                task.result = result.get("data")
                task.observation = result.get("observation", "执行成功")
            else:
                task.status = "failed"
                task.result = result.get("data") if isinstance(result, dict) else None
                task.observation = result.get("error", "执行失败") if isinstance(result, dict) else str(result)

        except Exception as e:
            task.status = "failed"
            task.observation = f"执行异常: {e}"
            logger.error(f"[ReActExecutor] {task.task_id} 异常: {e}")

        task.finished_at = time.time()
        logger.info(f"[ReActExecutor] {task.task_id} {task.status} | tool={task.tool_name} | obs={task.observation[:60]}...")

    async def _execute_tool_call(
        self, task: TaskNode, params: Dict[str, Any], session: SessionMemory
    ) -> Dict[str, Any]:
        """执行外部工具调用"""
        from app.core.tool_registry import ToolCall as RegistryToolCall

        tool = self.registry.get(task.tool_name)
        if not tool:
            return {"success": False, "error": f"未知工具: {task.tool_name}"}

        # 参数校验
        missing = tool.validate_params(params)
        if missing:
            return {"success": False, "error": f"缺失必填参数: {missing}"}

        # 执行
        turn_id = len(session.working_memory.turns) + 1 if session and hasattr(session, "working_memory") else None
        call = RegistryToolCall(
            name=task.tool_name,
            params=params,
            session_id=getattr(session, "session_id", None),
            turn_id=turn_id,
        )
        result = await self.registry.execute(call)

        return {
            "success": result.success,
            "data": result.data,
            "error": result.error,
            "observation": f"工具 {task.tool_name} 执行{'成功' if result.success else '失败'}",
        }

    async def _execute_llm_reasoning(self, task: TaskNode, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行纯LLM推理"""
        if self.llm is None:
            self.llm = LLMClient.from_config("chat")

        system = params.get("system_prompt", "你是一位专业的求职顾问。")
        prompt = params.get("prompt", params.get("user_prompt", json.dumps(params, ensure_ascii=False)))

        try:
            raw = await self.llm.generate(
                prompt=prompt,
                system=system,
                temperature=params.get("temperature", 0.3),
                max_tokens=params.get("max_tokens", 1000),
                timeout=TIMEOUT_STANDARD,  # 20s，LLM 推理任务
            )
            return {"success": True, "data": {"output": raw}, "observation": "LLM推理成功"}
        except Exception as e:
            return {"success": False, "error": f"LLM推理失败: {e}"}

    async def _execute_aggregate(self, task: TaskNode, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行聚合任务"""
        return {"success": True, "data": {"aggregation": params}, "observation": "聚合完成"}

    # ═══════════════════════════════════════════════════════
    # 动态槽位填充
    # ═══════════════════════════════════════════════════════

    def _resolve_dynamic_params(self, params: Any, graph: TaskGraph, session: SessionMemory) -> Any:
        """递归解析参数中的动态占位符（跨任务引用）"""
        if isinstance(params, str):
            return self._resolve_string(params, graph, session)
        if isinstance(params, dict):
            return {k: self._resolve_dynamic_params(v, graph, session) for k, v in params.items()}
        if isinstance(params, list):
            return [self._resolve_dynamic_params(item, graph, session) for item in params]
        return params

    def _resolve_string(self, text: str, graph: TaskGraph, session: SessionMemory) -> Any:
        """解析字符串中的占位符"""
        if not text or "{{" not in text:
            return text

        # 纯占位符 → 返回原始Python对象
        match = re.fullmatch(r"\{\{([\w.\[\]]+)\}\}", text.strip())
        if match:
            val = self._get_value(match.group(1), graph, session)
            return val if val is not None else text

        # 嵌入占位符 → 字符串替换
        def replacer(m: re.Match) -> str:
            val = self._get_value(m.group(1), graph, session)
            if val is None:
                return ""
            if isinstance(val, (dict, list)):
                return json.dumps(val, ensure_ascii=False)
            return str(val)

        return re.sub(r"\{\{([\w.\[\]]+)\}\}", replacer, text)

    def _get_value(self, path: str, graph: TaskGraph, session: SessionMemory) -> Any:
        """根据路径获取值"""
        parts = path.split(".")
        if not parts:
            return None

        # Txx.output.xxx → 当前graph任务结果
        if re.match(r"^T\d+", parts[0]):
            task = graph.get_task(parts[0])
            if not task or task.status != "success":
                return None
            val = task.result
            start_idx = 1
            if len(parts) > 1 and parts[1] == "output":
                start_idx = 2
            for p in parts[start_idx:]:
                if isinstance(val, dict):
                    val = val.get(p)
                elif isinstance(val, list):
                    # 支持 chunks[0] 语法
                    try:
                        idx = int(p)
                        val = val[idx] if 0 <= idx < len(val) else None
                    except ValueError:
                        return None
                else:
                    return None
            return val

        # entity.xxx → 从graph的静态上下文中获取
        if parts[0] == "entity" and len(parts) >= 2:
            # 尝试从任务的resolved_params中获取，或从session获取
            if session and hasattr(session, "global_slots"):
                return session.global_slots.get(parts[1])
            return None

        # evidence_cache → 从session获取上轮检索结果
        if parts[0] == "evidence_cache":
            if session and hasattr(session, "evidence_cache"):
                return session.evidence_cache
            return []

        # session.xxx
        if parts[0] == "session" and len(parts) >= 2:
            return getattr(session, parts[1], None)

        return None
