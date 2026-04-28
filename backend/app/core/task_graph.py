"""
TaskGraph 数据模型 —— 执行编排的核心数据结构

职责：
1. 定义 TaskNode、TaskGraph、ExecutionStrategy、HistoryCacheEntry 等数据类
2. 提供拓扑排序、依赖检查、状态查询、并行组计算、关键路径计算等工具方法
3. 支持运行时状态跟踪（status / result / observation / retry_count）
"""

import hashlib
import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from app.core.tool_registry import TOOL_REGISTRY_META

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# 1. Fallback 定义
# ═══════════════════════════════════════════════════════

@dataclass
class TaskFallback:
    """任务失败时的处理策略"""
    action: str                       # "retry" | "skip" | "ask_user" | "abort"
    reason: str
    downstream_impact: str = ""
    default_params: Dict[str, Any] = field(default_factory=dict)
    max_retries: int = 3              # 仅 action=retry 时有效
    backoff_base_ms: int = 1000       # 指数退避基数


# ═══════════════════════════════════════════════════════
# 2. 任务节点
# ═══════════════════════════════════════════════════════

@dataclass
class TaskNode:
    """计划中的单个任务节点"""
    task_id: str
    task_type: str                    # "tool_call" | "llm_reasoning" | "aggregate" | "ask_user"
    description: str = ""
    tool_name: Optional[str] = None   # tool_call 时必填
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    execution_mode: str = "auto"      # "auto" | "manual"
    fallback: Optional[TaskFallback] = None
    output_schema: Dict[str, Any] = field(default_factory=dict)

    # ── 运行时状态 ──
    status: str = "pending"           # pending | running | success | failed | skipped | aborted
    result: Any = None
    observation: str = ""
    retry_count: int = 0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    # ── 缓存复用标记 ──
    params_hash: str = ""             # 参数指纹，用于历史缓存比对
    reused_from_history: bool = False # 是否复用历史缓存
    history_task_id: str = ""         # 复用自哪个历史任务

    def is_terminal(self) -> bool:
        """是否为终态"""
        return self.status in ("success", "failed", "skipped", "aborted")

    def is_executable(self, graph: "TaskGraph") -> bool:
        """判断任务是否可以执行（所有依赖已到达终态）"""
        if self.status != "pending":
            return False
        for dep_id in self.dependencies:
            dep = graph.get_task(dep_id)
            if dep is None or not dep.is_terminal():
                return False
        return True

    def compute_params_hash(self) -> str:
        """计算参数的确定性指纹（用于缓存复用判断）"""
        try:
            canonical = json.dumps(self.parameters, sort_keys=True, ensure_ascii=False)
            return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        except Exception:
            return ""


# ═══════════════════════════════════════════════════════
# 3. 执行策略
# ═══════════════════════════════════════════════════════

@dataclass
class ExecutionStrategy:
    """执行策略与元信息"""
    parallel_groups: List[List[str]] = field(default_factory=list)
    critical_path: List[str] = field(default_factory=list)
    estimated_cost: str = "medium"    # low | medium | high


# ═══════════════════════════════════════════════════════
# 4. 历史缓存条目
# ═══════════════════════════════════════════════════════

@dataclass
class HistoryCacheEntry:
    """历史执行缓存条目，用于 Replan 时的复用判断"""
    task_id: str
    tool_name: Optional[str]
    params_hash: str
    status: str                       # success | failed | skipped
    output: Any = None
    timestamp: float = field(default_factory=time.time)

    def is_equivalent(self, other_params_hash: str) -> bool:
        """判断历史缓存是否与当前参数等价（仅成功缓存可复用）"""
        return self.params_hash == other_params_hash and self.status == "success"


# ═══════════════════════════════════════════════════════
# 5. TaskGraph 主类
# ═══════════════════════════════════════════════════════

@dataclass
class TaskGraph:
    """完整的任务图（执行计划）"""
    planner_thought: str = ""
    tasks: List[TaskNode] = field(default_factory=list)
    execution_strategy: ExecutionStrategy = field(default_factory=ExecutionStrategy)
    context: Dict[str, Any] = field(default_factory=dict)

    # ── 历史缓存（Replan 场景）──
    history_cache: List[HistoryCacheEntry] = field(default_factory=list)

    # ── 全局状态 ──
    global_status: str = "pending"
    clarification_question: str = ""
    clarification_options: List[str] = field(default_factory=list)

    # ── 占位符解析缓存 ──
    _resolved_cache: Dict[str, Any] = field(default_factory=dict, repr=False)

    def get_task(self, task_id: str) -> Optional[TaskNode]:
        for t in self.tasks:
            if t.task_id == task_id:
                return t
        return None

    def get_tasks_by_status(self, status: str) -> List[TaskNode]:
        return [t for t in self.tasks if t.status == status]

    def get_ready_tasks(self) -> List[TaskNode]:
        """获取所有依赖已满足且状态为 pending 的任务"""
        return [t for t in self.tasks if t.is_executable(self)]

    def has_pending_or_running(self) -> bool:
        return any(t.status in ("pending", "running") for t in self.tasks)

    def get_failed_or_aborted(self) -> List[TaskNode]:
        return [t for t in self.tasks if t.status in ("failed", "aborted")]

    def get_skipped(self) -> List[TaskNode]:
        return [t for t in self.tasks if t.status == "skipped"]

    def get_downstream_tasks(self, task_id: str) -> List[TaskNode]:
        """获取指定任务的所有下游任务（直接+间接）"""
        direct = [t for t in self.tasks if task_id in t.dependencies]
        all_downstream = set(direct)
        for d in direct:
            all_downstream.update(self.get_downstream_tasks(d.task_id))
        return list(all_downstream)

    def get_upstream_tasks(self, task_id: str) -> List[TaskNode]:
        """获取指定任务的所有上游任务（直接+间接）"""
        task = self.get_task(task_id)
        if not task:
            return []
        direct = [self.get_task(dep) for dep in task.dependencies if self.get_task(dep)]
        all_upstream = set(direct)
        for u in direct:
            if u:
                all_upstream.update(self.get_upstream_tasks(u.task_id))
        return list(all_upstream)

    def check_circular_dependencies(self) -> Optional[List[str]]:
        """检查循环依赖，返回循环路径或 None"""
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs(tid: str, path: List[str]) -> Optional[List[str]]:
            visited.add(tid)
            rec_stack.add(tid)
            task = self.get_task(tid)
            if task:
                for dep in task.dependencies:
                    if dep not in visited:
                        cycle = dfs(dep, path + [dep])
                        if cycle:
                            return cycle
                    elif dep in rec_stack:
                        try:
                            cycle_start = path.index(dep)
                        except ValueError:
                            cycle_start = len(path)
                        return path[cycle_start:] + [tid]
            rec_stack.remove(tid)
            return None

        for t in self.tasks:
            if t.task_id not in visited:
                cycle = dfs(t.task_id, [t.task_id])
                if cycle:
                    return cycle
        return None

    def compute_parallel_groups(self) -> List[List[str]]:
        """基于依赖关系自动计算并行组（拓扑分层）"""
        remaining = {t.task_id for t in self.tasks}
        groups: List[List[str]] = []

        while remaining:
            group = []
            for tid in list(remaining):
                task = self.get_task(tid)
                if task:
                    deps_done = all(dep not in remaining for dep in task.dependencies)
                    if deps_done:
                        group.append(tid)
            if not group:
                logger.warning("[TaskGraph] 依赖死锁，强制推进")
                group = [list(remaining)[0]]
            groups.append(group)
            remaining -= set(group)

        return groups

    def compute_critical_path(self) -> List[str]:
        """基于工具成本估算关键路径（最长耗时链）"""
        cost_map = {"low": 1, "medium": 2, "high": 3}

        def path_cost(task_ids: List[str]) -> int:
            total = 0
            for tid in task_ids:
                t = self.get_task(tid)
                if t and t.tool_name:
                    meta = TOOL_REGISTRY_META.get(t.tool_name, {})
                    total += cost_map.get(meta.get("cost_level", "medium"), 2)
                else:
                    total += 1
            return total

        end_ids = [t.task_id for t in self.tasks if t.task_type == "aggregate"]
        if not end_ids:
            if not self.tasks:
                return []
            end_ids = [self.tasks[-1].task_id]

        longest: List[str] = []
        longest_cost = -1

        def find_paths(end_id: str, current: List[str]):
            nonlocal longest, longest_cost
            task = self.get_task(end_id)
            if not task or not task.dependencies:
                rev = list(reversed(current))
                c = path_cost(rev)
                if c > longest_cost:
                    longest_cost = c
                    longest = rev
                return
            for dep in task.dependencies:
                find_paths(dep, current + [dep])

        for eid in end_ids:
            find_paths(eid, [eid])
        return longest

    def validate(self) -> List[str]:
        """验证 TaskGraph 合法性，返回错误列表"""
        errors: List[str] = []

        cycle = self.check_circular_dependencies()
        if cycle:
            errors.append(f"循环依赖: {' -> '.join(cycle)}")

        ids = [t.task_id for t in self.tasks]
        dupes = [k for k, v in Counter(ids).items() if v > 1]
        if dupes:
            errors.append(f"重复 task_id: {dupes}")

        for t in self.tasks:
            for dep in t.dependencies:
                if not self.get_task(dep):
                    errors.append(f"任务 {t.task_id} 依赖不存在: {dep}")

        for t in self.tasks:
            if t.task_type == "tool_call" and not t.tool_name:
                errors.append(f"任务 {t.task_id} 为 tool_call 但缺 tool_name")

        if self.tasks and self.tasks[-1].task_type != "aggregate":
            errors.append("[WARN] 末任务建议为 aggregate")

        declared_flat = set()
        for g in self.execution_strategy.parallel_groups:
            declared_flat.update(g)
        task_ids = set(t.task_id for t in self.tasks)
        if declared_flat and declared_flat != task_ids:
            missing = task_ids - declared_flat
            extra = declared_flat - task_ids
            if missing:
                errors.append(f"parallel_groups 缺: {missing}")
            if extra:
                errors.append(f"parallel_groups 多余: {extra}")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "planner_thought": self.planner_thought,
            "tasks": [
                {
                    "task_id": t.task_id,
                    "task_type": t.task_type,
                    "tool_name": t.tool_name,
                    "description": t.description,
                    "parameters": t.parameters,
                    "dependencies": t.dependencies,
                    "execution_mode": t.execution_mode,
                    "fallback": {
                        "action": t.fallback.action,
                        "reason": t.fallback.reason,
                        "downstream_impact": t.fallback.downstream_impact,
                        "default_params": t.fallback.default_params,
                        "max_retries": t.fallback.max_retries,
                    } if t.fallback else None,
                    "output_schema": t.output_schema,
                    "status": t.status,
                    "result": t.result,
                    "observation": t.observation,
                    "retry_count": t.retry_count,
                    "params_hash": t.params_hash,
                    "reused_from_history": t.reused_from_history,
                    "history_task_id": t.history_task_id,
                }
                for t in self.tasks
            ],
            "execution_strategy": {
                "parallel_groups": self.execution_strategy.parallel_groups,
                "critical_path": self.execution_strategy.critical_path,
                "estimated_cost": self.execution_strategy.estimated_cost,
            },
            "context": self.context,
            "global_status": self.global_status,
            "clarification_question": self.clarification_question,
            "clarification_options": self.clarification_options,
        }
