"""
统一埋点追踪器（Telemetry）。

设计原则：
1. 零侵入：业务代码只需一行 tracker.track()
2. 结构化：所有事件输出为 JSON，便于脚本解析
3. 可扩展：当前写日志，后续可切 ClickHouse/Kafka
4. 上下文透传：session_id / turn_id / eval_context 沿调用链传递
"""

import json
import logging
import time
from typing import Optional, Dict, Any

logger = logging.getLogger("telemetry")


class TelemetryTracker:
    """
    埋点追踪器实例。
    每个请求独立创建，不共享状态。
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        turn_id: Optional[int] = None,
        eval_context: Optional[Dict[str, Any]] = None,
    ):
        self.session_id = session_id
        self.turn_id = turn_id
        self.eval_context = eval_context or {}

    def track(self, event_type: str, payload: Dict[str, Any]):
        """
        记录单个埋点事件。

        Args:
            event_type: 事件类型（intent_classified / plan_generated / ...）
            payload: 事件字段（会自动合并 session_id / turn_id / eval_context）
        """
        event = {
            "event": event_type,
            "session_id": payload.get("session_id") or self.session_id,
            "turn_id": payload.get("turn_id") or self.turn_id,
            "timestamp": time.time(),
            "eval_context": self.eval_context,
            **{k: v for k, v in payload.items() if k not in ("session_id", "turn_id")},
        }
        # 统一输出到 telemetry 专用 logger
        logger.info(json.dumps(event, ensure_ascii=False, default=str))

    def bind(
        self,
        session_id: Optional[str] = None,
        turn_id: Optional[int] = None,
        eval_context: Optional[Dict[str, Any]] = None,
    ):
        """动态绑定上下文（用于从 request 中读取后更新）"""
        if session_id:
            self.session_id = session_id
        if turn_id:
            self.turn_id = turn_id
        if eval_context:
            self.eval_context = eval_context


def create_tracker(
    session_id: Optional[str] = None,
    turn_id: Optional[int] = None,
    eval_context: Optional[Dict[str, Any]] = None,
) -> TelemetryTracker:
    """工厂函数：创建新的 Tracker 实例"""
    return TelemetryTracker(session_id=session_id, turn_id=turn_id, eval_context=eval_context)
