"""
长期记忆管理接口 —— 供管理员查看/操作用户画像和会话数据

前缀：/api/v1/admin/memory
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.db import (
    load_long_term_memory,
    delete_long_term_memory,
    load_session_meta,
    delete_session_meta,
    list_all_long_term_memory,
    list_all_sessions,
    list_sessions_by_user,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/admin/memory", tags=["memory-admin"])


# ═══════════════════════════════════════════════════════
# 响应模型
# ═══════════════════════════════════════════════════════

class LongTermMemoryListItem(BaseModel):
    user_id: str
    resume_fingerprint: str = ""
    last_updated: Optional[float] = None


class LongTermMemoryDetail(BaseModel):
    user_id: str
    entities: dict = Field(default_factory=dict)
    preferences: dict = Field(default_factory=dict)
    resume_fingerprint: str = ""
    topic_flags: dict = Field(default_factory=dict)
    last_updated: Optional[float] = None


class SessionListItem(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    current_topic: str = ""
    last_active: Optional[float] = None


class SessionDetail(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    current_topic: str = ""
    evidence_cache_query: str = ""
    created_at: Optional[float] = None
    last_active: Optional[float] = None


class DeleteResponse(BaseModel):
    success: bool
    message: str


class UpdateLongTermMemoryRequest(BaseModel):
    entities: Optional[dict] = None
    preferences: Optional[dict] = None
    resume_fingerprint: Optional[str] = None
    topic_flags: Optional[dict] = None


# ═══════════════════════════════════════════════════════
# 1. 长期记忆管理
# ═══════════════════════════════════════════════════════

@router.get("/users", response_model=list[LongTermMemoryListItem])
async def list_users(limit: int = 100, offset: int = 0):
    """列出所有用户长期记忆（分页）"""
    rows = list_all_long_term_memory(limit=limit, offset=offset)
    return [LongTermMemoryListItem(**r) for r in rows]


@router.get("/users/{user_id}", response_model=LongTermMemoryDetail)
async def get_user_memory(user_id: str):
    """查看指定用户的完整长期记忆"""
    lt = load_long_term_memory(user_id)
    if lt is None:
        raise HTTPException(status_code=404, detail=f"用户 {user_id} 的长期记忆不存在")
    return LongTermMemoryDetail(
        user_id=lt.user_id,
        entities=lt.entities,
        preferences=lt.preferences,
        resume_fingerprint=lt.resume_fingerprint,
        topic_flags=lt.topic_flags,
        last_updated=lt.last_updated,
    )


@router.put("/users/{user_id}", response_model=LongTermMemoryDetail)
async def update_user_memory(user_id: str, req: UpdateLongTermMemoryRequest):
    """手动修正指定用户的长期记忆（部分更新）"""
    lt = load_long_term_memory(user_id)
    if lt is None:
        raise HTTPException(status_code=404, detail=f"用户 {user_id} 的长期记忆不存在")

    import time
    if req.entities is not None:
        lt.entities = req.entities
    if req.preferences is not None:
        lt.preferences = req.preferences
    if req.resume_fingerprint is not None:
        lt.resume_fingerprint = req.resume_fingerprint
    if req.topic_flags is not None:
        lt.topic_flags = req.topic_flags
    lt.last_updated = time.time()

    from app.core.db import save_long_term_memory
    save_long_term_memory(lt)

    return LongTermMemoryDetail(
        user_id=lt.user_id,
        entities=lt.entities,
        preferences=lt.preferences,
        resume_fingerprint=lt.resume_fingerprint,
        topic_flags=lt.topic_flags,
        last_updated=lt.last_updated,
    )


@router.delete("/users/{user_id}", response_model=DeleteResponse)
async def delete_user_memory(user_id: str):
    """删除指定用户的长期记忆"""
    ok = delete_long_term_memory(user_id)
    if ok:
        return DeleteResponse(success=True, message=f"用户 {user_id} 的长期记忆已删除")
    raise HTTPException(status_code=500, detail="删除失败")


# ═══════════════════════════════════════════════════════
# 2. 会话管理
# ═══════════════════════════════════════════════════════

@router.get("/sessions", response_model=list[SessionListItem])
async def list_sessions(limit: int = 100, offset: int = 0):
    """列出所有会话元数据（分页）"""
    rows = list_all_sessions(limit=limit, offset=offset)
    return [SessionListItem(**r) for r in rows]


@router.get("/sessions/user/{user_id}", response_model=list[SessionListItem])
async def list_sessions_by_user_id(user_id: str):
    """列出某用户的所有会话"""
    rows = list_sessions_by_user(user_id)
    return [SessionListItem(**r) for r in rows]


@router.get("/sessions/{session_id}", response_model=SessionDetail)
async def get_session(session_id: str):
    """查看指定会话的元数据"""
    meta = load_session_meta(session_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")
    return SessionDetail(**meta)


@router.delete("/sessions/{session_id}", response_model=DeleteResponse)
async def delete_session(session_id: str):
    """删除指定会话的元数据"""
    ok = delete_session_meta(session_id)
    if ok:
        return DeleteResponse(success=True, message=f"会话 {session_id} 已删除")
    raise HTTPException(status_code=500, detail="删除失败")
