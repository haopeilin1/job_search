"""
长期记忆持久化层 —— 基于 SQLite（标准库，零额外依赖）

存储内容：
- long_term_memory：用户画像（实体、偏好、简历指纹）
- session_memory：会话元数据（user_id 关联、当前话题、最后活跃时间）

数据库文件：backend/data/memory.db
"""

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional, Dict, Any

from app.core.memory import LongTermMemory

logger = logging.getLogger(__name__)

# 数据库文件路径（与 ChromaDB 同目录）
_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "memory.db"


def _get_conn() -> sqlite3.Connection:
    """获取数据库连接（自动创建文件和表）"""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _init_tables(conn)
    return conn


def _init_tables(conn: sqlite3.Connection):
    """初始化表结构（支持迁移旧表）"""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS long_term_memory (
            user_id TEXT PRIMARY KEY,
            entities TEXT NOT NULL DEFAULT '{}',
            preferences TEXT NOT NULL DEFAULT '{}',
            resume_fingerprint TEXT DEFAULT '',
            topic_flags TEXT NOT NULL DEFAULT '{}',
            last_updated REAL
        );

        CREATE TABLE IF NOT EXISTS session_memory (
            session_id TEXT PRIMARY KEY,
            user_id TEXT,
            current_topic TEXT DEFAULT 'general',
            evidence_cache_query TEXT DEFAULT '',
            created_at REAL,
            last_active REAL
        );

        CREATE INDEX IF NOT EXISTS idx_session_user ON session_memory(user_id);
    """)

    # 迁移：旧表可能没有 dialogue_history 字段
    try:
        conn.execute("SELECT dialogue_history FROM session_memory LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE session_memory ADD COLUMN dialogue_history TEXT DEFAULT '[]'")

    conn.commit()


# ═══════════════════════════════════════════════════════
# 1. LongTermMemory CRUD
# ═══════════════════════════════════════════════════════

def save_long_term_memory(lt: LongTermMemory) -> bool:
    """保存或更新用户长期记忆"""
    try:
        conn = _get_conn()
        conn.execute(
            """
            INSERT INTO long_term_memory (user_id, entities, preferences, resume_fingerprint, topic_flags, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                entities=excluded.entities,
                preferences=excluded.preferences,
                resume_fingerprint=excluded.resume_fingerprint,
                topic_flags=excluded.topic_flags,
                last_updated=excluded.last_updated
            """,
            (
                lt.user_id,
                json.dumps(lt.entities, ensure_ascii=False),
                json.dumps(lt.preferences, ensure_ascii=False),
                lt.resume_fingerprint,
                json.dumps(lt.topic_flags, ensure_ascii=False),
                lt.last_updated,
            ),
        )
        conn.commit()
        conn.close()
        logger.info(f"[DB] LongTermMemory saved | user_id={lt.user_id}")
        return True
    except Exception as e:
        logger.error(f"[DB] save_long_term_memory failed: {e}")
        return False


def load_long_term_memory(user_id: str) -> Optional[LongTermMemory]:
    """从数据库加载用户长期记忆"""
    try:
        conn = _get_conn()
        row = conn.execute(
            "SELECT * FROM long_term_memory WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        conn.close()

        if row is None:
            return None

        return LongTermMemory(
            user_id=row["user_id"],
            entities=json.loads(row["entities"]),
            preferences=json.loads(row["preferences"]),
            resume_fingerprint=row["resume_fingerprint"],
            topic_flags=json.loads(row["topic_flags"]),
            last_updated=row["last_updated"],
        )
    except Exception as e:
        logger.error(f"[DB] load_long_term_memory failed: {e}")
        return None


def delete_long_term_memory(user_id: str) -> bool:
    """删除用户长期记忆"""
    try:
        conn = _get_conn()
        conn.execute("DELETE FROM long_term_memory WHERE user_id = ?", (user_id,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"[DB] delete_long_term_memory failed: {e}")
        return False


# ═══════════════════════════════════════════════════════
# 2. Session 元数据 CRUD（仅保存关系映射，不存完整对话）
# ═══════════════════════════════════════════════════════

def save_session_meta(session_id: str, user_id: Optional[str] = None,
                       current_topic: str = "general",
                       evidence_cache_query: str = "",
                       dialogue_history: Optional[list] = None) -> bool:
    """保存会话元数据（含完整对话历史）"""
    try:
        conn = _get_conn()
        now = time.time()
        conn.execute(
            """
            INSERT INTO session_memory (session_id, user_id, current_topic, evidence_cache_query, dialogue_history, created_at, last_active)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                user_id=excluded.user_id,
                current_topic=excluded.current_topic,
                evidence_cache_query=excluded.evidence_cache_query,
                dialogue_history=excluded.dialogue_history,
                last_active=excluded.last_active
            """,
            (session_id, user_id, current_topic, evidence_cache_query,
             json.dumps(dialogue_history or [], ensure_ascii=False), now, now),
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"[DB] save_session_meta failed: {e}")
        return False


def load_session_meta(session_id: str) -> Optional[Dict[str, Any]]:
    """加载会话元数据"""
    try:
        conn = _get_conn()
        row = conn.execute(
            "SELECT * FROM session_memory WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        conn.close()

        if row is None:
            return None

        return dict(row)
    except Exception as e:
        logger.error(f"[DB] load_session_meta failed: {e}")
        return None


def delete_session_meta(session_id: str) -> bool:
    """删除会话元数据"""
    try:
        conn = _get_conn()
        conn.execute("DELETE FROM session_memory WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"[DB] delete_session_meta failed: {e}")
        return False


def list_sessions_by_user(user_id: str) -> list[Dict[str, Any]]:
    """列出某用户的所有会话"""
    try:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM session_memory WHERE user_id = ? ORDER BY last_active DESC",
            (user_id,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"[DB] list_sessions_by_user failed: {e}")
        return []


def list_all_long_term_memory(limit: int = 100, offset: int = 0) -> list[Dict[str, Any]]:
    """列出所有用户长期记忆（分页）"""
    try:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT user_id, resume_fingerprint, last_updated FROM long_term_memory ORDER BY last_updated DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"[DB] list_all_long_term_memory failed: {e}")
        return []


def list_all_sessions(limit: int = 100, offset: int = 0) -> list[Dict[str, Any]]:
    """列出所有会话元数据（分页）"""
    try:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT session_id, user_id, current_topic, last_active FROM session_memory ORDER BY last_active DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"[DB] list_all_sessions failed: {e}")
        return []


def load_session_dialogue_history(session_id: str) -> list[dict]:
    """加载会话的完整对话历史"""
    try:
        conn = _get_conn()
        row = conn.execute(
            "SELECT dialogue_history FROM session_memory WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        conn.close()
        if row and row["dialogue_history"]:
            return json.loads(row["dialogue_history"])
        return []
    except Exception as e:
        logger.error(f"[DB] load_session_dialogue_history failed: {e}")
        return []
