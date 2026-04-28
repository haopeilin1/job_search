"""
全局运行时状态：LLM 配置与简历内存数据库

双路配置逻辑：
1. 启动时从 .env / 环境变量加载默认值（测试/开发方便）
2. 运行时用户可通过 /api/v1/settings/llm 接口覆盖（生产环境用户自主配置）
3. 若 .env 未填写（留空），则使用空默认值，强制用户从界面填入

简历持久化：
- 内存字典 resumes_db + active_resume_id 在每次变更后自动写入 data/resumes.json
- 启动时自动从 data/resumes.json 恢复
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any

from app.schemas.settings import LLMConfigSchema, LLMModelConfig
from app.core.config import settings

logger = logging.getLogger(__name__)

# ---------- 辅助函数：为各层构建配置，未填写时 fallback 到 chat ----------
def _build_layer_config(base_url: str, api_key: str, model: str, fallback: LLMModelConfig) -> LLMModelConfig:
    """构建分层模型配置，空值 fallback 到 chat 配置"""
    return LLMModelConfig(
        base_url=base_url or fallback.base_url,
        api_key=api_key or fallback.api_key,
        model=model or fallback.model,
    )


# ---------- LLM 配置（启动时从 .env 加载，运行时可通过 API 覆盖） ----------
_chat_cfg = LLMModelConfig(
    base_url=settings.CHAT_BASE_URL,
    api_key=settings.CHAT_API_KEY,
    model=settings.CHAT_MODEL,
)

llm_config_store: LLMConfigSchema = LLMConfigSchema(
    chat=_chat_cfg,
    core=_build_layer_config(settings.CORE_BASE_URL, settings.CORE_API_KEY, settings.CORE_MODEL, _chat_cfg),
    planner=_build_layer_config(settings.PLANNER_BASE_URL, settings.PLANNER_API_KEY, settings.PLANNER_MODEL, _chat_cfg),
    memory=_build_layer_config(settings.MEMORY_BASE_URL, settings.MEMORY_API_KEY, settings.MEMORY_MODEL, _chat_cfg),
    vision=LLMModelConfig(
        base_url=settings.VISION_BASE_URL,
        api_key=settings.VISION_API_KEY,
        model=settings.VISION_MODEL,
    ),
    # Embedding 配置：若 .env 中填写了独立配置则使用，否则为 None（运行时 fallback 到 chat）
    embedding=LLMModelConfig(
        base_url=settings.EMBEDDING_BASE_URL or settings.CHAT_BASE_URL,
        api_key=settings.EMBEDDING_API_KEY or settings.CHAT_API_KEY,
        model=settings.EMBEDDING_MODEL,
    ) if (settings.EMBEDDING_BASE_URL or settings.EMBEDDING_API_KEY) else None,
)


# ---------- 简历持久化 ----------
_RESUME_DATA_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "resumes.json"


def _load_resumes() -> tuple[Dict[str, Dict[str, Any]], Optional[str]]:
    """从磁盘加载简历数据，返回 (resumes_db, active_resume_id)"""
    if not _RESUME_DATA_FILE.exists():
        logger.info(f"[ResumeStore] 持久化文件不存在，跳过加载 | path={_RESUME_DATA_FILE}")
        return {}, None

    try:
        with open(_RESUME_DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            logger.warning(f"[ResumeStore] 持久化文件格式异常（应为 list），跳过加载")
            return {}, None

        db: Dict[str, Dict[str, Any]] = {}
        active_id: Optional[str] = None
        for item in data:
            rid = item.get("id")
            if not rid:
                continue
            db[rid] = item
            # 恢复 active_resume_id
            meta = item.get("parsed_schema", {}).get("meta", {})
            if meta.get("is_active"):
                active_id = rid

        logger.info(f"[ResumeStore] 加载成功 | resumes={len(db)} | active={active_id}")
        return db, active_id
    except Exception as e:
        logger.error(f"[ResumeStore] 加载失败: {e}")
        return {}, None


def _save_resumes(db: Dict[str, Dict[str, Any]], active_id: Optional[str] = None) -> None:
    """将简历数据持久化到磁盘"""
    try:
        # 更新所有记录的 is_active 标记
        for rid, item in db.items():
            ps = item.get("parsed_schema", {})
            meta = ps.get("meta", {})
            if isinstance(meta, dict):
                meta["is_active"] = (rid == active_id)

        records = list(db.values())
        _RESUME_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_RESUME_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        logger.info(f"[ResumeStore] 持久化成功 | resumes={len(records)} | active={active_id}")
    except Exception as e:
        logger.error(f"[ResumeStore] 持久化失败: {e}")


def save_resumes() -> None:
    """外部接口：将当前内存中的简历数据持久化到磁盘"""
    _save_resumes(resumes_db, active_resume_id)


# ---------- 简历内存数据库（启动时自动恢复） ----------
_resumes_db_loaded, _active_id_loaded = _load_resumes()
resumes_db: Dict[str, Dict[str, Any]] = _resumes_db_loaded
active_resume_id: Optional[str] = _active_id_loaded
