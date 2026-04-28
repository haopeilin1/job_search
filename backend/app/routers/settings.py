from fastapi import APIRouter

from app.core import state
from app.schemas import LLMConfigSchema

router = APIRouter(prefix="/api/v1/settings", tags=["settings"])


@router.get("/llm", response_model=LLMConfigSchema, response_model_by_alias=True)
async def get_llm_config():
    """获取当前 LLM 配置"""
    return state.llm_config_store


@router.put("/llm", response_model=LLMConfigSchema, response_model_by_alias=True)
async def update_llm_config(config: LLMConfigSchema):
    """更新 LLM 配置"""
    state.llm_config_store = config
    return state.llm_config_store
