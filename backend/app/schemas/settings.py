from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class LLMModelConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    base_url: str = Field(..., alias="baseUrl", description="API 基础地址")
    api_key: str = Field(..., alias="apiKey", description="API 密钥")
    model: str = Field(..., description="模型名称")


class LLMConfigSchema(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # ── 按功能难度/成本敏感度分层配置 ──
    chat: LLMModelConfig = Field(..., description="对话回复模型（最终回复生成）")
    core: LLMModelConfig = Field(..., description="核心业务模型（匹配分析、面试题、简历解析）")
    planner: LLMModelConfig = Field(..., description="规划推理模型（意图识别、query改写、plan生成、澄清）")
    memory: LLMModelConfig = Field(..., description="记忆管理模型（压缩、提取、话题检测）")
    vision: LLMModelConfig = Field(..., description="多模态分析模型（图片OCR、解析）")
    # embedding 为可选字段：
    #   - 测试时从 .env 加载，后端自动填充
    #   - 生产时前端暂不展示 UI，后端有默认值兜底
    embedding: Optional[LLMModelConfig] = Field(None, description="Embedding 模型配置（可选，默认复用 chat）")
