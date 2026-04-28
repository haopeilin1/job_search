"""
Embedding 模型封装 —— 将文本转换为向量

设计决策：
1. 优先使用 OpenAI 兼容 API（text-embedding-3-small），复用用户配置的 base_url/api_key
2. 如果 API 调用失败（如提供商不支持 embedding），自动降级为 mock embedding（仅用于开发测试）
3. 批量处理接口，减少 API 调用次数
"""

import logging
import random
from typing import List, Optional

import httpx

from app.core.state import llm_config_store
from app.core.config import settings

logger = logging.getLogger(__name__)

# 默认 embedding 模型（从配置读取，避免与 config.py 不一致）
DEFAULT_EMBEDDING_MODEL = settings.EMBEDDING_MODEL
# Embedding 维度（text-embedding-3-small 为 1536；mock 时固定为 384 节省空间）
EMBEDDING_DIM = 1536
MOCK_DIM = 1024


class EmbeddingClient:
    """
    Embedding 客户端。

    支持两种模式：
    1. API 模式：调用 OpenAI 兼容的 embedding API
    2. Mock 模式：生成随机向量（API 不可用时降级）
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or ""
        self.model = model or DEFAULT_EMBEDDING_MODEL

    @classmethod
    def from_config(cls) -> "EmbeddingClient":
        """从全局状态读取 embedding 或 chat 配置创建客户端"""
        # 优先使用独立的 embedding 配置（.env 或用户界面设置）
        if llm_config_store.embedding:
            cfg = llm_config_store.embedding
            logger.info(f"[EmbeddingClient] using dedicated embedding config: model={cfg.model}")
        else:
            # fallback 到 chat 配置（兼容旧逻辑）
            cfg = llm_config_store.chat
            logger.info(f"[EmbeddingClient] embedding config not set, falling back to chat config: model={settings.EMBEDDING_MODEL}")
        return cls(base_url=cfg.base_url, api_key=cfg.api_key, model=cfg.model if llm_config_store.embedding else settings.EMBEDDING_MODEL)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        批量将文本转换为向量。
        支持分批处理（兼容 dashscope 等提供商的单次 10 条限制）。

        Args:
            texts: 文本列表

        Returns:
            向量列表，每个向量是 float 数组
        """
        if not texts:
            return []

        # 过滤空文本
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            return []

        # 分批处理，每批最多 10 条（dashscope 等提供商限制）
        BATCH_SIZE = 10
        all_results = []

        for i in range(0, len(valid_texts), BATCH_SIZE):
            batch = valid_texts[i:i + BATCH_SIZE]
            try:
                batch_results = await self._api_embed(batch)
                all_results.extend(batch_results)
            except Exception as e:
                logger.warning(f"[EmbeddingClient] API embed failed for batch {i // BATCH_SIZE + 1}: {e}, falling back to mock")
                batch_results = self._mock_embed(batch)
                all_results.extend(batch_results)

        return all_results

    async def _api_embed(self, texts: List[str]) -> List[List[float]]:
        """调用 OpenAI 兼容 embedding API"""
        url = f"{self.base_url}/embeddings"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "input": texts,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

            # 按 index 排序
            embeddings = sorted(data["data"], key=lambda x: x["index"])
            result = [item["embedding"] for item in embeddings]

            logger.info(
                f"[EmbeddingClient] model={self.model} texts={len(texts)} "
                f"dim={len(result[0]) if result else 0}"
            )
            return result

    def _mock_embed(self, texts: List[str]) -> List[List[float]]:
        """
        Mock embedding：基于文本 hash 生成确定性伪随机向量。

        注意：此模式仅用于开发和测试，向量没有语义意义，检索结果随机。
        生产环境必须配置真实的 embedding API。
        """
        logger.warning("[EmbeddingClient] USING MOCK EMBEDDING — vectors are random, retrieval quality is zero")
        result = []
        for text in texts:
            # 使用文本 hash 作为种子，保证同一文本总是得到同一向量
            seed = hash(text) % (2 ** 31)
            rng = random.Random(seed)
            vec = [rng.uniform(-1, 1) for _ in range(MOCK_DIM)]
            # 归一化
            norm = sum(x * x for x in vec) ** 0.5
            if norm > 0:
                vec = [x / norm for x in vec]
            result.append(vec)
        return result
