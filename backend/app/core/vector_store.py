"""
ChromaDB 向量库封装 —— JD 知识库的向量存储与检索

设计决策：
1. 使用 ChromaDB 内存模式（开发阶段），生产环境应切换为 PersistentClient
2. collection 名为 jd_knowledge，metadata 包含 jd_id/company/position 等，支持灵活过滤
3. 入库时自动向量化，检索时自动 embed 查询文本
4. 提供两种 chunk 策略的对比接口，便于 A/B 实验
"""

import logging
from typing import List, Optional

from app.core.chunking import Chunk, chunk_fixed_size, chunk_semantic
from app.core.embedding import EmbeddingClient

logger = logging.getLogger(__name__)

# 延迟导入 chromadb，避免模块未安装时直接报错
try:
    import chromadb
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False
    logger.warning("[VectorStore] chromadb not installed, vector storage is disabled")


class VectorStore:
    """
    JD 知识库向量存储。

    封装 ChromaDB 的 collection 操作，提供高层次的 add/query/delete 接口。
    """

    COLLECTION_NAME = "jd_knowledge"

    def __init__(self, embedding_client: Optional[EmbeddingClient] = None):
        self.embedding_client = embedding_client
        self._client = None
        self._collection = None

        if _CHROMA_AVAILABLE:
            try:
                # 使用持久化模式，数据保存在 backend/data/chroma_db
                import os
                persist_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "chroma_db")
                persist_dir = os.path.abspath(persist_dir)
                os.makedirs(persist_dir, exist_ok=True)
                self._client = chromadb.PersistentClient(path=persist_dir)
                self._collection = self._client.get_or_create_collection(
                    name=self.COLLECTION_NAME,
                    metadata={"description": "JD knowledge base for career matching"},
                )
                count = self._collection.count()
                logger.info(f"[VectorStore] collection '{self.COLLECTION_NAME}' ready (persist_dir={persist_dir}, existing_docs={count})")
            except Exception as e:
                logger.error(f"[VectorStore] failed to init chromadb: {e}")
        else:
            logger.warning("[VectorStore] chromadb unavailable, all operations are no-ops")

    # ──────────────────────────── 入库 ────────────────────────────

    async def add_jd(
        self,
        jd_schema: dict,
        strategy: str = "semantic",
    ) -> List[str]:
        """
        将 JD 切分为 chunks，向量化后入库。

        Args:
            jd_schema: 结构化 JD 字典
            strategy: 切分策略，"semantic"（推荐）或 "fixed"（对比实验）

        Returns:
            入库的 chunk ID 列表
        """
        if not self._collection:
            logger.warning("[VectorStore] add_jd skipped: chromadb not available")
            return []

        # 1. 切分
        if strategy == "fixed":
            chunks = chunk_fixed_size(jd_schema)
        else:
            chunks = chunk_semantic(jd_schema)

        if not chunks:
            logger.warning(f"[VectorStore] add_jd: no chunks generated for jd_id={jd_schema.get('jd_id')}")
            return []

        # 打印每个 chunk 的详情（供调试观察）
        logger.info(f"[VectorStore] ====== Chunk Details (jd_id={jd_schema.get('jd_id')}) ======")
        for i, c in enumerate(chunks):
            meta = c.metadata
            preview = c.content[:80].replace('\n', ' ')
            logger.info(
                f"  chunk[{i}] section={meta.get('section','?'):20s} "
                f"priority={meta.get('priority','-'):6s} "
                f"len={len(c.content):4d} | {preview}..."
            )
        logger.info(f"[VectorStore] ====== End Chunk Details ======")

        # 2. 向量化
        texts = [c.content for c in chunks]
        if self.embedding_client:
            embeddings = await self.embedding_client.embed(texts)
        else:
            logger.warning("[VectorStore] add_jd: no embedding client, skipping vectors")
            embeddings = []

        # 3. 构造 ChromaDB 数据
        jd_id = jd_schema.get("jd_id", "")
        chunk_ids = [f"{jd_id}_chunk_{i}" for i in range(len(chunks))]
        documents = texts
        metadatas = [c.metadata for c in chunks]

        # 4. 入库
        try:
            if embeddings:
                self._collection.add(
                    ids=chunk_ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )
            else:
                # 无 embedding 时只存文本（ChromaDB 会自动用默认 embedding，但质量差）
                self._collection.add(
                    ids=chunk_ids,
                    documents=documents,
                    metadatas=metadatas,
                )
            logger.info(f"[VectorStore] add_jd jd_id={jd_id} strategy={strategy} chunks={len(chunks)}")
        except Exception as e:
            logger.error(f"[VectorStore] add_jd failed: {e}")
            return []

        return chunk_ids

    # ──────────────────────────── 检索 ────────────────────────────

    async def query(
        self,
        text: str,
        filters: Optional[dict] = None,
        top_k: int = 5,
    ) -> List[dict]:
        """
        向量检索 + 元数据过滤。

        Args:
            text: 查询文本（如用户问题或简历片段）
            filters: 元数据过滤条件，如 {"company": "ByteDance"}
            top_k: 返回 top_k 个结果

        Returns:
            检索结果列表，每个元素包含 chunk 内容、metadata、距离
        """
        if not self._collection:
            logger.warning("[VectorStore] query skipped: chromadb not available")
            return []

        # 向量化查询文本
        query_embedding = None
        if self.embedding_client:
            embeddings = await self.embedding_client.embed([text])
            if embeddings:
                query_embedding = embeddings[0]

        try:
            kwargs = {
                "n_results": top_k,
            }
            if query_embedding:
                kwargs["query_embeddings"] = [query_embedding]
            else:
                # 无 embedding 时退化为纯文本检索
                kwargs["query_texts"] = [text]

            if filters:
                # ChromaDB where 子句格式：单字段直接传，多字段需用 $and 包装
                if len(filters) > 1:
                    kwargs["where"] = {"$and": [{k: v} for k, v in filters.items()]}
                else:
                    kwargs["where"] = filters

            results = self._collection.query(**kwargs)

            # 整理为结构化输出
            output = []
            for i in range(len(results["ids"][0])):
                output.append({
                    "chunk_id": results["ids"][0][i],
                    "content": results["documents"][0][i] if results.get("documents") else None,
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else None,
                    "distance": results["distances"][0][i] if results.get("distances") else None,
                })
            return output
        except Exception as e:
            logger.error(f"[VectorStore] query failed: {e}")
            return []

    # ──────────────────────────── 删除 ────────────────────────────

    def delete_jd(self, jd_id: str) -> bool:
        """
        按 jd_id 删除所有相关 chunks。

        Args:
            jd_id: JD 唯一标识

        Returns:
            是否成功
        """
        if not self._collection:
            logger.warning("[VectorStore] delete_jd skipped: chromadb not available")
            return False

        try:
            self._collection.delete(where={"jd_id": jd_id})
            logger.info(f"[VectorStore] delete_jd jd_id={jd_id}")
            return True
        except Exception as e:
            logger.error(f"[VectorStore] delete_jd failed: {e}")
            return False
