#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重建 ChromaDB 向量索引

用法:
    cd backend
    python scripts/rebuild_chroma.py

说明:
    1. 删除损坏的 chroma_db 目录
    2. 从 data/jds.json 加载所有 JD
    3. 重新切分 chunks + 向量化 + 入库
"""

import asyncio
import json
import os
import shutil
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.core.vector_store import VectorStore
from app.core.embedding import EmbeddingClient
from app.core.chunking import chunk_semantic


CHROMA_DIR = os.path.join(BACKEND_DIR, "data", "chroma_db")
JDS_FILE = os.path.join(BACKEND_DIR, "data", "jds.json")


async def main():
    print("=" * 60)
    print("  ChromaDB 向量索引重建工具")
    print("=" * 60)

    # 1. 删除旧的 chromadb
    if os.path.exists(CHROMA_DIR):
        print(f"\n[1/4] 删除旧索引: {CHROMA_DIR}")
        shutil.rmtree(CHROMA_DIR)
        print("      已删除")
    else:
        print(f"\n[1/4] 旧索引不存在，跳过删除")

    # 2. 加载 JD 数据
    print(f"\n[2/4] 加载 JD 数据: {JDS_FILE}")
    if not os.path.exists(JDS_FILE):
        print("      错误: jds.json 不存在!")
        return

    with open(JDS_FILE, "r", encoding="utf-8") as f:
        jds = json.load(f)
    print(f"      共 {len(jds)} 条 JD")

    # 3. 初始化向量库
    print(f"\n[3/4] 初始化 ChromaDB")
    embedding_client = EmbeddingClient.from_config()
    vector_store = VectorStore(embedding_client=embedding_client)
    print(f"      集合: {vector_store.COLLECTION_NAME}")
    print(f"      当前文档数: {vector_store._collection.count()}")

    # 4. 逐条入库
    print(f"\n[4/4] 向量化入库")
    total_chunks = 0
    success_count = 0

    for idx, jd in enumerate(jds, 1):
        company = jd.get("company", "未知")
        position = jd.get("position") or jd.get("title", "未知")
        print(f"\n  [{idx}/{len(jds)}] {company} · {position}")

        try:
            chunk_ids = await vector_store.add_jd(jd, strategy="semantic")
            total_chunks += len(chunk_ids)
            success_count += 1
            print(f"        入库成功 | chunks={len(chunk_ids)}")
        except Exception as e:
            print(f"        入库失败: {e}")

    # 5. 汇总
    final_count = vector_store._collection.count()
    print(f"\n{'=' * 60}")
    print(f"  重建完成")
    print(f"  JD 总数: {len(jds)} | 成功: {success_count} | 失败: {len(jds) - success_count}")
    print(f"  总 chunks: {total_chunks}")
    print(f"  ChromaDB 文档数: {final_count}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
