#!/usr/bin/env python3
"""
对比两种检索策略的效果：

1. 【混合检索+重排序】（现有机制）
   - 向量路独立召回 top-20
   - BM25 路独立召回 top-20
   - 合并去重
   - 两路分数 min-max 归一化后加权混合（70%向量 + 30%BM25）
   - 取混合分数 top-20 作为重排序候选池
   - CrossEncoder 重排序，输出 top-10

2. 【纯向量检索+无重排序】（对比机制）
   - 直接向量检索，top-k=10
   - 无 BM25 混合
   - 无 CrossEncoder 重排序

用法:
    cd backend
    python scripts/compare_retrieval_strategies.py
    python scripts/compare_retrieval_strategies.py --queries "问题A" "问题B" "问题C"
"""

import argparse
import asyncio
import json
import sys
import datetime
from pathlib import Path

# Windows 控制台默认 GBK，强制设置 UTF-8 避免中文乱码
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# 把 backend 加入路径，确保能 import app.*
BACKEND_ROOT = Path(__file__).parent.parent.resolve()
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.vector_store import VectorStore
from app.core.embedding import EmbeddingClient
from app.core.config import settings


# ═══════════════════════════════════════════════════════
# 1. 纯向量检索（无 BM25、无重排序）
# ═══════════════════════════════════════════════════════

async def pure_vector_retrieve(query: str, top_k: int = 10):
    """纯向量检索，直接返回 VectorStore.query 的结果"""
    vs = VectorStore()
    vs.embedding_client = EmbeddingClient.from_config()

    results = await vs.query(query, filters=None, top_k=top_k)
    return results


# ═══════════════════════════════════════════════════════
# 2. 调用现有混合检索+重排序
# ═══════════════════════════════════════════════════════

async def hybrid_retrieve(query: str):
    """调用现有的混合检索+重排序逻辑"""
    from app.core.tools import _kb_retrieve_stub

    result = await _kb_retrieve_stub(query=query, company=None, position=None, top_k=None)
    if result.success:
        return result.data
    else:
        return {"error": result.error}


# ═══════════════════════════════════════════════════════
# 3. 打印辅助函数
# ═══════════════════════════════════════════════════════

def print_divider(title: str, width: int = 90):
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_chunk(idx: int, chunk: dict, show_scores: bool = True):
    """打印单个 chunk 的完整信息"""
    chunk_id = chunk.get("chunk_id", "N/A")
    content = chunk.get("content", "")
    metadata = chunk.get("metadata", {}) or {}

    company = metadata.get("company", "N/A")
    position = metadata.get("position", "N/A")
    section = metadata.get("section", "N/A")
    jd_id = metadata.get("jd_id", "N/A")

    print(f"\n  ── [{idx}] chunk_id={chunk_id} ──")
    print(f"      jd_id={jd_id} | company={company} | position={position} | section={section}")

    if show_scores:
        distance = chunk.get("distance")
        bm25_score = chunk.get("bm25_score")
        hybrid_score = chunk.get("hybrid_score")
        rerank_score = chunk.get("rerank_score")

        scores = []
        if distance is not None:
            scores.append(f"vec_distance={distance:.6f}")
        if bm25_score is not None:
            scores.append(f"bm25={bm25_score:.6f}")
        if hybrid_score is not None:
            scores.append(f"hybrid={hybrid_score:.6f}")
        if rerank_score is not None:
            scores.append(f"rerank={rerank_score:.6f}")

        if scores:
            print(f"      scores: {' | '.join(scores)}")

    # 打印完整内容，不截断
    print(f"      content:")
    for line in content.split("\n"):
        print(f"        {line}")
    print(f"      ── end ──")


def print_strategy_header(strategy_name: str, query: str, extra_info: dict = None):
    print_divider(f"策略: {strategy_name} | query: {query}", width=90)
    if extra_info:
        for k, v in extra_info.items():
            print(f"  {k}: {v}")


# ═══════════════════════════════════════════════════════
# 4. 对比执行
# ═══════════════════════════════════════════════════════

async def compare_for_query(query: str):
    """对单个查询，分别执行两种策略并打印完整结果"""

    print_divider(f"QUERY: {query}", width=90)

    # ── 策略 A: 纯向量检索 top-10 ──
    print_strategy_header("纯向量检索（无 BM25、无重排序）", query,
                          extra_info={"top_k": 10, "description": "直接 VectorStore.query(top_k=10)"})

    vec_results = await pure_vector_retrieve(query, top_k=10)
    print(f"  召回数量: {len(vec_results)}")

    for i, chunk in enumerate(vec_results, 1):
        print_chunk(i, chunk, show_scores=True)

    # ── 策略 B: 混合检索+重排序 ──
    print_strategy_header("混合检索+重排序（现有机制）", query,
                          extra_info={
                              "vec_top_k": settings.RETRIEVAL_VEC_TOP_K,
                              "bm25_top_k": settings.RETRIEVAL_BM25_TOP_K,
                              "hybrid_top_k": settings.RETRIEVAL_TOP_K,
                              "rerank_top_k": settings.RERANKER_TOP_K,
                              "vec_weight": settings.RETRIEVAL_VEC_WEIGHT,
                              "bm25_weight": settings.RETRIEVAL_BM25_WEIGHT,
                              "description": "向量20 + BM25 20 → 去重 → 混合 top-20 → 重排序 top-10",
                          })

    hybrid_data = await hybrid_retrieve(query)

    if "error" in hybrid_data:
        print(f"  [ERROR] 混合检索失败: {hybrid_data['error']}")
        return

    hybrid_chunks = hybrid_data.get("chunks", [])
    pool_size = hybrid_data.get("pool_size", 0)
    vec_only = hybrid_data.get("vec_only", 0)
    bm25_only = hybrid_data.get("bm25_only", 0)
    both = hybrid_data.get("both", 0)
    rerank_info = hybrid_data.get("rerank", {})

    print(f"  混合候选池大小: {pool_size} (vec_only={vec_only}, bm25_only={bm25_only}, both={both})")
    print(f"  重排序信息: {json.dumps(rerank_info, ensure_ascii=False)}")
    print(f"  最终输出数量: {len(hybrid_chunks)}")

    for i, chunk in enumerate(hybrid_chunks, 1):
        print_chunk(i, chunk, show_scores=True)


async def main():
    parser = argparse.ArgumentParser(description="对比两种检索策略的效果")
    parser.add_argument(
        "--queries", "-q",
        nargs="+",
        default=[
            "推荐算法工程师",
            "有 LLM 经验的后端开发",
            "熟悉 TensorFlow 的算法专家",
        ],
        help="批量输入查询问题（空格分隔，带空格的问题请加引号）",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出文件路径（默认自动生成带时间戳的文件）",
    )
    args = parser.parse_args()

    queries = args.queries

    # 构建输出文件路径
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"compare_retrieval_{timestamp}.txt")

    # 使用 utf-8-sig 写入（带 BOM，Windows 记事本打开不会乱码）
    output_file = open(output_path, "w", encoding="utf-8-sig")

    class Tee:
        """同时输出到终端和文件"""
        def __init__(self, stdout, file):
            self.stdout = stdout
            self.file = file
        def write(self, data):
            self.stdout.write(data)
            self.file.write(data)
        def flush(self):
            self.stdout.flush()
            self.file.flush()

    # 暂存原始 stdout，替换成 Tee
    old_stdout = sys.stdout
    tee = Tee(old_stdout, output_file)
    sys.stdout = tee

    try:
        print_divider("检索策略对比实验", width=90)
        print(f"  实验问题数: {len(queries)}")
        print(f"  问题列表:")
        for i, q in enumerate(queries, 1):
            print(f"    {i}. {q}")

        print(f"\n  【策略A】纯向量检索: VectorStore.query(top_k=10)")
        print(f"  【策略B】混合检索+重排序: 向量{settings.RETRIEVAL_VEC_TOP_K} + BM25 {settings.RETRIEVAL_BM25_TOP_K} → 去重 → 混合取top-{settings.RETRIEVAL_TOP_K} → 重排序top-{settings.RERANKER_TOP_K}")
        print(f"  输出文件: {output_path.resolve()}")

        for query in queries:
            await compare_for_query(query)

        print_divider("对比实验结束", width=90)
    finally:
        sys.stdout = old_stdout
        output_file.close()
        print(f"\n[OK] 完整结果已保存到: {output_path.resolve()}")
        print(f"     请用 VS Code、记事本（Win10+）或任何支持 UTF-8 的编辑器打开。")


if __name__ == "__main__":
    asyncio.run(main())
