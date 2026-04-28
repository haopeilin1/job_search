#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 CrossEncoder 批量重排序功能

验证不同 batch_size 和候选数量下的重排序一致性。
"""

import asyncio
import time

from app.core.reranker import rerank


SAMPLE_CANDIDATES = [
    {"content": "字节跳动算法工程师岗位要求：精通 Python、Go，熟悉推荐系统"},
    {"content": "阿里巴巴产品经理 JD：负责电商产品线，需要数据分析能力"},
    {"content": "腾讯后端开发岗位：高并发系统设计经验，熟悉分布式架构"},
    {"content": "字节跳动数据分析师：SQL、Python、数据可视化工具"},
    {"content": "美团算法岗：机器学习、深度学习、NLP 经验"},
    {"content": "京东前端工程师：React、Vue、前端性能优化"},
    {"content": "滴滴后端开发：微服务、Kubernetes、云原生"},
    {"content": "网易游戏开发：Unity、C++、游戏引擎"},
    {"content": "百度 NLP 算法：BERT、GPT、大模型微调"},
    {"content": "小红书推荐算法：协同过滤、向量召回"},
]


async def test_basic_rerank():
    """基础重排序测试"""
    print("=" * 60)
    print("Test 1: 基础重排序")
    query = "Python 算法工程师"
    results = await rerank(query, SAMPLE_CANDIDATES, top_k=5)
    print(f"Query: {query}")
    print(f"Candidates: {len(SAMPLE_CANDIDATES)}")
    for idx, score in results:
        print(f"  [{idx}] score={score:.4f} | {SAMPLE_CANDIDATES[idx]['content'][:50]}...")
    assert len(results) == 5
    assert results[0][1] >= results[-1][1]  # 降序
    print("  PASSED")


async def test_batch_size_consistency():
    """验证不同 batch_size 结果一致"""
    print("=" * 60)
    print("Test 2: batch_size 一致性")
    query = "后端开发 微服务"

    results_b1 = await rerank(query, SAMPLE_CANDIDATES, top_k=5, batch_size=1)
    results_b4 = await rerank(query, SAMPLE_CANDIDATES, top_k=5, batch_size=4)
    results_b8 = await rerank(query, SAMPLE_CANDIDATES, top_k=5, batch_size=8)

    # 检查索引顺序一致
    idx_b1 = [i for i, _ in results_b1]
    idx_b4 = [i for i, _ in results_b4]
    idx_b8 = [i for i, _ in results_b8]

    print(f"  batch=1 indices: {idx_b1}")
    print(f"  batch=4 indices: {idx_b4}")
    print(f"  batch=8 indices: {idx_b8}")

    assert idx_b1 == idx_b4 == idx_b8, "不同 batch_size 结果应一致"
    print("  PASSED")


async def test_large_candidates():
    """大批量候选测试"""
    print("=" * 60)
    print("Test 3: 大批量候选 (100+)")
    large_candidates = [
        {"content": f"JD #{i}: {'Python' if i % 3 == 0 else 'Java'} 开发工程师，{'算法' if i % 5 == 0 else '业务'}方向"}
        for i in range(120)
    ]
    query = "Python 算法"

    start = time.time()
    results = await rerank(query, large_candidates, top_k=10, batch_size=16)
    elapsed = time.time() - start

    print(f"  Candidates: 120, batch_size=16")
    print(f"  Top-10 indices: {[i for i, _ in results]}")
    print(f"  Elapsed: {elapsed:.2f}s")
    assert len(results) == 10
    print("  PASSED")


async def test_empty_candidates():
    """空候选测试"""
    print("=" * 60)
    print("Test 4: 空候选")
    results = await rerank("test", [], top_k=5)
    assert results == []
    print("  PASSED")


async def main():
    print("CrossEncoder 批量重排序测试")
    print()
    await test_basic_rerank()
    await test_batch_size_consistency()
    await test_large_candidates()
    await test_empty_candidates()
    print("=" * 60)
    print("All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
