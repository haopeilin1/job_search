"""
CrossEncoder 重排序模块 —— 对召回结果做精排

使用 BAAI/bge-reranker-base（通过 ModelScope 镜像下载），
对 (query, chunk) 对打分，输出 top-k 最相关结果。
"""

import logging
import os
from pathlib import Path
from typing import List, Tuple

from app.core.config import settings

logger = logging.getLogger(__name__)

# 模型全局单例（懒加载）
_reranker_model = None
_reranker_tokenizer = None


def _get_model_cache_dir() -> str:
    """模型缓存目录：backend/data/models"""
    base = Path(__file__).resolve().parent.parent.parent / "data" / "models"
    base.mkdir(parents=True, exist_ok=True)
    return str(base)


def _load_reranker(model_name: str = settings.RERANKER_MODEL):
    """加载 CrossEncoder 模型（优先从本地缓存，否则通过 ModelScope 下载）"""
    global _reranker_model, _reranker_tokenizer
    if _reranker_model is not None:
        return _reranker_model, _reranker_tokenizer

    cache_dir = _get_model_cache_dir()
    local_path = os.path.join(cache_dir, model_name)

    # 1. 检查本地是否已有缓存
    if os.path.exists(local_path) and os.path.isdir(local_path):
        model_path = local_path
        logger.info(f"[Reranker] 使用本地缓存模型: {model_path}")
    else:
        # 2. 通过 ModelScope 下载
        logger.info(f"[Reranker] 本地未找到模型，尝试通过 ModelScope 下载: {model_name}")
        try:
            from modelscope import snapshot_download
            model_path = snapshot_download(
                model_name,
                cache_dir=cache_dir,
            )
            logger.info(f"[Reranker] 模型下载完成: {model_path}")
        except Exception as e:
            logger.error(f"[Reranker] ModelScope 下载失败: {e}")
            raise RuntimeError(f"无法加载重排序模型 {model_name}: {e}")

    # 3. 加载模型和 tokenizer
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        model.eval()

        # CPU 推理即可（模型不大）
        if not torch.cuda.is_available():
            logger.info("[Reranker] 使用 CPU 推理")
        else:
            model = model.cuda()
            logger.info("[Reranker] 使用 CUDA 推理")

        _reranker_model = model
        _reranker_tokenizer = tokenizer
        logger.info(f"[Reranker] 模型加载完成: {model_name}")
        return model, tokenizer

    except Exception as e:
        logger.error(f"[Reranker] 模型加载失败: {e}")
        raise RuntimeError(f"无法加载重排序模型: {e}")


async def rerank(query: str, candidates: List[dict],
                 top_k: int = 10,
                 max_length: int = 512,
                 batch_size: int = 8) -> List[Tuple[int, float]]:
    """
    对召回候选进行 CrossEncoder 重排序。

    Args:
        query: 用户查询文本
        candidates: 候选 chunk 列表，每个元素需包含 "content" 字段
        top_k: 返回 top_k 个结果
        max_length: tokenizer 最大长度
        batch_size: 推理 batch size

    Returns:
        [(候选在 candidates 中的原始索引, 重排序分数), ...]，按分数降序
    """
    if not candidates:
        return []

    model, tokenizer = _load_reranker()

    import torch

    # 1. 构造 (query, doc) 对
    pairs = [(query, c.get("content", "")[:2000]) for c in candidates]  # 截断过长文本

    # 2. batch 推理
    all_scores = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            inputs = tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            outputs = model(**inputs)
            scores = outputs.logits.view(-1).float()
            # 如果 logits 是 sigmoid 前输出，取 sigmoid 得到 [0,1] 概率
            if scores.min() < 0 or scores.max() > 1:
                scores = torch.sigmoid(scores)
            all_scores.extend(scores.cpu().tolist())

    # 3. 按分数排序
    indexed_scores = list(enumerate(all_scores))
    indexed_scores.sort(key=lambda x: x[1], reverse=True)

    logger.info(
        f"[Reranker] 重排序完成 | candidates={len(candidates)} -> top={min(top_k, len(indexed_scores))} | "
        f"best_score={indexed_scores[0][1]:.4f}"
    )
    return indexed_scores[:top_k]
