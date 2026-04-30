"""
验证 embedding 配置恢复后的检索质量。

测试用例：eval_chen_09 — "我的RAG和Embedding经验对哪个岗最有用"

验证维度：
1. embedding API 是否正常（非 mock）
2. 向量检索召回的 JD 是否语义相关
3. 混合召回 vs 纯 BM25 的差异
4. global_rank 输出质量是否改善
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.embedding import EmbeddingClient
from app.core.tools import _kb_retrieve_stub, GlobalRankTool


QUERY = "RAG Embedding 经验 岗位"
RESUME_ID = "cc50dcfb-aeba-41a3-989a-9e485aa2fc3f"


def load_resume(resume_id: str) -> str:
    with open(Path(__file__).resolve().parent.parent / "data" / "resumes.json") as f:
        resumes = json.load(f)
    for r in resumes:
        if r["id"] == resume_id:
            return r.get("parsed_schema", {}).get("meta", {}).get("raw_text", "")
    return ""


async def test_embedding_api():
    print("=" * 60)
    print("【Step 1】验证 Embedding API 是否正常")
    print("=" * 60)
    client = EmbeddingClient.from_config()
    print(f"  base_url: {client.base_url}")
    print(f"  model: {client.model}")

    texts = ["RAG和Embedding经验", "Java后端开发", "AI产品经理", "设计规范与组件化"]
    result = await client.embed(texts)

    print(f"  返回维度: {[len(r) for r in result]}")
    for i, (text, vec) in enumerate(zip(texts, result)):
        stats = f"min={min(vec):.4f} max={max(vec):.4f} mean={sum(vec)/len(vec):.6f}"
        # 判断是否为 mock 向量（mock 是基于 hash 的伪随机，真实 embedding 均值接近 0）
        is_likely_mock = abs(sum(vec)/len(vec)) > 0.05 and (max(vec) - min(vec)) > 1.5
        status = "❌ MOCK?" if is_likely_mock else "✅ 真实向量"
        print(f"  [{i}] '{text[:20]}' → {stats} {status}")

    # 验证语义相似度：RAG 和 AI产品 应该比 RAG 和设计规范 更相似
    import numpy as np
    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    sim_rag_ai = cosine(result[0], result[2])
    sim_rag_design = cosine(result[0], result[3])
    print(f"\n  语义相似度验证:")
    print(f"    'RAG经验' ↔ 'AI产品经理': {sim_rag_ai:.4f}")
    print(f"    'RAG经验' ↔ '设计规范':   {sim_rag_design:.4f}")
    if sim_rag_ai > sim_rag_design:
        print(f"    ✅ RAG 与 AI产品 更相似（embedding 语义理解正常）")
    else:
        print(f"    ❌ RAG 与设计规范 更相似（embedding 可能仍异常）")
    return sim_rag_ai > sim_rag_design


async def test_retrieval():
    print(f"\n{'=' * 60}")
    print("【Step 2】验证检索质量（混合召回）")
    print(f"{'=' * 60}")
    print(f"  查询: '{QUERY}'")

    kb_result = await _kb_retrieve_stub(query=QUERY, top_k=15)
    chunks = kb_result.data.get("chunks", []) if isinstance(kb_result.data, dict) else []
    print(f"  召回 chunks: {len(chunks)}")

    # 按 jd_id 聚合看唯一 JD
    jd_ids = set()
    companies = []
    for c in chunks:
        meta = c.get("metadata", {}) or {}
        jd_id = meta.get("jd_id", c.get("chunk_id", ""))
        jd_ids.add(jd_id)
        companies.append((meta.get("company", "?"), meta.get("position", "?")))

    print(f"  覆盖唯一 JD: {len(jd_ids)}")
    print(f"\n  召回岗位列表:")
    for i, (company, position) in enumerate(companies[:10], 1):
        # 判断语义相关性
        is_relevant = any(kw in position.lower() for kw in ["ai", "产品", "大模型", "算法", "推荐", "搜索"])
        mark = "✅" if is_relevant else "⚠️"
        print(f"    {mark} {i}. {company} · {position}")

    return chunks


async def test_global_rank(chunks: list):
    print(f"\n{'=' * 60}")
    print("【Step 3】验证 global_rank 输出质量")
    print(f"{'=' * 60}")

    resume_text = load_resume(RESUME_ID)
    tool = GlobalRankTool()

    result = await tool.execute({
        "candidate_jds": chunks,
        "resume_text": resume_text,
        "top_k": 5,
    })

    if not result.success:
        print(f"  ❌ global_rank 失败: {result.error}")
        return

    data = result.data
    is_template = data.get("_template_mode", False)
    rankings = data.get("rankings", [])
    meta = data.get("_coarse_filter_meta", {})

    print(f"  模式: {'模板输出' if is_template else 'LLM精排'}")
    print(f"  粗筛: {meta.get('input_jds')} → {meta.get('output_jds')} JD")
    print(f"\n  Top 5 推荐:")
    for r in rankings:
        print(f"    #{r['rank']} {r['company']} · {r['position']}")
        print(f"       score={r.get('match_score')}, priority={r.get('apply_priority')}")
        print(f"       reason: {r.get('recommend_reason', 'N/A')[:60]}")


async def main():
    ok = await test_embedding_api()
    chunks = await test_retrieval()
    await test_global_rank(chunks)

    print(f"\n{'=' * 60}")
    print("【结论】")
    print(f"{'=' * 60}")
    if ok:
        print("✅ Embedding API 已恢复正常（返回真实语义向量）")
        print("✅ 检索结果应基于语义相似度，而非随机召回")
    else:
        print("❌ Embedding API 仍可能返回 mock 向量，需进一步检查配置")


if __name__ == "__main__":
    asyncio.run(main())
