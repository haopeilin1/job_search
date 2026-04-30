"""
global_rank 修改验证脚本
验证内容：
1. JD 数 ≤ 8 时是否跳过 LLM，使用模板输出
2. 简历信息提取是否改善（结构化数据 fallback）
3. 模板输出质量 vs 原 LLM 输出
"""
import asyncio
import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.tools import GlobalRankTool, _kb_retrieve_stub
from app.core.config import settings

CASES = [
    {
        "name": "eval_chen_01",
        "message": "帮我看看有什么适合我的AI产品实习岗",
        "resume_id": "cc50dcfb-aeba-41a3-989a-9e485aa2fc3f",
        "query": "AI产品实习岗",
    },
    {
        "name": "eval_chen_09",
        "message": "我的RAG和Embedding经验对哪个岗最有用",
        "resume_id": "cc50dcfb-aeba-41a3-989a-9e485aa2fc3f",
        "query": "RAG Embedding 经验 岗位",
    },
    {
        "name": "eval_sup_05",
        "message": "帮我看看有什么Java后端岗",
        "resume_id": "cc50dcfb-aeba-41a3-989a-9e485aa2fc3f",
        "query": "Java后端",
        "position": "Java后端",
    },
]


def load_resume(resume_id: str) -> str:
    with open(Path(__file__).resolve().parent.parent / "data" / "resumes.json") as f:
        resumes = json.load(f)
    for r in resumes:
        if r["id"] == resume_id:
            return r.get("parsed_schema", {}).get("meta", {}).get("raw_text", "")
    return ""


async def run_case(case: dict):
    print(f"\n{'='*80}")
    print(f"CASE: {case['name']} | {case['message']}")
    print(f"{'='*80}")

    resume_text = load_resume(case["resume_id"])
    print(f"Resume length: {len(resume_text)} chars")

    # Step 1: kb_retrieve
    print("\n[Step 1] kb_retrieve ...")
    t0 = time.time()
    kb_result = await _kb_retrieve_stub(
        query=case["query"],
        position=case.get("position"),
        top_k=15,
    )
    kb_time = time.time() - t0
    chunks = kb_result.data.get("chunks", []) if isinstance(kb_result.data, dict) else kb_result.data
    print(f"  kb_retrieve: {kb_time:.2f}s | chunks={len(chunks)}")

    if not kb_result.success or not chunks:
        print("  ❌ kb_retrieve 失败或无结果")
        return None

    # Step 2: 调用修改后的 GlobalRankTool
    print(f"\n[Step 2] GlobalRankTool (threshold={settings.GLOBAL_RANK_LLM_THRESHOLD}) ...")
    tool = GlobalRankTool()

    # 先验证简历提取
    resume_summary = tool._extract_resume_summary(resume_text)
    print(f"  简历提取: skills={resume_summary.get('hard_skills', [])}, "
          f"years={resume_summary.get('years_of_experience')}, "
          f"edu={resume_summary.get('education')}, domain={resume_summary.get('domain')}")

    t0 = time.time()
    result = await tool.execute({
        "candidate_jds": chunks,
        "resume_text": resume_text,
        "top_k": 5,
    })
    total_time = time.time() - t0
    print(f"  总耗时: {total_time:.2f}s | success={result.success}")

    if not result.success:
        print(f"  ❌ 失败: {result.error}")
        return None

    data = result.data
    rankings = data.get("rankings", [])
    is_template = data.get("_template_mode", False)
    meta = data.get("_coarse_filter_meta", {})

    print(f"\n[Step 3] 结果分析")
    print(f"  模式: {'模板输出（跳过LLM）' if is_template else 'LLM精排'}")
    print(f"  粗筛: input={meta.get('input_jds')} -> output={meta.get('output_jds')} "
          f"(过滤率 {meta.get('filter_ratio',0)*100:.0f}%)")
    print(f"  推荐岗位数: {len(rankings)}")

    print(f"\n  排序结果:")
    for r in rankings:
        print(f"    #{r['rank']} {r['company']} · {r['position']}")
        print(f"       score={r.get('match_score')}, priority={r.get('apply_priority')}")
        print(f"       reason: {r.get('recommend_reason', 'N/A')[:80]}")
        print(f"       match: {r.get('key_match', [])}")
        print(f"       gap: {r.get('key_gap', [])}")

    print(f"\n  策略建议: {data.get('strategy_advice', 'N/A')[:120]}")

    return {
        "case": case["name"],
        "time": total_time,
        "is_template": is_template,
        "rankings": rankings,
        "meta": meta,
    }


async def main():
    results = []
    for case in CASES:
        try:
            r = await run_case(case)
            if r:
                results.append(r)
        except Exception as e:
            print(f"❌ {case['name']} 失败: {e}")
            import traceback
            traceback.print_exc()

    # 汇总
    print(f"\n{'='*80}")
    print("【实验汇总】")
    print(f"{'='*80}")
    for r in results:
        mode = "模板" if r["is_template"] else "LLM"
        print(f"\n{r['case']}: {mode}模式 | {r['time']:.2f}s")
        print(f"  粗筛: {r['meta'].get('input_jds')} -> {r['meta'].get('output_jds')}")
        for rank in r['rankings'][:3]:
            print(f"  #{rank['rank']} {rank['company']}·{rank['position']} "
                  f"score={rank['match_score']} priority={rank['apply_priority']}")


if __name__ == "__main__":
    asyncio.run(main())
