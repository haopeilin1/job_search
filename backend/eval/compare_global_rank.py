"""
global_rank 对比实验：完整版（粗筛+LLM精排）vs 简化版（仅粗筛）

实验设计：
1. 对3个 explore case，先调用 kb_retrieve 获取候选 chunks
2. 用 GlobalRankTool 完整流程跑一遍（聚合→粗筛→LLM精排）
3. 用简化流程跑一遍（聚合→粗筛→按粗筛分数排序返回，不做LLM）
4. 对比：排序差异、推荐理由质量、耗时、Token消耗
"""
import asyncio
import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.tools import GlobalRankTool, _kb_retrieve_stub
from app.core.llm_client import LLMClient

# 三个测试 case
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


class NoLLMGlobalRankTool(GlobalRankTool):
    """覆盖 execute，跳过 LLM 精排，直接返回粗筛排序结果"""

    async def execute(self, params: dict):
        from app.core.tool_registry import ToolResult as NewToolResult
        from app.core.config import settings

        candidate_chunks = params.get("candidate_jds", [])
        resume_text = params.get("resume_text", "")
        top_k = params.get("top_k", 5)

        if not candidate_chunks:
            return NewToolResult(success=False, error="candidate_jds 为空")

        # 1. 聚合
        aggregated_jds = self._aggregate_chunks_by_jd(candidate_chunks)
        if not aggregated_jds:
            return NewToolResult(success=False, error="chunk 聚合后无有效 JD")

        # 2. 粗筛
        resume_summary = self._extract_resume_summary(resume_text)
        coarse_top_k = max(top_k * settings.COARSE_FILTER_MULTIPLIER, settings.COARSE_FILTER_MIN_POOL)
        filtered_jds = self._coarse_filter(resume_summary, aggregated_jds, top_k=coarse_top_k)

        if not filtered_jds:
            filtered_jds = aggregated_jds[:coarse_top_k]

        # 3. 简化版：不做 LLM，按粗筛内的 hybrid_score 排序返回
        # 注意：_coarse_filter 已经按规则分数排序，但规则分数不等于 hybrid_score
        # 这里我们保留粗筛的排序，但给出一个简化版的 ranking 输出
        rankings = []
        for i, jd in enumerate(filtered_jds[:top_k]):
            summary = self._build_jd_summary(jd, max_chars=400)
            rankings.append({
                "rank": i + 1,
                "jd_id": jd.get("jd_id", ""),
                "company": jd.get("company", "未知"),
                "position": jd.get("position", "未知"),
                "match_score": round(min(60 + jd.get("avg_hybrid_score", 0) * 30, 95)),
                "recommend_reason": f"检索相关度: {jd.get('avg_hybrid_score', 0):.2f}",
                "key_match": [],
                "key_gap": [],
                "apply_priority": "中",
                "_coarse_only": True,
                "_jd_summary": summary,
            })

        data = {
            "rankings": rankings,
            "strategy_advice": "（仅粗筛，无LLM精排）",
            "_coarse_filter_meta": {
                "input_jds": len(aggregated_jds),
                "output_jds": len(filtered_jds),
                "filter_ratio": round(1 - len(filtered_jds) / len(aggregated_jds), 2) if aggregated_jds else 0,
            },
        }
        return NewToolResult(success=True, data=data)


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
    print(f"  kb_retrieve: {kb_time:.2f}s | chunks={len(kb_result.data) if kb_result.success else 0}")

    if not kb_result.success or not kb_result.data:
        print("  ❌ kb_retrieve 失败或无结果")
        return None

    chunks = kb_result.data.get("chunks", []) if isinstance(kb_result.data, dict) else kb_result.data

    # Step 2: 完整版（粗筛 + LLM 精排）
    print("\n[Step 2-A] 完整版（粗筛 + LLM 精排）...")
    full_tool = GlobalRankTool()
    t0 = time.time()
    full_result = await full_tool.execute({
        "candidate_jds": chunks,
        "resume_text": resume_text,
        "top_k": 5,
    })
    full_time = time.time() - t0
    print(f"  耗时: {full_time:.2f}s | success={full_result.success}")

    # Step 3: 简化版（仅粗筛）
    print("\n[Step 2-B] 简化版（仅粗筛，无 LLM）...")
    coarse_tool = NoLLMGlobalRankTool()
    t0 = time.time()
    coarse_result = await coarse_tool.execute({
        "candidate_jds": chunks,
        "resume_text": resume_text,
        "top_k": 5,
    })
    coarse_time = time.time() - t0
    print(f"  耗时: {coarse_time:.2f}s | success={coarse_result.success}")

    # 对比分析
    print("\n[Step 3] 对比分析")
    print("-" * 80)

    full_data = full_result.data if full_result.success else {}
    coarse_data = coarse_result.data if coarse_result.success else {}

    full_rankings = full_data.get("rankings", [])
    coarse_rankings = coarse_data.get("rankings", [])

    print(f"粗筛元数据: {json.dumps(full_data.get('_coarse_filter_meta', coarse_data.get('_coarse_filter_meta')), ensure_ascii=False)}")
    print()

    # 排序对比
    print("【排序对比】")
    full_list = [(r.get("company","?"), r.get("position","?")) for r in full_rankings]
    coarse_list = [(r.get("company","?"), r.get("position","?")) for r in coarse_rankings]

    print(f"  完整版 top5: {full_list}")
    print(f"  简化版 top5: {coarse_list}")

    # 计算 Kendall Tau 近似（逆序对数）
    # 构建公司+岗位的字符串映射（去除空格避免格式差异）
    def _key(r):
        c = (r.get('company','?') or '').replace(' ','').replace('\u3000','')
        p = (r.get('position','?') or '').replace(' ','').replace('\u3000','')
        return f"{c}|{p}"

    full_keys = [_key(r) for r in full_rankings]
    coarse_keys = [_key(r) for r in coarse_rankings]

    # 计算有多少位置不同
    common = set(full_keys) & set(coarse_keys)
    print(f"  共同岗位数: {len(common)} / {max(len(full_keys), len(coarse_keys))}")

    # 位置差异
    pos_diff = 0
    for k in common:
        p1 = full_keys.index(k)
        p2 = coarse_keys.index(k)
        pos_diff += abs(p1 - p2)
    if common:
        print(f"  平均位置偏差: {pos_diff / len(common):.1f}")

    print()
    print("【推荐理由对比（仅完整版有）】")
    for r in full_rankings[:3]:
        print(f"  {r.get('company')} · {r.get('position')}: {r.get('recommend_reason', 'N/A')[:80]}")
        print(f"    match_score={r.get('match_score')}, priority={r.get('apply_priority')}")
        print(f"    key_match={r.get('key_match', [])}")
        print(f"    key_gap={r.get('key_gap', [])}")

    print()
    print("【策略建议对比】")
    print(f"  完整版: {full_data.get('strategy_advice', 'N/A')[:120]}")
    print(f"  简化版: {coarse_data.get('strategy_advice', 'N/A')[:120]}")

    print()
    print(f"【耗时对比】完整版={full_time:.2f}s, 简化版={coarse_time:.2f}s, 差异={full_time-coarse_time:.2f}s")

    return {
        "case": case["name"],
        "full_time": full_time,
        "coarse_time": coarse_time,
        "full_rankings": full_rankings,
        "coarse_rankings": coarse_rankings,
        "coarse_meta": full_data.get("_coarse_filter_meta", {}),
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
        print(f"\n{r['case']}:")
        print(f"  粗筛: input={r['coarse_meta'].get('input_jds')} -> output={r['coarse_meta'].get('output_jds')} "
              f"(过滤率 {r['coarse_meta'].get('filter_ratio',0)*100:.0f}%)")
        print(f"  耗时: 完整版={r['full_time']:.2f}s, 简化版={r['coarse_time']:.2f}s, 节省={r['full_time']-r['coarse_time']:.2f}s")
        full_list = [(x['company'], x['position']) for x in r['full_rankings']]
        coarse_list = [(x['company'], x['position']) for x in r['coarse_rankings']]
        print(f"  完整版排序: {full_list}")
        print(f"  简化版排序: {coarse_list}")


if __name__ == "__main__":
    asyncio.run(main())
