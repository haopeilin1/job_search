"""轻量意图识别准确率测试 - 只测 intent router，不跑完整链路"""
import asyncio
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.llm_intent import LLMIntentRouter, LLMIntentType
from app.core.memory import SessionMemory
from app.core.query_rewrite import QueryRewriteResult


def load_dataset(path: str) -> list:
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def load_resumes() -> dict:
    resumes_file = Path(__file__).parent / "test_resumes.json"
    with open(resumes_file, "r", encoding="utf-8") as f:
        return json.load(f)


def get_resume_text(resume_id: str, resumes: dict) -> str:
    for r in resumes.get("resumes", []):
        if r.get("id") == resume_id:
            return r.get("text", "")
    return ""


def check_intent_hit(predicted: list, gold: list) -> tuple:
    """
    严格匹配：predicted 必须包含 gold 中所有意图才算命中。
    返回 (hit, predicted_set, missing, extra)
    """
    pred_set = set(predicted)
    gold_set = set(gold)
    missing = gold_set - pred_set
    extra = pred_set - gold_set
    hit = len(missing) == 0
    return hit, pred_set, missing, extra


async def test_single_case(case: dict, router: LLMIntentRouter, resumes: dict) -> dict:
    sid = case["session_id"]
    message = case["message"]
    eval_ctx = case.get("eval_context", {})
    gold_intents = eval_ctx.get("gold_intents", [])
    resume_id = case.get("resume_id")
    resume_text = get_resume_text(resume_id, resumes) if resume_id else ""

    session = SessionMemory(session_id=sid)
    rewrite = QueryRewriteResult(
        rewritten_query=message,
        follow_up_type="none",
        original_message=message,
    )

    result = await router.route_multi(
        rewrite_result=rewrite,
        session=session,
        attachments=[],
        raw_message=message,
    )

    predicted = [c.intent_type.value for c in result.candidates]
    primary = result.primary_intent.value if result.primary_intent else None
    hit, pred_set, missing, extra = check_intent_hit(predicted, gold_intents)

    return {
        "case_id": sid,
        "message": message,
        "gold_intents": gold_intents,
        "predicted_intents": predicted,
        "primary_intent": primary,
        "needs_clarification": result.needs_clarification,
        "hit": hit,
        "missing": list(missing),
        "extra": list(extra),
    }


async def main():
    dataset_path = Path(__file__).parent.parent / "eval_runs" / "v1_20250529" / "single_turn" / "dataset.jsonl"
    if not dataset_path.exists():
        print(f"数据集不存在: {dataset_path}")
        sys.exit(1)

    cases = load_dataset(str(dataset_path))
    resumes = load_resumes()
    router = LLMIntentRouter()

    print(f"加载 {len(cases)} 条测试用例，开始意图识别准确率测试...")
    print("(只测 intent router，不跑 planner/executor/judge，成本较低)")
    print()

    results = []
    stats = {
        "total": len(cases),
        "hit": 0,
        "miss": 0,
        "by_intent": defaultdict(lambda: {"total": 0, "hit": 0}),
    }

    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['session_id']}: {case['message'][:30]}...", end=" ")
        try:
            r = await test_single_case(case, router, resumes)
            results.append(r)

            if r["hit"]:
                stats["hit"] += 1
                print(f"✓ hit (predicted={r['predicted_intents']})")
            else:
                stats["miss"] += 1
                print(f"✗ miss (gold={r['gold_intents']}, predicted={r['predicted_intents']}, missing={r['missing']}, extra={r['extra']})")

            # 按 gold 意图统计
            for gi in r["gold_intents"]:
                stats["by_intent"][gi]["total"] += 1
                if r["hit"]:
                    stats["by_intent"][gi]["hit"] += 1

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "case_id": case["session_id"],
                "error": str(e),
            })

    print()
    print("=" * 60)
    print("意图识别准确率报告")
    print("=" * 60)
    print(f"总用例: {stats['total']}")
    print(f"命中:   {stats['hit']} ({stats['hit']/stats['total']*100:.1f}%)")
    print(f"未命中: {stats['miss']} ({stats['miss']/stats['total']*100:.1f}%)")
    print()
    print("按意图类型统计:")
    for intent, s in sorted(stats["by_intent"].items()):
        print(f"  {intent:15s}: {s['hit']}/{s['total']} ({s['hit']/s['total']*100:.1f}%)")

    # 保存结果
    out_dir = Path(__file__).parent.parent / "eval_runs" / "v1_20250529" / "single_turn"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "intent_accuracy_test.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total": stats["total"],
                "hit": stats["hit"],
                "miss": stats["miss"],
                "accuracy": round(stats["hit"] / stats["total"], 4) if stats["total"] else 0,
                "by_intent": dict(stats["by_intent"]),
            },
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
