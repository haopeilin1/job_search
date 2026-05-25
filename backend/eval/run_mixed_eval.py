"""
混合评测：单轮用例走本地 intent_only，多轮用例走 HTTP（保证 session 状态机正确）。
后端需支持 eval_context.intent_only=true（只执行意图识别，掐断后续流程）。
"""
import asyncio
import json
import sys
from collections import defaultdict

sys.path.insert(0, "d:/git_vscode/job_search/backend")

import httpx
from app.core.llm_client import LLMClient
from app.core.llm_intent import LLMIntentRouter
from app.core.memory import SessionMemory
from app.core.query_rewrite import QueryRewriter

from eval.intent_only_eval import (
    load_dataset,
    load_resumes,
    get_resume_text,
    set_session_resume,
    eval_single_case as eval_single_local,
)

BASE_URL = "http://127.0.0.1:8002"
CHAT_URL = f"{BASE_URL}/api/v1/chat"


async def eval_single_http(client: httpx.AsyncClient, case: dict, session_id: str, reset_session: bool, resume_id: str):
    """HTTP 方式评测单条用例（只到意图识别层）"""
    message = case["message"]
    eval_ctx = case.get("eval_context", {})
    gold_intents = sorted(set(eval_ctx.get("gold_intents", [])))

    # 激活简历
    if resume_id:
        try:
            await client.put(f"{BASE_URL}/api/v1/resumes/{resume_id}/activate", timeout=10.0)
        except Exception as e:
            print(f"  [Warn] 激活简历失败: {e}")

    payload = {
        "session_id": session_id,
        "message": message,
        "eval_context": {
            "intent_only": True,
            "reset_session": reset_session,
            "resume_id": resume_id,
        },
    }

    try:
        resp = await client.post(
            CHAT_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {
            "case_id": case["session_id"],
            "group": case.get("session_group", ""),
            "message": message,
            "gold_intents": gold_intents,
            "pred_intents": ["ERROR"],
            "match": False,
            "error": str(e),
        }

    # 提取预测意图
    route_meta = data.get("route_meta", {})
    demands = route_meta.get("demands", [])
    pred = [d.get("intent_type", "") for d in demands]
    if data.get("is_clarification"):
        pred.append("clarification")
    pred = sorted(set(pred)) if pred else []

    match = set(pred) == set(gold_intents)

    return {
        "case_id": case["session_id"],
        "group": case.get("session_group", ""),
        "message": message,
        "gold_intents": gold_intents,
        "pred_intents": pred,
        "match": match,
        "error": None,
    }


async def main():
    cases = load_dataset()
    resumes = load_resumes()
    print(f"加载 {len(cases)} 条测试用例\n", flush=True)

    groups = defaultdict(list)
    singles = []
    for c in cases:
        g = c.get("session_group")
        if g:
            groups[g].append(c)
        else:
            singles.append(c)

    results = []

    # ── 单轮用例：本地调用（快）──
    rewriter = QueryRewriter()
    llm_for_intent = LLMClient.from_config("chat")
    router = LLMIntentRouter(chat_llm=llm_for_intent)

    print(f"[LOCAL] 单轮用例: {len(singles)} 条")
    for case in singles:
        session = SessionMemory(session_id=case["session_id"])
        resume_text = get_resume_text(case, resumes)
        set_session_resume(session, resume_text)

        r = await eval_single_local(case, session, rewriter, router)
        results.append(r)
        mark = "OK" if r["match"] else "XX"
        print(f"  {mark} {r['case_id']}: gold={r['gold_intents']} pred={r['pred_intents']}")

    # ── 多轮用例：HTTP（保证状态机正确）──
    print(f"\n[HTTP] 多轮用例: {sum(len(v) for v in groups.values())} 条（{len(groups)} 组）")
    async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:
        group_session_map = {}
        for group_id, group_cases in sorted(groups.items()):
            group_cases.sort(key=lambda c: c["session_id"])
            print(f"\nGroup: {group_id}")

            for idx, case in enumerate(group_cases):
                reset_session = True
                sid = case["session_id"]

                if group_id in group_session_map:
                    reset_session = False
                    sid = group_session_map[group_id]
                    print(f"  -> 复用 session: {sid}")
                else:
                    group_session_map[group_id] = sid

                resume_id = case.get("resume_id", "")
                if not resume_id:
                    mapping = resumes.get("session_resume_map", {})
                    resume_id = mapping.get(case.get("session_id", ""), "")

                r = await eval_single_http(client, case, sid, reset_session, resume_id)
                results.append(r)

                mark = "OK" if r["match"] else "XX"
                print(f"  {mark} {r['case_id']}: gold={r['gold_intents']} pred={r['pred_intents']}")
                if r.get("error"):
                    print(f"     ERROR: {r['error']}")

    # ── 统计 ──
    case_order = {c["session_id"]: i for i, c in enumerate(cases)}
    results.sort(key=lambda r: case_order.get(r["case_id"], 999))

    total = len(results)
    correct = sum(1 for r in results if r["match"])

    print("\n" + "=" * 110)
    print(f"{'Case ID':<18} {'Group':<10} {'Match':<6} {'Gold Intents':<30} {'Pred Intents':<30} {'Message'}")
    print("=" * 110)
    for r in results:
        gold_str = ", ".join(r["gold_intents"])
        pred_str = ", ".join(r["pred_intents"])
        match_mark = "PASS" if r["match"] else "FAIL"
        msg = r["message"][:35] + "..." if len(r["message"]) > 35 else r["message"]
        print(f"{r['case_id']:<18} {r.get('group',''):<10} {match_mark:<6} {gold_str:<30} {pred_str:<30} {msg}")
    print("=" * 110)
    print(f"\n总计: {total} 条 | 命中: {correct} 条 | 准确率: {correct / total * 100:.1f}%")

    misses = [r for r in results if not r["match"]]
    if misses:
        print(f"\n未命中详情 ({len(misses)} 条):")
        for r in misses:
            print(f"  {r['case_id']}: gold={r['gold_intents']} pred={r['pred_intents']}")
            print(f"    msg: {r['message']}")


if __name__ == "__main__":
    asyncio.run(main())
