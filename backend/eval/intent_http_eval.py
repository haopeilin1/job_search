#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多轮对话意图识别 HTTP 评测
调用 /api/v1/chat，只提取意图识别结果进行对比

用法:
    cd backend && python eval/intent_http_eval.py
"""

import asyncio
import json
import sys
from pathlib import Path
from collections import defaultdict

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EVAL_DIR = Path(__file__).resolve().parent
DATASET_FILE = EVAL_DIR / "test_dataset.jsonl"
RESUMES_FILE = EVAL_DIR / "test_resumes.json"
BASE_URL = "http://127.0.0.1:8002"
CHAT_URL = f"{BASE_URL}/api/v1/chat"

INTENT_ALIASES = {
    "position_explore": "explore",
    "match_assess": "assess",
    "interview_prepare": "prepare",
    "general_chat": "chat",
    "attribute_verify": "verify",
    "resume_manage": "manage",
    "explore": "explore",
    "assess": "assess",
    "prepare": "prepare",
    "verify": "verify",
    "manage": "manage",
    "chat": "chat",
    "clarification": "clarification",
}


def normalize_intent(intent: str) -> str:
    return INTENT_ALIASES.get(intent, intent)


def load_dataset() -> list:
    cases = []
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def load_resumes() -> dict:
    with open(RESUMES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_resume_id(case: dict, resumes: dict) -> str:
    rid = case.get("resume_id")
    if not rid:
        mapping = resumes.get("session_resume_map", {})
        rid = mapping.get(case.get("session_id", ""), "")
    return rid


async def activate_resume(client: httpx.AsyncClient, resume_id: str):
    if not resume_id:
        return
    try:
        resp = await client.put(f"{BASE_URL}/api/v1/resumes/{resume_id}/activate", timeout=10.0)
        if resp.status_code != 200:
            raise RuntimeError(f"激活简历失败: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"  [Error] 激活简历异常: {e}")
        raise


async def eval_single_case(
    client: httpx.AsyncClient,
    case: dict,
    session_id: str,
    reset_session: bool,
    resume_id: str,
) -> dict:
    message = case["message"]
    eval_ctx = case.get("eval_context", {})
    gold_intents = sorted(set(eval_ctx.get("gold_intents", [])))

    # 激活简历
    await activate_resume(client, resume_id)

    payload = {
        "session_id": session_id,
        "message": message,
        "eval_context": {"reset_session": True} if reset_session else {},
    }

    try:
        resp = await client.post(
            CHAT_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=300.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {
            "case_id": case["session_id"],
            "message": message,
            "gold_intents": gold_intents,
            "pred_intents": ["ERROR"],
            "match": False,
            "error": str(e),
        }

    # 提取预测意图
    route_meta = data.get("route_meta", {})
    demands = route_meta.get("demands", [])
    is_clarification = data.get("is_clarification", False)

    pred_intents = set()
    for d in demands:
        it = d.get("intent_type") or d.get("intent") or ""
        pred_intents.add(normalize_intent(it))
    if is_clarification:
        pred_intents.add("clarification")

    pred_intents = sorted(pred_intents)
    match = set(pred_intents) == set(gold_intents)

    return {
        "case_id": case["session_id"],
        "group": case.get("session_group"),
        "message": message,
        "gold_intents": gold_intents,
        "pred_intents": pred_intents,
        "match": match,
        "needs_clarification": is_clarification,
        "error": None,
    }


async def main():
    cases = load_dataset()
    resumes = load_resumes()

    # 只取有多轮对话的用例
    multi_turn_cases = [c for c in cases if c.get("session_group")]
    print(f"多轮对话用例: {len(multi_turn_cases)} 条\n")

    # 按 group 分组
    groups = defaultdict(list)
    for c in multi_turn_cases:
        groups[c["session_group"]].append(c)

    results = []
    group_session_map = {}

    async with httpx.AsyncClient(timeout=180.0, trust_env=False) as client:
        for group_id in sorted(groups.keys()):
            group_cases = sorted(groups[group_id], key=lambda x: x["session_id"])
            print(f"Group: {group_id}")

            for idx, case in enumerate(group_cases):
                reset_session = True
                sid = case["session_id"]

                if group_id in group_session_map:
                    reset_session = False
                    sid = group_session_map[group_id]
                    print(f"  -> 复用 session: {sid}")
                else:
                    group_session_map[group_id] = sid

                resume_id = get_resume_id(case, resumes)
                r = await eval_single_case(client, case, sid, reset_session, resume_id)
                results.append(r)

                mark = "OK" if r["match"] else "XX"
                print(f"  {mark} {r['case_id']}: gold={r['gold_intents']} pred={r['pred_intents']}")
                if r["error"]:
                    print(f"     ERROR: {r['error']}")

    # 按 group + case_id 排序输出
    results.sort(key=lambda r: (r.get("group") or "", r["case_id"]))

    total = len(results)
    correct = sum(1 for r in results if r["match"])

    print("\n" + "=" * 110)
    print(f"{'Case ID':<18} {'Group':<10} {'Match':<6} {'Gold Intents':<25} {'Pred Intents':<25} {'Message'}")
    print("=" * 110)
    for r in results:
        gold_str = ", ".join(r["gold_intents"])
        pred_str = ", ".join(r["pred_intents"])
        match_mark = "PASS" if r["match"] else "FAIL"
        msg = r["message"][:30] + "..." if len(r["message"]) > 30 else r["message"]
        print(f"{r['case_id']:<18} {r.get('group',''):<10} {match_mark:<6} {gold_str:<25} {pred_str:<25} {msg}")
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
