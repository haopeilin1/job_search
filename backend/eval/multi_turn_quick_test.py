#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多轮对话 HTTP 快速测试 —— 短超时，只验证关键场景

用法:
    cd backend && PYTHONUNBUFFERED=1 python eval/multi_turn_quick_test.py
"""

import asyncio
import json
import sys
import io
import time
from pathlib import Path
from collections import defaultdict

import httpx

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

BASE_URL = "http://127.0.0.1:8005"
CHAT_URL = f"{BASE_URL}/api/v1/chat"
INTENT_ALIASES = {
    "position_explore": "explore", "match_assess": "assess",
    "interview_prepare": "prepare", "general_chat": "chat",
    "attribute_verify": "verify", "resume_manage": "manage",
    "explore": "explore", "assess": "assess", "prepare": "prepare",
    "verify": "verify", "manage": "manage", "chat": "chat",
    "clarification": "clarification",
}


def normalize_intent(intent: str) -> str:
    return INTENT_ALIASES.get(intent, intent)


def load_dataset() -> list:
    cases = []
    with open(Path(__file__).resolve().parent / "test_dataset.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


async def test_turn(client, case, session_id, reset_session, turn_index):
    message = case["message"]
    eval_ctx = case.get("eval_context", {})
    gold_intents = sorted(set(eval_ctx.get("gold_intents", [])))

    # 激活简历
    resume_id = case.get("resume_id", "")
    if resume_id:
        try:
            await client.put(f"{BASE_URL}/api/v1/resumes/{resume_id}/activate", timeout=8.0)
        except Exception as e:
            print(f"  [WARN] 激活简历失败: {repr(e)}", flush=True)

    payload = {
        "session_id": session_id,
        "message": message,
        "eval_context": {"reset_session": True} if reset_session else {},
    }

    t0 = time.time()
    try:
        resp = await client.post(CHAT_URL, json=payload, headers={"Content-Type": "application/json"}, timeout=180.0)
        resp.raise_for_status()
        data = resp.json()
        latency_ms = (time.time() - t0) * 1000
    except asyncio.TimeoutError:
        print(f"  [TIMEOUT] Chat 请求超时 (180s)", flush=True)
        return {"case_id": case["session_id"], "error": "timeout", "gold_intents": gold_intents}
    except Exception as e:
        print(f"  [ERROR] Chat 请求失败: {e}", flush=True)
        return {"case_id": case["session_id"], "error": str(e), "gold_intents": gold_intents}

    route_meta = data.get("route_meta", {})
    demands = route_meta.get("demands", [])
    is_clarification = data.get("is_clarification", False)

    pred_intents = set()
    pred_slots = {}
    tools_called = []
    for d in demands:
        it = d.get("intent_type") or d.get("intent") or ""
        pred_intents.add(normalize_intent(it))
        if d.get("entities"):
            pred_slots.update(d["entities"])
        if d.get("tools"):
            for t in d["tools"]:
                tools_called.append(t.get("tool", ""))
    if is_clarification:
        pred_intents.add("clarification")

    pred_intents = sorted(pred_intents)
    reply = data.get("reply", "") or ""
    match = set(pred_intents) == set(gold_intents)

    print(f"  {'[PASS]' if match else '[FAIL]'} Intent: gold={gold_intents} pred={pred_intents}", flush=True)
    print(f"       Clarify: {is_clarification} | Tools: {tools_called}", flush=True)
    print(f"       Reply: {reply[:100]}... | Latency: {latency_ms:.0f}ms", flush=True)

    return {
        "case_id": case["session_id"],
        "message": message,
        "gold_intents": gold_intents,
        "pred_intents": pred_intents,
        "match": match,
        "needs_clarification": is_clarification,
        "tools_called": tools_called,
        "reply": reply,
        "latency_ms": latency_ms,
        "pred_slots": pred_slots,
    }


async def main():
    cases = load_dataset()
    multi_turn_cases = [c for c in cases if c.get("session_group")]
    print(f"多轮对话用例: {len(multi_turn_cases)} 条\n", flush=True)

    groups = defaultdict(list)
    for c in multi_turn_cases:
        groups[c["session_group"]].append(c)

    results = []

    async with httpx.AsyncClient(timeout=65.0, trust_env=False) as client:
        for group_id in sorted(groups.keys()):
            group_cases = sorted(groups[group_id], key=lambda x: x["session_id"])
            print(f"\n{'='*70}", flush=True)
            print(f"[Group: {group_id}] 共 {len(group_cases)} 轮", flush=True)
            print(f"{'='*70}", flush=True)

            session_id = None
            for idx, case in enumerate(group_cases):
                reset_session = (idx == 0)
                sid = case["session_id"] if reset_session else session_id
                session_id = sid

                print(f"\n  Turn {idx+1}: {case['session_id']}", flush=True)
                print(f"    Msg: {case['message']}", flush=True)
                if not reset_session:
                    print(f"    -> 复用 session: {sid}", flush=True)

                r = await test_turn(client, case, sid, reset_session, idx)
                results.append(r)

    # 汇总
    total = len(results)
    errors = [r for r in results if r.get("error")]
    correct = [r for r in results if not r.get("error") and r.get("match")]
    print(f"\n{'='*70}", flush=True)
    print("[汇总]", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"总计: {total} 轮 | 成功: {total - len(errors)} 轮 | 错误/超时: {len(errors)} 轮", flush=True)
    print(f"意图命中: {len(correct)}/{total - len(errors)} = {len(correct)/(total-len(errors))*100:.1f}%", flush=True)

    if errors:
        print(f"\n错误/超时的 case:", flush=True)
        for r in errors:
            print(f"  {r['case_id']}: {r['error']}", flush=True)

    # 保存
    output_path = Path(__file__).resolve().parent / f"multi_turn_quick_{int(time.time())}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
