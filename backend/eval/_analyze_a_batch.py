import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

with open(Path(__file__).parent / "v2_eval_report_1777365098.json", "r", encoding="utf-8") as f:
    data = json.load(f)

INTENT_LABEL_MAP = {
    "explore": "position_explore", "assess": "match_assess",
    "verify": "attribute_verify", "prepare": "interview_prepare",
    "chat": "general_chat", "clarification": "clarification",
}

print("=== 意图识别分析 ===")
for c in data["cases"]:
    if not c.get("intent_result"):
        continue
    pred = set(d["intent_type"] for d in c["intent_result"].get("demands", []))
    if c["intent_result"].get("needs_clarification"):
        pred.add("clarification")
    gold = c.get("gold_intents", [])
    gold_mapped = set(INTENT_LABEL_MAP.get(g, g) for g in gold)
    match = "OK" if pred == gold_mapped else "MISMATCH"
    print(f"  {c['case_id']}: {match} pred={pred} gold={gold_mapped}")

print("\n=== 任务失败分析 ===")
for c in data["cases"]:
    if not c["task_success"]:
        print(f"  {c['case_id']}: tools={c.get('executed_tools')} failed={c.get('failed_tools')}")
        tg = c.get("task_graph", {})
        print(f"    global_status: {tg.get('global_status')}")
        for tid, t in tg.get("tasks", {}).items():
            print(f"    {tid}: {t['tool_name']} status={t['status']}")

print("\n=== 澄清触发分析 ===")
for c in data["cases"]:
    if c.get("intent_result") and c["intent_result"].get("needs_clarification"):
        print(f"  {c['case_id']}: {c['message']}")
        print(f"    clarify_q: {c['intent_result'].get('clarification_question', '')[:60]}")
