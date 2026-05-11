import json
import os

with open('eval/results/run1/_report_judge.json', 'r', encoding='utf-8') as f:
    judge = json.load(f)

judge_map = {j['case_id']: j for j in judge.get('judge_breakdown', [])}

folder = 'eval/results/run1'
mapping = {
    'position_explore': 'explore',
    'match_assess': 'assess',
    'attribute_verify': 'verify',
    'interview_prepare': 'prepare',
    'general_chat': 'chat',
    'clarification': 'clarification'
}

print('=== 意图不匹配但 Judge 通过 (11个) ===')
for fname in sorted(os.listdir(folder)):
    if not fname.startswith('eval_') or not fname.endswith('.json'): continue
    with open(os.path.join(folder, fname), 'r', encoding='utf-8') as f:
        c = json.load(f)
    case_id = c.get('case_id', '')
    gold = set(c.get('gold_intents', []))
    pred = c.get('pred_intent', '')
    pred_mapped = mapping.get(pred, pred)
    intent_match = pred_mapped in gold if pred_mapped else False
    
    j = judge_map.get(case_id)
    if j and j.get('resolved') and not intent_match:
        reason = j.get('reason', '')
        print(f"  {case_id}: gold={gold} pred={pred_mapped} [{c.get('scenario')}]")
        print(f"    reason: {reason[:100]}...")

print()
print('=== 意图匹配但 Judge 失败 (3个) ===')
for fname in sorted(os.listdir(folder)):
    if not fname.startswith('eval_') or not fname.endswith('.json'): continue
    with open(os.path.join(folder, fname), 'r', encoding='utf-8') as f:
        c = json.load(f)
    case_id = c.get('case_id', '')
    gold = set(c.get('gold_intents', []))
    pred = c.get('pred_intent', '')
    pred_mapped = mapping.get(pred, pred)
    intent_match = pred_mapped in gold if pred_mapped else False
    
    j = judge_map.get(case_id)
    if j and not j.get('resolved') and intent_match:
        reason = j.get('reason', '')
        print(f"  {case_id}: gold={gold} pred={pred_mapped} [{c.get('scenario')}]")
        print(f"    reason: {reason}")

print()
print('=== 意图不匹配且 Judge 失败 (7个) ===')
for fname in sorted(os.listdir(folder)):
    if not fname.startswith('eval_') or not fname.endswith('.json'): continue
    with open(os.path.join(folder, fname), 'r', encoding='utf-8') as f:
        c = json.load(f)
    case_id = c.get('case_id', '')
    gold = set(c.get('gold_intents', []))
    pred = c.get('pred_intent', '')
    pred_mapped = mapping.get(pred, pred)
    intent_match = pred_mapped in gold if pred_mapped else False
    
    j = judge_map.get(case_id)
    if j and not j.get('resolved') and not intent_match:
        reason = j.get('reason', '')
        print(f"  {case_id}: gold={gold} pred={pred_mapped} [{c.get('scenario')}]")
        print(f"    reason: {reason}")
