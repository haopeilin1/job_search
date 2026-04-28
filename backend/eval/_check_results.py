import json

with open('eval/eval_results_20260427_173923.jsonl', 'r', encoding='utf-8') as f:
    results = [json.loads(line) for line in f if line.strip()]

print(f'总用例: {len(results)}')
print(f'成功(200): {sum(1 for r in results if r["status_code"]==200)}')
print(f'错误: {sum(1 for r in results if r["error"])}')

print('\n失败的用例:')
for r in results:
    if r['status_code'] != 200:
        sid = r['session_id']
        print(f'  {sid}: status={r["status_code"]} error={str(r["error"])[:80]}')

print('\n成功的用例:')
for r in results:
    if r['status_code'] == 200:
        sid = r['session_id']
        resp_type = r.get('response',{}).get('type','?')
        print(f'  {sid}: type={resp_type} latency={r["latency_ms"]}ms')
