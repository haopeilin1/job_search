import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from run_eval_v3 import TurnResult, compute_metrics

reports = [
    'v3_report_wang.json',
    'v3_report_gen_sup.json',
    'v3_report_eval_chen_06.json',
    'v3_report_eval_chen_07.json',
    'v3_report_eval_chen_14.json',
    'v3_report_eval_li_11.json',
    'v3_report_eval_li_15.json',
    'v3_report_eval_li_16.json',
    'v3_report_chen_p1.json',
    'v3_report_chen_p2.json',
    'v3_report_batch_p3.json',
    'v3_report_li_p1.json',
    'v3_report_li_p2.json',
]

all_cases = []
for r in reports:
    with open(r, 'r', encoding='utf-8') as f:
        data = json.load(f)
    cases = data.get('cases', [])
    all_cases.extend(cases)
    print(f'{r}: {len(cases)} cases')

print(f'\nTotal cases: {len(all_cases)}')

case_map = {}
for c in all_cases:
    cid = c['case_id']
    if cid not in case_map:
        case_map[cid] = c
    else:
        if case_map[cid].get('has_exception') and not c.get('has_exception'):
            case_map[cid] = c

unique_cases = list(case_map.values())
print(f'Unique cases: {len(unique_cases)}')

# Convert nested dicts to dataclass objects
from run_eval_v3 import ComponentResult

def fix_case(c):
    for key in ['rewrite', 'intent', 'planner', 'executor']:
        if c.get(key) and isinstance(c[key], dict):
            c[key] = ComponentResult(**c[key])
    return c

fixed_cases = [fix_case(c.copy()) for c in unique_cases]
results = [TurnResult(**c) for c in fixed_cases]
metrics = compute_metrics(results)

final_report = {
    'metrics': metrics,
    'cases': unique_cases,
    'summary': {
        'total_cases': len(unique_cases),
        'timestamp': datetime.now().isoformat(),
    }
}

with open('v3_report_final.json', 'w', encoding='utf-8') as f:
    json.dump(final_report, f, ensure_ascii=False, indent=2)

print('\nFinal report saved: v3_report_final.json')
print(f'Total unique cases: {len(unique_cases)}')
m = metrics['outcome']
print(f'Task success rate: {m["task_success_rate"]:.1%}')
print(f'Exception rate: {m["exception_rate"]:.1%}')
print(f'Intent strict hit: {metrics["intent"]["strict_hit_rate"]:.1%}')
print(f'Tool primary hit: {metrics["tool"]["primary_hit_rate"]:.1%}')
print(f'Avg latency: {m["avg_latency_ms"]:.0f}ms')
