#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成人工核对的详细报告"""

import json
from pathlib import Path
from collections import defaultdict

EVAL_DIR = Path(__file__).parent

files = [
    'v3_report_wang.json', 'v3_report_gen_sup.json',
    'v3_report_eval_chen_06.json', 'v3_report_eval_chen_07.json', 'v3_report_eval_chen_14.json',
    'v3_report_eval_li_11.json', 'v3_report_eval_li_15.json', 'v3_report_eval_li_16.json',
    'v3_report_chen_p1.json', 'v3_report_chen_p2.json', 'v3_report_batch_p3.json',
    'v3_report_li_p1.json', 'v3_report_li_p2.json',
]

all_stability = {}
all_cases = {}

for fname in files:
    with open(EVAL_DIR / fname, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for c in data.get('cases', []):
        cid = c['case_id']
        if cid not in all_cases or (all_cases[cid].get('has_exception') and not c.get('has_exception')):
            all_cases[cid] = c
    for s in data.get('stability', []):
        all_stability[s['case_id']] = s

# 按 case_id 排序
case_ids = sorted(all_cases.keys())

def fmt_bool(v):
    return '✅' if v else '❌'

def fmt_json(d, max_len=2000):
    s = json.dumps(d, ensure_ascii=False, indent=2)
    if len(s) > max_len:
        s = s[:max_len] + '\n... (truncated)'
    return s

def shorten_text(text, max_len=500):
    if not text:
        return '(空)'
    if len(text) > max_len:
        return text[:max_len] + '...'
    return text

lines = []
lines.append('# v3 ReAct Agent 评测详细核对报告')
lines.append('')
lines.append(f'**总 case 数**: {len(case_ids)}')
lines.append(f'**生成时间**: 2026-05-12')
lines.append('')
lines.append('---')
lines.append('')

# 逐个 case
for cid in case_ids:
    c = all_cases[cid]
    stab = all_stability.get(cid, {})
    runs = stab.get('runs', [])
    
    lines.append(f'## {cid}')
    lines.append('')
    lines.append(f"**批次**: `{c['batch']}`")
    lines.append(f"**用户消息**: {c['message']}")
    lines.append(f"**场景**: {c['scenario']}")
    lines.append(f"**Gold 意图**: `{c['gold_intents']}`")
    lines.append(f"**Gold 槽位**: `{fmt_json(c['gold_slots'], 300)}`")
    lines.append(f"**预期工具**: `{c['expected_tools']}`")
    lines.append('')
    
    # 稳定性测试表格
    lines.append('### 稳定性测试（3轮）')
    lines.append('')
    lines.append('| 轮次 | 成功 | 异常 | 意图严格命中 | 工具主要命中 | 延迟(ms) | 执行工具 | LLM调用 |')
    lines.append('|------|------|------|-------------|-------------|---------|---------|--------|')
    for r in runs:
        tools_str = ', '.join(r.get('executed_tools', [])) or '(无)'
        lines.append(
            f"| {r['run']} | {fmt_bool(r['success'])} | {fmt_bool(r['has_exception'])} | "
            f"{fmt_bool(r['intent_strict_hit'])} | {fmt_bool(r['tool_primary_hit'])} | "
            f"{r['latency_ms']:.0f} | `{tools_str}` | {r.get('llm_calls', 0)} |"
        )
    lines.append('')
    
    if stab:
        lines.append(
            f"**汇总**: 成功率={stab.get('success_rate',0)*100:.1f}% | "
            f"一致性={stab.get('result_consistency',0)*100:.1f}% | "
            f"延迟CV={stab.get('latency_cv',0):.2f} | "
            f"平均延迟={stab.get('avg_latency_ms',0):.0f}ms"
        )
        lines.append('')
    
    # 最后一次完整运行详情
    lines.append('### 最后一次完整运行详情')
    lines.append('')
    
    # Query Rewrite
    rewrite = c.get('rewrite_result')
    if rewrite:
        lines.append('**Query 改写**:')
        lines.append('```json')
        lines.append(fmt_json(rewrite, 1000))
        lines.append('```')
        lines.append('')
    
    # Intent Result
    intent_res = c.get('intent_result')
    if intent_res:
        lines.append('**意图识别结果**:')
        lines.append('```json')
        lines.append(fmt_json(intent_res, 1500))
        lines.append('```')
        lines.append('')
    
    # Task Graph
    task_graph = c.get('task_graph')
    if task_graph:
        lines.append('**Planner 任务图**:')
        lines.append('```json')
        lines.append(fmt_json(task_graph, 1500))
        lines.append('```')
        lines.append('')
    
    # Executed tools
    lines.append(f"**执行的工具**: `{c.get('executed_tools', [])}`")
    lines.append(f"**失败的工具**: `{c.get('failed_tools', [])}`")
    lines.append(f"**Replan 次数**: {c.get('replan_count', 0)}")
    lines.append('')
    
    # Process Quality
    pq = c.get('process_quality')
    if pq:
        lines.append('**过程质量评估**:')
        lines.append('```json')
        lines.append(fmt_json(pq, 1500))
        lines.append('```')
        lines.append('')
    
    # Final Response
    lines.append('**最终回复**:')
    lines.append('```')
    lines.append(shorten_text(c.get('final_response', ''), 800))
    lines.append('```')
    lines.append('')
    
    # Judge Result
    judge = c.get('judge_result')
    if judge:
        lines.append(f"**Judge 判定**: resolved={judge.get('resolved')} | reason={judge.get('reason')} | source={judge.get('source')}")
        lines.append('')
    
    lines.append('---')
    lines.append('')

# 总结
lines.append('')
lines.append('# 总结')
lines.append('')

# 统计所有 stability
total_runs = 0
success_runs = 0
exception_runs = 0
intent_hit_runs = 0
tool_hit_runs = 0

for cid in case_ids:
    stab = all_stability.get(cid, {})
    for r in stab.get('runs', []):
        total_runs += 1
        if r['success']:
            success_runs += 1
        if r['has_exception']:
            exception_runs += 1
        if r['intent_strict_hit']:
            intent_hit_runs += 1
        if r['tool_primary_hit']:
            tool_hit_runs += 1

lines.append(f'**总运行次数**: {total_runs}（{len(case_ids)} case × 3 轮）')
lines.append(f'**任务成功次数/率**: {success_runs}/{total_runs} ({success_runs/total_runs*100:.1f}%)')
lines.append(f'**异常次数/率**: {exception_runs}/{total_runs} ({exception_runs/total_runs*100:.1f}%)')
lines.append(f'**意图严格命中次数/率**: {intent_hit_runs}/{total_runs} ({intent_hit_runs/total_runs*100:.1f}%)')
lines.append(f'**工具主要命中次数/率**: {tool_hit_runs}/{total_runs} ({tool_hit_runs/total_runs*100:.1f}%)')
lines.append('')

# 按批次统计
batch_stats = defaultdict(lambda: {'total_runs': 0, 'success': 0, 'exception': 0, 'intent_hit': 0, 'tool_hit': 0})
for cid in case_ids:
    c = all_cases[cid]
    stab = all_stability.get(cid, {})
    for r in stab.get('runs', []):
        batch_stats[c['batch']]['total_runs'] += 1
        if r['success']:
            batch_stats[c['batch']]['success'] += 1
        if r['has_exception']:
            batch_stats[c['batch']]['exception'] += 1
        if r['intent_strict_hit']:
            batch_stats[c['batch']]['intent_hit'] += 1
        if r['tool_primary_hit']:
            batch_stats[c['batch']]['tool_hit'] += 1

lines.append('## 按批次统计')
lines.append('')
lines.append('| 批次 | 总轮次 | 成功 | 异常 | 意图严格命中 | 工具主要命中 |')
lines.append('|------|--------|------|------|-------------|-------------|')
for batch, stats in sorted(batch_stats.items()):
    total = stats['total_runs']
    lines.append(
        f"| {batch} | {total} | {stats['success']} ({stats['success']/total*100:.1f}%) | "
        f"{stats['exception']} ({stats['exception']/total*100:.1f}%) | "
        f"{stats['intent_hit']} ({stats['intent_hit']/total*100:.1f}%) | "
        f"{stats['tool_hit']} ({stats['tool_hit']/total*100:.1f}%) |"
    )
lines.append('')

# 写入文件
out_path = EVAL_DIR / 'detailed_check_report.md'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f'报告已生成: {out_path}')
print(f'总 case 数: {len(case_ids)}')
print(f'总轮次: {total_runs}')
