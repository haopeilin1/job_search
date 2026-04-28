import json

with open('data/jds.json', 'r', encoding='utf-8') as f:
    jds = json.load(f)

print(f"Total JDs: {len(jds)}")
print()

# 检查 structured_summary 的完整性
missing_fields = {}
for jd in jds:
    ss = jd.get('structured_summary', {})
    for k in ['hard_requirements', 'soft_requirements', 'key_responsibilities', 'team_background', 'growth_space', 'interview_focus']:
        if k not in ss or not ss[k]:
            missing_fields[k] = missing_fields.get(k, 0) + 1

print("=== structured_summary 缺失字段统计 ===")
for k, cnt in sorted(missing_fields.items(), key=lambda x: -x[1]):
    print(f"  {k}: {cnt}/{len(jds)} 缺失")

print()

# 检查 raw_text vs description
short_raw = 0
for jd in jds:
    raw_len = len(jd.get('raw_text', ''))
    if raw_len < 300:
        short_raw += 1
        print(f"  Short raw_text: {jd.get('company')} | {jd.get('position')} | {raw_len} chars")

print(f"\nShort raw_text (<300 chars): {short_raw}/{len(jds)}")

print()

# 检查 sections 字段
print("=== sections 字段 ===")
for jd in jds[:3]:
    sections = jd.get('sections', {})
    print(f"{jd.get('company')} | {jd.get('position')} | sections_keys={list(sections.keys())}")

print()

# 检查 keywords
print("=== keywords 字段 ===")
for jd in jds[:3]:
    kw = jd.get('keywords', [])
    print(f"{jd.get('company')} | {jd.get('position')} | keywords={kw}")
