import json

with open('data/jds.json', 'r', encoding='utf-8') as f:
    jds = json.load(f)

print("=== sections 内容采样 ===")
for jd in jds[:5]:
    print(f"\n--- {jd.get('company')} | {jd.get('position')} ---")
    sections = jd.get('sections', {})
    for k, v in sections.items():
        if isinstance(v, list):
            print(f"  {k}: {v}")
        elif isinstance(v, str):
            print(f"  {k}: {v[:200]}")
        else:
            print(f"  {k}: {v}")
