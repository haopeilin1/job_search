import json

with open('data/resumes.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for r in data:
    schema = r.get('parsed_schema', {})
    name = schema.get('basic_info', {}).get('name', '')
    if '雨桐' in name:
        print('=== 陈雨桐简历分析 ===')
        print(f'姓名: {name}')
        
        edu = schema.get('education', [])
        print(f'教育: {len(edu)}段')
        for e in edu:
            print(f'  - {e.get("school")} | {e.get("degree")} | {e.get("major")}')
        
        projects = schema.get('projects', [])
        print(f'项目: {len(projects)}个')
        for p in projects:
            print(f'  - {p.get("name")} | {p.get("role")}')
        
        skills = schema.get('skills', {})
        tech = skills.get('technical', [])
        biz = skills.get('business', [])
        print(f'技术技能({len(tech)}): {tech}')
        print(f'业务技能({len(biz)}): {biz}')
        
        work = schema.get('work_experience', [])
        print(f'工作经历: {len(work)}段 (在校生，无正式工作经历)')
        
        raw = schema.get('meta', {}).get('raw_text', '')
        print(f'原始文本: {len(raw)}字符')
        break
