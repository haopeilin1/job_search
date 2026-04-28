import json
from pathlib import Path

# 1. 检查简历存储
resume_file = Path('data/resumes.json')
if resume_file.exists():
    with open(resume_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print('=== resumes.json 结构 ===')
    print(f"Type: {type(data).__name__}")
    if isinstance(data, list):
        print(f"简历数量: {len(data)}")
        for r in data:
            rid = r.get("id", "N/A")
            name = r.get("name", "N/A")
            text_len = len(r.get("text", ""))
            print(f"  - ID: {rid}, Name: {name}, Text长度: {text_len}")
    elif isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
        if "resumes" in data:
            print(f"简历数量: {len(data['resumes'])}")
            for r in data["resumes"]:
                rid = r.get("id", "N/A")
                name = r.get("name", "N/A")
                text_len = len(r.get("text", ""))
                print(f"  - ID: {rid}, Name: {name}, Text长度: {text_len}")
        if "active_resume_id" in data:
            print(f"Active Resume ID: {data['active_resume_id']}")
else:
    print("resumes.json 不存在")

# 2. 检查JD数据
print("\n=== JD 数据 ===")
jd_files = list(Path("data").glob("*.json"))
for jf in jd_files:
    if "resume" not in jf.name:
        print(f"Found: {jf.name}")

# 3. 检查向量库
print("\n=== 向量库 ===")
vs_file = Path("data/vector_store.json")
if vs_file.exists():
    print(f"vector_store.json 大小: {vs_file.stat().st_size} bytes")
else:
    print("vector_store.json 不存在")

# 4. 检查是否有 jd_chunks 或类似文件
print("\n=== 其他数据文件 ===")
for p in Path("data").iterdir():
    if p.is_file():
        print(f"  {p.name}: {p.stat().st_size} bytes")
