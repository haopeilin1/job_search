"""清理注入的测试简历，恢复用户原始数据"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.core import state as app_state

# 1. 从内存中移除eval简历
removed = []
for rid in list(app_state.resumes_db.keys()):
    if rid.startswith("eval_resume_"):
        del app_state.resumes_db[rid]
        removed.append(rid)

# 2. 如果当前active是eval简历，恢复为第一个非eval简历
if app_state.active_resume_id and app_state.active_resume_id.startswith("eval_resume_"):
    first_real = next((k for k in app_state.resumes_db.keys() if not k.startswith("eval_resume_")), None)
    app_state.active_resume_id = first_real

app_state.save_resumes()

print(f"已清理 {len(removed)} 份测试简历: {removed}")
print(f"当前活跃简历: {app_state.active_resume_id}")
print(f"剩余简历数: {len(app_state.resumes_db)}")
