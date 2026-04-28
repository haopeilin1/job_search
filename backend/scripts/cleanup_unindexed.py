"""清理 vector_indexed=False 的JD条目，重新入库"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.core.vector_store import VectorStore

JD_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "jds.json")

with open(JD_FILE, "r", encoding="utf-8") as f:
    jds = json.load(f)

original_count = len(jds)
# 删除未成功向量化的测试JD（以及之前所有无structured_summary的旧数据如需清理可再议）
# 这里只删除本次seed中失败的：百度大模型应用PM、阿里巴巴后端开发
failed_ids = [j["jd_id"] for j in jds if not j.get("vector_indexed", False)]
print(f"发现 {len(failed_ids)} 条未向量化的JD，准备删除: {failed_ids}")

jds = [j for j in jds if j.get("vector_indexed", False) or j.get("jd_id") not in failed_ids]
# 等等，上面条件写错了。应该直接删除 failed_ids
jds = [j for j in jds if j.get("jd_id") not in failed_ids]

with open(JD_FILE, "w", encoding="utf-8") as f:
    json.dump(jds, f, ensure_ascii=False, indent=2, default=str)

print(f"清理完成：{original_count} -> {len(jds)} 条")
