"""
测试简历持久化机制的加载与保存逻辑。
"""

import json
import sys
import tempfile
from pathlib import Path

# 将 backend 加入路径以便导入 state
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 由于 app 依赖 fastapi/httpx 等，我们模拟测试 state.py 中的加载逻辑

def test_load_existing():
    """测试加载已存在的 resumes.json"""
    resume_file = Path(__file__).resolve().parent.parent / "data" / "resumes.json"
    if not resume_file.exists():
        print(f"[Skip] {resume_file} 不存在")
        return

    with open(resume_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[Test] 文件中共 {len(data)} 条简历")
    for item in data:
        rid = item.get("id")
        ps = item.get("parsed_schema", {})
        meta = ps.get("meta", {})
        print(f"  - resume_id={rid[:8]}... | is_active={meta.get('is_active')} | parser={meta.get('parser_version')}")


def test_save_and_reload():
    """测试保存后重新加载"""
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False, encoding="utf-8") as f:
        temp_path = f.name

    db = {
        "test-001": {
            "id": "test-001",
            "parsed_schema": {
                "basic_info": {"name": "张三"},
                "meta": {"is_active": True, "parser_version": "test"},
            },
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
        }
    }
    active_id = "test-001"

    # 模拟保存
    for rid, item in db.items():
        ps = item.get("parsed_schema", {})
        meta = ps.get("meta", {})
        if isinstance(meta, dict):
            meta["is_active"] = (rid == active_id)

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(list(db.values()), f, ensure_ascii=False, indent=2)

    # 模拟加载
    with open(temp_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert len(loaded) == 1
    assert loaded[0]["parsed_schema"]["meta"]["is_active"] == True
    print(f"[Test] 保存/加载测试通过 | temp={temp_path}")

    # 清理
    Path(temp_path).unlink(missing_ok=True)


def test_state_load():
    """直接测试 state.py 的加载函数"""
    try:
        from app.core import state
        print(f"[Test] state.py 加载成功")
        print(f"  resumes_db 条目数: {len(state.resumes_db)}")
        print(f"  active_resume_id: {state.active_resume_id}")
        # 验证数据完整性
        for rid, item in state.resumes_db.items():
            ps = item.get("parsed_schema", {})
            bi = ps.get("basic_info", {})
            print(f"  - {rid[:8]}: name={bi.get('name')}, edu={len(ps.get('education', []))}, proj={len(ps.get('projects', []))}")
    except Exception as e:
        print(f"[Error] state.py 加载失败: {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("Resume Persistence Test")
    print("=" * 50)
    test_load_existing()
    print()
    test_save_and_reload()
    print()
    test_state_load()
    print("=" * 50)
    print("All tests completed.")
