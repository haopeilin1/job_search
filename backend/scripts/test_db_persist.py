#!/usr/bin/env python3
"""
长期记忆持久化测试
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from app.core.memory import LongTermMemory
from app.core.db import (
    save_long_term_memory,
    load_long_term_memory,
    save_session_meta,
    load_session_meta,
    delete_long_term_memory,
    delete_session_meta,
)


def test_long_term_memory():
    print("=" * 50)
    print("Test: LongTermMemory CRUD")
    print("=" * 50)

    # 1. 创建并保存
    lt = LongTermMemory(
        user_id="user_001",
        entities={"技能": ["Python", "RAG"], "公司": ["百度", "阿里"]},
        preferences={"行业": "AI", "城市": "北京"},
        resume_fingerprint="fp_abc123",
    )
    ok = save_long_term_memory(lt)
    print(f"[Save] user_001 -> {'OK' if ok else 'FAIL'}")

    # 2. 加载
    lt_loaded = load_long_term_memory("user_001")
    if lt_loaded:
        print(f"[Load] user_id={lt_loaded.user_id}")
        print(f"[Load] entities={lt_loaded.entities}")
        print(f"[Load] preferences={lt_loaded.preferences}")
        print(f"[Load] resume_fp={lt_loaded.resume_fingerprint}")
        assert lt_loaded.entities["技能"] == ["Python", "RAG"]
        assert lt_loaded.preferences["城市"] == "北京"
        print("[Assert] All passed")
    else:
        print("[Load] FAIL")

    # 3. 更新
    lt.preferences["薪资期望"] = "30k-50k"
    save_long_term_memory(lt)
    lt2 = load_long_term_memory("user_001")
    print(f"[Update] preferences={lt2.preferences}")
    assert lt2.preferences["薪资期望"] == "30k-50k"
    print("[Assert] Update passed")

    # 4. 删除
    delete_long_term_memory("user_001")
    lt3 = load_long_term_memory("user_001")
    print(f"[Delete] after delete -> {'None OK' if lt3 is None else 'FAIL'}")


def test_session_meta():
    print("\n" + "=" * 50)
    print("Test: SessionMeta CRUD")
    print("=" * 50)

    ok = save_session_meta(
        session_id="s_001",
        user_id="user_001",
        current_topic="baidu_jobs",
        evidence_cache_query="百度有什么要求",
    )
    print(f"[Save] s_001 -> {'OK' if ok else 'FAIL'}")

    meta = load_session_meta("s_001")
    if meta:
        print(f"[Load] session_id={meta['session_id']}")
        print(f"[Load] user_id={meta['user_id']}")
        print(f"[Load] topic={meta['current_topic']}")
        print(f"[Load] query={meta['evidence_cache_query']}")
    else:
        print("[Load] FAIL")

    delete_session_meta("s_001")
    meta2 = load_session_meta("s_001")
    print(f"[Delete] after delete -> {'None OK' if meta2 is None else 'FAIL'}")


if __name__ == "__main__":
    test_long_term_memory()
    test_session_meta()
    print("\n" + "=" * 50)
    print("All DB persist tests completed")
    print("=" * 50)
