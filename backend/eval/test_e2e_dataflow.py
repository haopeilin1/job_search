#!/usr/bin/env python3
"""
端到端数据流验证测试（HTTP 方式）。
验证全链路的数据流动、记忆继承、埋点完整性。
"""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

BASE_URL = "http://127.0.0.1:8002"
RESUME_ID = "cc50dcfb-aeba-41a3-989a-9e485aa2fc3f"  # 陈雨桐


async def test_e2e():
    async with httpx.AsyncClient(timeout=180.0) as client:
        # 1. 激活简历
        print("=== Step 1: 激活简历 ===")
        resp = await client.put(f"{BASE_URL}/api/v1/resumes/{RESUME_ID}/activate")
        print(f"  激活简历: {resp.status_code}")
        
        session_id = "test_e2e_session_001"
        
        # 2. 第一轮：岗位探索
        print("\n=== Step 2: 第一轮 - 岗位探索 ===")
        payload1 = {
            "session_id": session_id,
            "message": "帮我看看有什么适合我的AI产品实习岗",
            "eval_context": {
                "reset_session": True,
                "gold_intents": ["explore"],
                "scenario": "E2E测试-岗位探索",
            }
        }
        start = time.time()
        resp1 = await client.post(f"{BASE_URL}/api/v1/chat", json=payload1)
        latency1 = time.time() - start
        print(f"  HTTP状态: {resp1.status_code}, 延迟: {latency1:.2f}s")
        
        if resp1.status_code != 200:
            print(f"  ERROR: {resp1.text[:500]}")
            return
        
        data1 = resp1.json()
        debug1 = data1.get("debug_info", {})
        
        # 检查意图识别
        intent_debug = debug1.get("intent", {})
        demands = intent_debug.get("demands", [])
        print(f"  识别意图: {[d.get('intent') for d in demands]}")
        print(f"  意图来源: {[d.get('source', 'unknown') for d in demands]}")
        print(f"  是否需要澄清: {intent_debug.get('needs_clarification')}")
        
        # 检查任务图
        tg_debug = debug1.get("task_graph", {})
        tasks = tg_debug.get("tasks", {})
        print(f"  任务数量: {len(tasks)}")
        for tid, t in tasks.items():
            print(f"    - {tid}: {t.get('tool_name')} ({t.get('status')})")
        print(f"  任务图状态: {tg_debug.get('global_status')}")
        print(f"  replan_reason: {tg_debug.get('replan_reason', 'N/A')}")
        print(f"  replan_count: {tg_debug.get('replan_count', 'N/A')}")
        
        # 检查回复
        reply1 = data1.get("reply", {})
        content1 = reply1.get("content") or reply1.get("text", "")
        print(f"  回复长度: {len(content1)} 字符")
        print(f"  回复预览: {content1[:100]}...")
        
        # 检查 session 历史
        session_history = debug1.get("session_history", [])
        print(f"  session_history 轮数: {len(session_history)}")
        
        # 3. 第二轮：上下文引用（verify）
        print("\n=== Step 3: 第二轮 - 上下文引用 ===")
        payload2 = {
            "session_id": session_id,
            "message": "上面那个岗工资多少",
            "eval_context": {
                "gold_intents": ["verify"],
                "scenario": "E2E测试-上下文引用",
            }
        }
        start = time.time()
        resp2 = await client.post(f"{BASE_URL}/api/v1/chat", json=payload2)
        latency2 = time.time() - start
        print(f"  HTTP状态: {resp2.status_code}, 延迟: {latency2:.2f}s")
        
        if resp2.status_code != 200:
            print(f"  ERROR: {resp2.text[:500]}")
            return
        
        data2 = resp2.json()
        debug2 = data2.get("debug_info", {})
        
        intent_debug2 = debug2.get("intent", {})
        demands2 = intent_debug2.get("demands", [])
        print(f"  识别意图: {[d.get('intent') for d in demands2]}")
        print(f"  是否需要澄清: {intent_debug2.get('needs_clarification')}")
        
        # 检查多轮继承
        session_history2 = debug2.get("session_history", [])
        print(f"  session_history 轮数: {len(session_history2)}")
        if len(session_history2) >= 2:
            print(f"  第一轮用户消息: {session_history2[0].get('user_message', '')}")
            print(f"  第二轮用户消息: {session_history2[1].get('user_message', '')}")
            print("  PASS: 多轮对话历史正确继承")
        else:
            print("  WARNING: session_history 轮数不足，可能未正确继承")
        
        # 检查 evidence_cache
        evidence_cache = debug2.get("evidence_cache", [])
        print(f"  evidence_cache 大小: {len(evidence_cache)}")
        if evidence_cache:
            print("  PASS: evidence_cache 在多轮中继承")
        
        # 检查 global_slots
        llm_agent = data2.get("llm_agent", {})
        global_slots = llm_agent.get("global_slots", {})
        print(f"  global_slots: {global_slots}")
        
        # 4. 总结
        print("\n=== 端到端测试总结 ===")
        print(f"  第一轮延迟: {latency1:.2f}s")
        print(f"  第二轮延迟: {latency2:.2f}s")
        print(f"  意图识别可追溯: {'source' in (demands[0] if demands else {})}")
        print(f"  任务图有 replan_count: {'replan_count' in tg_debug}")
        print(f"  多轮历史继承: {len(session_history2) >= 2}")
        print(f"  evidence_cache 继承: {len(evidence_cache) > 0}")
        
        # 保存详细结果供分析
        result_file = Path(__file__).resolve().parent / "results" / "e2e_test_result.json"
        result_file.parent.mkdir(exist_ok=True)
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump({
                "round1": {
                    "latency": latency1,
                    "intent": intent_debug,
                    "task_graph": tg_debug,
                    "reply_preview": content1[:200],
                },
                "round2": {
                    "latency": latency2,
                    "intent": intent_debug2,
                    "session_history_length": len(session_history2),
                    "evidence_cache_size": len(evidence_cache),
                    "global_slots": global_slots,
                }
            }, f, ensure_ascii=False, indent=2)
        print(f"\n  详细结果已保存: {result_file}")


if __name__ == "__main__":
    asyncio.run(test_e2e())
