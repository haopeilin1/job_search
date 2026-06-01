#!/usr/bin/env python3
"""
全链路集成测试 —— 验证 Plan-and-Execute + Reflection + Replan 完整链路

覆盖场景：
  1. 单意图串行依赖（assess: kb_retrieve → match_analyze）
  2. 缓存复用 + 追问（verify: evidence_cache rerank → qa_synthesize）
  3. 时效性查询（触发 external_search + 冲突检测）
  4. 面试准备（interview_gen 自适应）

运行方式：
    cd backend && python eval/test_integrated_chain.py
"""

import asyncio
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.memory import SessionMemory
from app.routers.chat import chat_endpoint
from app.services.handlers import ChatRequest
from app.routers.resumes import activate_resume


# ═══════════════════════════════════════════════════════
# 测试配置
# ═══════════════════════════════════════════════════════

SESSION_ID = "test_integrated_chain_001"
RESUME_ID = "cc50dcfb-aeba-41a3-989a-9e485aa2fc3f"  # AI简历


# ═══════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════

async def send_message(
    session_id: str,
    message: str,
    eval_context: dict = None,
    resume_id: str = None,
) -> dict:
    """发送单条消息，返回 chat_endpoint 结果"""
    if resume_id:
        try:
            await activate_resume(resume_id)
        except Exception as e:
            print(f"  [!] 切换简历失败: {e}")

    request = ChatRequest(
        session_id=session_id,
        message=message,
        eval_context=eval_context or {},
    )
    return await chat_endpoint(request)


def extract_tools(data: dict) -> list:
    """从返回数据中提取执行的工具列表"""
    debug = data.get("debug_info", {})
    tg = debug.get("task_graph", {})
    tasks = tg.get("tasks", {})
    tools = []
    for tid, t in tasks.items():
        if t.get("tool_name") and t.get("status") == "success":
            tools.append(t["tool_name"])
    return sorted(set(tools))


def extract_reflection(data: dict) -> dict:
    """提取反思结果"""
    debug = data.get("debug_info", {})
    return debug.get("reflection_result", {})


def extract_replan(data: dict) -> dict:
    """提取 replan 信息"""
    debug = data.get("debug_info", {})
    tg = debug.get("task_graph", {})
    return {
        "replan_count": tg.get("replan_count", 0),
        "replan_reason": tg.get("replan_reason", ""),
        "global_status": tg.get("global_status", ""),
    }


def print_round_result(round_num: int, message: str, data: dict):
    """打印单轮测试结果"""
    print(f"\n{'='*60}")
    print(f"Round {round_num}: {message}")
    print(f"{'='*60}")

    reply = data.get("reply", {})
    content = reply.get("content") or reply.get("text", "")
    print(f"  回复: {content[:200]}...")

    tools = extract_tools(data)
    print(f"  执行工具: {tools}")

    reflection = extract_reflection(data)
    if reflection:
        print(f"  反思: action={reflection.get('suggested_action')} | "
              f"complete={reflection.get('is_complete')} | "
              f"task={reflection.get('problematic_task')} | "
              f"reason={reflection.get('reason', '')[:60]}")

    replan = extract_replan(data)
    print(f"  replan: count={replan['replan_count']} | reason={replan['replan_reason']} | status={replan['global_status']}")

    # evidence_cache
    debug = data.get("debug_info", {})
    ec = debug.get("evidence_cache", [])
    print(f"  evidence_cache: {len(ec)}条")


# ═══════════════════════════════════════════════════════
# 测试用例
# ═══════════════════════════════════════════════════════

async def test_round_1_assess():
    """
    Round 1: 单意图串行依赖
    输入: 明确的公司+岗位，要求匹配分析
    期望: kb_retrieve → match_analyze
    验证: 意图识别、任务规划、依赖链执行
    """
    message = "帮我分析一下字节跳动的AI产品经理这个岗位，我能不能去？"
    data = await send_message(
        SESSION_ID, message,
        eval_context={
            "gold_intents": ["assess"],
            "expected_tools": ["kb_retrieve", "match_analyze"],
            "scenario": "单意图串行依赖_assess",
        },
        resume_id=RESUME_ID,
    )
    print_round_result(1, message, data)

    tools = extract_tools(data)
    issues = []
    if "kb_retrieve" not in tools:
        issues.append("缺少 kb_retrieve")
    if "match_analyze" not in tools:
        issues.append("缺少 match_analyze")

    reflection = extract_reflection(data)
    if reflection.get("suggested_action") not in ("pass", "note_uncertainty"):
        issues.append(f"反思异常: {reflection.get('suggested_action')}")

    return data, issues


async def test_round_2_cache_reuse():
    """
    Round 2: 缓存复用 + 追问
    输入: 追问薪资（复用 Round 1 的 evidence_cache）
    期望: kb_retrieve(复用/增量) → qa_synthesize
    验证: evidence_cache 复用、rerank 判断、答案完整性
    """
    message = "那这个岗工资大概多少？"
    data = await send_message(
        SESSION_ID, message,
        eval_context={
            "gold_intents": ["verify"],
            "expected_tools": ["kb_retrieve", "qa_synthesize"],
            "scenario": "缓存复用_追问薪资",
        },
    )
    print_round_result(2, message, data)

    tools = extract_tools(data)
    issues = []
    if "qa_synthesize" not in tools and "general_chat" not in tools:
        issues.append("缺少 qa_synthesize/general_chat")

    reflection = extract_reflection(data)
    # 注意：如果缓存里没有薪资信息，反思应该检测到答案不完整
    if not reflection.get("is_complete", True) and reflection.get("suggested_action") == "re_retrieve":
        print(f"  ✅ 反思正确检测到答案不完整: {reflection.get('missing_info')}")

    return data, issues


async def test_round_3_temporal():
    """
    Round 3: 时效性查询（触发 external_search）
    输入: "最近还在招吗"
    期望: kb_retrieve → external_search → qa_synthesize
    验证: 时效性检测、external_search 触发、冲突检测
    """
    message = "最近字节还在招这个AI产品经理吗？"
    data = await send_message(
        SESSION_ID, message,
        eval_context={
            "gold_intents": ["verify"],
            "expected_tools": ["kb_retrieve", "external_search", "qa_synthesize"],
            "scenario": "时效性查询_external_search",
        },
    )
    print_round_result(3, message, data)

    tools = extract_tools(data)
    issues = []
    if "external_search" not in tools:
        issues.append("[!] 未触发 external_search（可能知识库已覆盖或时效性检测未命中）")

    reflection = extract_reflection(data)
    if reflection.get("has_conflict"):
        print(f"  ✅ 反思检测到来源冲突: {reflection.get('missing_info')}")

    return data, issues


async def test_round_4_interview():
    """
    Round 4: 面试准备
    输入: 准备面试
    期望: interview_gen（有 match_analyze 历史结果时针对性出题）
    验证: interview_gen 自适应（有/无 match_result）
    """
    message = "帮我准备一下字节AI产品经理的面试"
    data = await send_message(
        SESSION_ID, message,
        eval_context={
            "gold_intents": ["prepare"],
            "expected_tools": ["interview_gen"],
            "scenario": "面试准备_interview_gen",
        },
    )
    print_round_result(4, message, data)

    tools = extract_tools(data)
    issues = []
    if "interview_gen" not in tools:
        issues.append("缺少 interview_gen")

    return data, issues


async def test_round_5_explore_empty_resume():
    """
    Round 5: 空简历 + 宽泛探索
    输入: 帮我推荐岗位（使用空简历）
    期望: kb_retrieve → global_rank
    验证: 空简历场景、global_rank 粗筛
    """
    # 先重置 session
    message = "帮我推荐几个适合我的岗位"
    data = await send_message(
        f"{SESSION_ID}_empty",
        message,
        eval_context={
            "gold_intents": ["explore"],
            "expected_tools": ["kb_retrieve", "global_rank"],
            "scenario": "空简历_岗位探索",
            "reset_session": True,
        },
        resume_id="empty",  # 空简历ID需要确认
    )
    print_round_result(5, message, data)

    tools = extract_tools(data)
    issues = []
    if "kb_retrieve" not in tools:
        issues.append("缺少 kb_retrieve")
    if "global_rank" not in tools:
        issues.append("缺少 global_rank")

    return data, issues


# ═══════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════

async def main():
    print("=" * 60)
    print("全链路集成测试开始")
    print("=" * 60)
    print(f"Session: {SESSION_ID}")
    print(f"Resume:  {RESUME_ID}")
    print()

    all_issues = []
    start = time.time()

    try:
        # Round 1: assess
        _, issues = await test_round_1_assess()
        all_issues.extend([(1, i) for i in issues])

        # Round 2: cache reuse
        _, issues = await test_round_2_cache_reuse()
        all_issues.extend([(2, i) for i in issues])

        # Round 3: temporal + external_search
        _, issues = await test_round_3_temporal()
        all_issues.extend([(3, i) for i in issues])

        # Round 4: interview
        _, issues = await test_round_4_interview()
        all_issues.extend([(4, i) for i in issues])

    except Exception as e:
        print(f"\n❌ 测试异常中断: {e}")
        traceback.print_exc()
        all_issues.append((0, f"异常中断: {e}"))

    elapsed = time.time() - start

    # 汇总
    print(f"\n{'='*60}")
    print("测试汇总")
    print(f"{'='*60}")
    print(f"总耗时: {elapsed:.1f}s")
    print(f"问题数: {len(all_issues)}")

    if all_issues:
        print("\n发现的问题:")
        for round_num, issue in all_issues:
            print(f"  Round {round_num}: {issue}")
    else:
        print("\n✅ 所有检查通过，未发现明显问题")

    # 逻辑漏洞分析（静态）
    print(f"\n{'='*60}")
    print("逻辑漏洞检查清单")
    print(f"{'='*60}")
    print("[ ] evidence_cache 复用: Round 2 是否真正复用了 Round 1 的缓存？")
    print("[ ] 答案完整性: Round 2 如果缓存里没薪资，反思是否检测到？")
    print("[ ] 时效性触发: Round 3 是否触发了 external_search？")
    print("[ ] 冲突检测: Round 3 如果有 kb+external，反思是否检测到冲突？")
    print("[ ] replan 调用: 如果反思建议 replan_tools，是否真的调用了 Planner.replan？")
    print("[ ] interview_gen: Round 4 是否有 match_result 传入？")
    print("[ ] 空简历: Round 5 global_rank 是否正确处理了空简历？")
    print("[ ] 无限循环: replan 后重新反思是否可能触发二次 replan？")
    print()
    print("注: 以上需要查看详细日志确认。运行带 --verbose 可输出完整 debug_info。")


if __name__ == "__main__":
    asyncio.run(main())
