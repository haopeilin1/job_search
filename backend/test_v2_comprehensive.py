#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2 链路全面测试脚本
覆盖: Tavily 访问、知识库检索、工具运行、简历-JD 匹配、面试题生成、
     多轮对话 evidence_cache 复用、兜底机制
"""

import asyncio
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))
os.chdir(os.path.dirname(__file__))

from app.core.config import settings
from app.core.llm_client import LLMClient
from app.core.memory import SessionMemory
from app.core.query_rewrite import QueryRewriter
from app.core.intent_recognition import IntentRecognizer
from app.core.planner import TaskPlanner
from app.core.react_executor import ReActExecutor
from app.core.memory import LongTermMemory
from app.core.mcp_search import tavily_search
from app.core.tool_registry import ToolRegistry
from app.core.tools import create_tool_registry
from app.core.vector_store import VectorStore
from app.core.embedding import EmbeddingClient

results = []


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def record(name, status, detail=""):
    symbol = {"PASS": "[OK]", "FAIL": "[XX]", "WARN": "[!!]", "SKIP": "[--]"}.get(
        status, "[??]"
    )
    print(f"  {symbol} {name}")
    if detail:
        print(f"      -> {detail}")
    results.append((name, status, detail))


def summary():
    print(f"\n{'=' * 60}")
    print("  测试汇总")
    print("=" * 60)
    total = len(results)
    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    skipped = sum(1 for _, s, _ in results if s == "SKIP")
    warn = sum(1 for _, s, _ in results if s == "WARN")
    print(f"  总计: {total}  |  通过: {passed}  |  失败: {failed}  |  警告: {warn}  |  跳过: {skipped}")
    if failed > 0:
        print("\n  失败项:")
        for name, _, detail in results:
            if _ == "FAIL":
                print(f"    - {name}: {detail}")
    print("=" * 60)
    return failed == 0


# ------------------------------------------------------------------
# 1. 基础设施测试
# ------------------------------------------------------------------
async def test_infrastructure():
    section("1. 基础设施测试")

    # 1.1 各模型层连通性
    from app.core.state import llm_config_store

    for layer in ["chat", "core", "planner", "memory", "vision"]:
        cfg = getattr(llm_config_store, layer)
        try:
            client = LLMClient(
                base_url=cfg.base_url, api_key=cfg.api_key, model=cfg.model
            )
            resp = await client.generate(
                prompt="你好，请回复'OK'",
                system="你是一个测试助手",
                max_tokens=10,
                timeout=20,
            )
            ok = "OK" in resp.upper() or "ok" in resp.lower()
            record(
                f"模型层 {layer} ({cfg.model})",
                "PASS" if ok else "WARN",
                f"响应: {resp[:50]}...",
            )
        except Exception as e:
            record(f"模型层 {layer} ({cfg.model})", "FAIL", str(e)[:120])

    # 1.2 Embedding
    try:
        from app.core.embedding import EmbeddingClient
        ec = EmbeddingClient.from_config()
        emb_list = await ec.embed(["测试文本"])
        emb = emb_list[0]
        ok = len(emb) > 100
        record(
            f"Embedding ({settings.EMBEDDING_MODEL})",
            "PASS" if ok else "FAIL",
            f"维度: {len(emb)}",
        )
    except Exception as e:
        record("Embedding", "FAIL", str(e)[:120])

    # 1.3 Tavily
    if settings.TAVILY_SEARCH_ENABLED and settings.TAVILY_API_KEY:
        try:
            chunks = await tavily_search("Python 编程", count=2)
            ok = len(chunks) > 0 and "content" in chunks[0]
            record(
                "Tavily 搜索",
                "PASS" if ok else "FAIL",
                f"返回 {len(chunks)} 条结果",
            )
        except Exception as e:
            record("Tavily 搜索", "FAIL", str(e)[:120])
    else:
        record("Tavily 搜索", "SKIP", "未启用或未配置 API Key")

    # 1.4 Reranker
    if settings.RERANKER_ENABLED:
        try:
            from app.core.reranker import rerank

            docs = [
                {"content": "Python is a programming language"},
                {"content": "Java is also a language"},
            ]
            scores = await rerank("Python", docs, top_k=2)
            ok = len(scores) == 2 and scores[0][1] >= scores[1][1]
            record(
                "Reranker",
                "PASS" if ok else "FAIL",
                f"得分: {[s[1] for s in scores]}",
            )
        except Exception as e:
            record("Reranker", "FAIL", str(e)[:120])
    else:
        record("Reranker", "SKIP", "未启用")

    # 1.5 知识库检索 (需要 data/db 有数据)
    try:
        vs = VectorStore()
        vs.embedding_client = EmbeddingClient.from_config()
        chunks = await vs.query("Python", top_k=3)
        ok = len(chunks) > 0
        record(
            "知识库检索",
            "PASS" if ok else "WARN",
            f"返回 {len(chunks)} 条 chunks",
        )
    except Exception as e:
        record("知识库检索", "FAIL", str(e)[:120])


# ------------------------------------------------------------------
# 2. 工具层测试
# ------------------------------------------------------------------
async def test_tools():
    section("2. 工具层测试")

    registry = create_tool_registry()

    # 2.1 kb_retrieve
    try:
        tool = registry.get("kb_retrieve")
        r = await tool.execute({"query": "Python", "top_k": 3})
        chunks = r.data.get("chunks", []) if isinstance(r.data, dict) else []
        ok = r.success and len(chunks) > 0
        record("Tool: kb_retrieve", "PASS" if ok else "FAIL", f"返回 {len(chunks)} 条")
    except Exception as e:
        record("Tool: kb_retrieve", "FAIL", str(e)[:120])

    # 2.2 match_analyze
    try:
        tool = registry.get("match_analyze")
        r = await tool.execute({
            "resume_text": "熟悉 Python, Django, 3年后端开发经验",
            "jd_source": "text",
            "jd_data": {"full_text": "招聘 Python 后端工程师，要求熟悉 Django 和 RESTful API 设计"},
        })
        ok = r.success and bool(r.data)
        record("Tool: match_analyze", "PASS" if ok else "FAIL", str(r.data)[:100])
    except Exception as e:
        record("Tool: match_analyze", "FAIL", str(e)[:120])

    # 2.3 interview_gen
    try:
        tool = registry.get("interview_gen")
        r = await tool.execute({
            "match_result": {
                "missing_experience": "缺乏微服务架构经验",
                "core_skills": "Python, Django, RESTful API, MySQL",
                "covered": ["Python", "Django"],
                "match_score": 65,
            },
        })
        questions = r.data.get("questions", []) if isinstance(r.data, dict) else []
        ok = r.success and len(questions) > 0
        record("Tool: interview_gen", "PASS" if ok else "FAIL", f"返回 {len(questions)} 题")
    except Exception as e:
        record("Tool: interview_gen", "FAIL", str(e)[:120])

    # 2.4 evidence_relevance_check
    try:
        tool = registry.get("evidence_relevance_check")
        r = await tool.execute({
            "query": "Python 开发",
            "evidence_chunks": [
                {"content": "Python is great for backend"},
                {"content": " unrelated Java stuff"},
            ],
        })
        ok = r.success and isinstance(r.data, dict) and "relevant" in r.data
        record("Tool: evidence_relevance", "PASS" if ok else "FAIL", str(r.data)[:100])
    except Exception as e:
        record("Tool: evidence_relevance", "FAIL", str(e)[:120])


# ------------------------------------------------------------------
# 3. 核心链路测试 (v2 ReAct Agent)
# ------------------------------------------------------------------
async def test_core_pipeline():
    section("3. 核心链路测试 (v2 ReAct Agent)")

    session = SessionMemory(session_id=f"test-{int(time.time())}")
    rewriter = QueryRewriter()
    intent_engine = IntentRecognizer()
    planner = TaskPlanner()
    executor = ReActExecutor()

    queries = [
        ("我想找 Python 后端开发工作", "首次查询"),
        ("那 Java 的呢", "指代消解/主题切换"),
        ("薪资范围呢", "追问/expand"),
    ]

    for q, desc in queries:
        print(f"\n  [{desc}] 用户: {q}")
        try:
            # QueryRewrite
            rw = await rewriter.rewrite(raw_query=q, session=session)
            record(f"Rewrite ({desc})", "PASS", f"type={rw.follow_up_type}, query={rw.rewritten_query[:40]}")

            # Intent
            intent = await intent_engine.recognize(
                rewrite_result=rw, session=session
            )
            l1 = intent.demands[0].intent_type if intent.demands else "none"
            record(f"Intent ({desc})", "PASS", f"L1={l1}, demands={len(intent.demands)}")

            # evidence_cache_summary
            cache_summary = ""
            if session.evidence_cache:
                cache_summary = "; ".join(
                    [c.get("content", "")[:80] for c in session.evidence_cache]
                )

            # Planner
            graph = await planner.create_graph(
                rewritten_query=rw.rewritten_query,
                demands=[{"intent_type": d.intent_type, "entities": d.entities, "priority": d.priority} for d in intent.demands],
                resolved_entities=intent.resolved_entities,
                resume_text="",
                search_keywords=rw.search_keywords or "",
                follow_up_type=rw.follow_up_type,
                evidence_cache_summary=cache_summary,
            )
            record(f"Planner ({desc})", "PASS", f"nodes={len(graph.tasks)}")

            # Executor
            graph = await executor.execute(graph, session)
            failed_nodes = [t for t in graph.tasks.values() if t.status == "failed"]
            if failed_nodes:
                record(
                    f"Executor ({desc})",
                    "WARN",
                    f"{len(failed_nodes)} 个节点失败: {[n.name for n in failed_nodes]}",
                )
            else:
                record(f"Executor ({desc})", "PASS", "全部成功")

            # 更新 evidence_cache
            for task in graph.tasks.values():
                if task.tool_name == "kb_retrieve" and task.result and task.status != "failed":
                    chunks = task.result if isinstance(task.result, list) else []
                    for c in chunks:
                        if isinstance(c, dict) and c not in session.evidence_cache:
                            session.evidence_cache.append(c)
                            if len(session.evidence_cache) > settings.EVIDENCE_CACHE_MAX_SIZE:
                                session.evidence_cache.pop(0)

            # 保存到 working_memory
            from app.core.memory import DialogueTurn
            turn = DialogueTurn(
                turn_id=len(session.working_memory.turns),
                user_message=q,
                assistant_reply=f"[test response for {desc}]",
                intent="",
                rewritten_query=rw.rewritten_query,
            )
            session.working_memory.append(turn)

        except Exception as e:
            record(f"Core pipeline ({desc})", "FAIL", str(e)[:200])


# ------------------------------------------------------------------
# 4. 业务场景测试
# ------------------------------------------------------------------
async def test_business_scenarios():
    section("4. 业务场景测试")

    session = SessionMemory(session_id=f"test-biz-{int(time.time())}")
    rewriter = QueryRewriter()
    intent_engine = IntentRecognizer()
    planner = TaskPlanner()
    executor = ReActExecutor()

    scenarios = [
        ("帮我写一份 Java 后端开发的简历", "resume_write"),
        ("这份简历和 JD 匹配度如何", "resume_jd_match"),
        ("给我出 3 道 Python 面试题", "interview_questions"),
        ("分析下我的面试表现", "interview_analysis"),
    ]

    for q, scenario in scenarios:
        print(f"\n  [场景: {scenario}] 用户: {q}")
        try:
            rw = await rewriter.rewrite(raw_query=q, session=session)
            intent = await intent_engine.recognize(
                rewrite_result=rw, session=session
            )
            cache_summary = ""
            if session.evidence_cache:
                cache_summary = "; ".join(
                    [c.get("content", "")[:80] for c in session.evidence_cache]
                )
            graph = await planner.create_graph(
                rewritten_query=rw.rewritten_query,
                demands=[{"intent_type": d.intent_type, "entities": d.entities, "priority": d.priority} for d in intent.demands],
                resolved_entities=intent.resolved_entities,
                resume_text="",
                search_keywords=rw.search_keywords or "",
                follow_up_type=rw.follow_up_type,
                evidence_cache_summary=cache_summary,
            )
            graph = await executor.execute(graph, session)
            failed = [t for t in graph.tasks.values() if t.status == "failed"]
            if failed:
                record(f"场景: {scenario}", "WARN", f"{len(failed)} 失败")
            else:
                record(f"场景: {scenario}", "PASS", f"{len(graph.tasks)} 个节点完成")

            # 更新 evidence_cache
            for task in graph.tasks.values():
                if task.tool_name == "kb_retrieve" and task.result and task.status != "failed":
                    chunks = task.result if isinstance(task.result, list) else []
                    for c in chunks:
                        if isinstance(c, dict) and c not in session.evidence_cache:
                            session.evidence_cache.append(c)
                            if len(session.evidence_cache) > settings.EVIDENCE_CACHE_MAX_SIZE:
                                session.evidence_cache.pop(0)
            from app.core.memory import DialogueTurn
            turn = DialogueTurn(
                turn_id=len(session.working_memory.turns),
                user_message=q,
                assistant_reply=f"[test response for {scenario}]",
                intent="",
                rewritten_query=rw.rewritten_query,
            )
            session.working_memory.append(turn)
        except Exception as e:
            record(f"场景: {scenario}", "FAIL", str(e)[:200])


# ------------------------------------------------------------------
# 5. 兜底与容错测试
# ------------------------------------------------------------------
async def test_resilience():
    section("5. 兜底与容错测试")

    session = SessionMemory(session_id=f"test-res-{int(time.time())}")

    # 5.1 LLM 超时 fallback (用极短 timeout 模拟)
    try:
        client = LLMClient(
            base_url=settings.CHAT_BASE_URL,
            api_key=settings.CHAT_API_KEY,
            model=settings.CHAT_MODEL,
        )
        # 正常调用
        resp = await client.generate(
            prompt="1+1=", system="", max_tokens=5, timeout=10
        )
        record("LLM 正常调用", "PASS", resp[:30])

        # 超时应触发重试后失败
        try:
            await client.generate(
                prompt="详细解释量子力学" * 100, system="", max_tokens=1000, timeout=0.001
            )
            record("LLM 超时处理", "FAIL", "应超时但未超时")
        except Exception:
            record("LLM 超时处理", "PASS", "正确抛出超时异常")
    except Exception as e:
        record("LLM 超时处理", "FAIL", str(e)[:120])

    # 5.2 空查询
    try:
        rewriter = QueryRewriter()
        rw = await rewriter.rewrite(raw_query="", session=session)
        record("空查询处理", "PASS", f"type={rw.follow_up_type}")
    except Exception as e:
        record("空查询处理", "FAIL", str(e)[:120])

    # 5.3 超长查询
    try:
        long_q = "Python " * 500
        rw = await rewriter.rewrite(raw_query=long_q, session=session)
        record("超长查询处理", "PASS", f"query_len={len(rw.rewritten_query)}")
    except Exception as e:
        record("超长查询处理", "FAIL", str(e)[:120])

    # 5.4 无意义查询
    try:
        rw = await rewriter.rewrite(raw_query="asdfghjkl", session=session)
        record("无意义查询处理", "PASS", f"type={rw.follow_up_type}")
    except Exception as e:
        record("无意义查询处理", "FAIL", str(e)[:120])


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
async def main():
    print("\n" + "=" * 60)
    print("  v2 链路全面测试")
    print("=" * 60)
    print(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Chat 模型: {settings.CHAT_MODEL}")
    print(f"  Embedding: {settings.EMBEDDING_MODEL}")
    print(f"  Tavily: {'启用' if settings.TAVILY_SEARCH_ENABLED else '禁用'}")
    print(f"  Reranker: {'启用' if settings.RERANKER_ENABLED else '禁用'}")
    print(f"  EvidenceCache: {'启用' if settings.EVIDENCE_CACHE_ENABLED else '禁用'}")
    print("=" * 60)

    await test_infrastructure()
    await test_tools()
    await test_core_pipeline()
    await test_business_scenarios()
    await test_resilience()

    ok = summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
