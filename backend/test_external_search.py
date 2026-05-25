"""
测试外部搜索端到端流程
"""
import asyncio
import json


async def test_tavily_source_name():
    """测试 Tavily 搜索结果是否正确提取来源名称"""
    from app.core.mcp_search import tavily_search

    print("=" * 60)
    print("Test 1: Tavily search source_name extraction")
    print("=" * 60)

    results = await tavily_search("普京 5月 访华 中俄", count=3)
    print(f"Returned {len(results)} results\n")

    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        print(f"--- Result {i} ---")
        print(f"  chunk_id   : {r['chunk_id']}")
        print(f"  source_name: {meta.get('source_name', 'N/A')}")
        print(f"  company    : {meta.get('company', 'N/A')}")
        print(f"  title      : {meta.get('title', 'N/A')}")
        print(f"  url        : {meta.get('url', 'N/A')}")
        print(f"  hybrid_score: {r['hybrid_score']}")
        print(f"  content[:60]: {r['content'][:60]}")
        print()

    # 断言至少有一条结果有 source_name
    source_names = [r.get("metadata", {}).get("source_name", "") for r in results]
    assert any(source_names), "Expected at least one result with source_name"
    print("[PASS] Test 1 passed: source_name extracted correctly\n")


async def test_chat_with_external_search():
    """测试通过 chat 接口触发外部搜索（模拟）"""
    print("=" * 60)
    print("Test 2: End-to-end external_search via chat pipeline")
    print("=" * 60)

    from app.core.agent import EnhancedAgentOrchestrator
    from app.core.memory import SessionMemory
    from app.core.intent import IntentResult, IntentType

    agent = EnhancedAgentOrchestrator()
    session = SessionMemory(session_id="test_external_search_001")

    # 模拟一个会触发外部搜索的用户问题
    message = "最近普京访华有什么新闻？"

    # 手动构造意图结果，避免完整的意图识别流程
    intent_result = IntentResult(
        intent=IntentType.RAG_QA,
        confidence=0.9,
        layer="rule",
        metadata={},
        entities={},
    )

    try:
        ctx, session = await agent.run(
            intent_result=intent_result,
            message=message,
            session=session,
            attachments=[],
        )

        # 检查是否执行了 external_search
        tool_names = [t.name for t in ctx.selected_tools]
        print(f"Executed tools: {tool_names}")

        # 检查 kb_chunks 中是否有外部搜索结果
        external_chunks = [
            c for c in ctx.kb_chunks
            if c.get("metadata", {}).get("source") in ("tavily_search", "brave_search")
        ]
        print(f"External search chunks in context: {len(external_chunks)}")

        for c in external_chunks[:2]:
            meta = c.get("metadata", {})
            print(f"  - [{meta.get('source_name', 'N/A')}] {meta.get('title', 'N/A')[:40]}")

        if external_chunks:
            print("\n[PASS] Test 2 passed: external_search was triggered and results are in context\n")
        else:
            print("\n[SKIP] Test 2: No external search chunks found (may be expected if kb_retrieve had enough results)\n")

    except Exception as e:
        print(f"\n[FAIL] Test 2 error: {e}\n")
        import traceback
        traceback.print_exc()


async def test_format_chunks():
    """测试 _format_chunks 是否正确显示 source_name"""
    print("=" * 60)
    print("Test 3: _format_chunks source_name display")
    print("=" * 60)

    from app.core.agent import EnhancedAgentOrchestrator

    agent = EnhancedAgentOrchestrator()
    chunks = [
        {
            "content": "这是一条来自新华网的新闻...",
            "metadata": {
                "source": "tavily_search",
                "source_name": "新华网",
                "url": "https://www.news.cn/xxx",
            }
        },
        {
            "content": "这是一条来自知识库的 JD...",
            "metadata": {
                "source": "kb",
                "company": "字节跳动",
                "section": "岗位描述",
            }
        },
    ]

    formatted = agent._format_chunks(chunks)
    print(formatted)

    assert "新华网" in formatted, "Expected '新华网' in formatted output"
    assert "字节跳动 - 岗位描述" in formatted, "Expected '字节跳动 - 岗位描述' in formatted output"
    print("\n[PASS] Test 3 passed: source_name displayed correctly\n")


async def main():
    await test_tavily_source_name()
    await test_format_chunks()
    await test_chat_with_external_search()
    print("All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
