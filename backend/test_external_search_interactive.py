#!/usr/bin/env python3
"""
交互式外部搜索测试脚本
用法: cd backend && python test_external_search_interactive.py
"""
import asyncio
import sys

sys.path.insert(0, '.')

from app.core.mcp_search import tavily_search


async def search_and_display(query: str):
    """执行搜索并格式化输出"""
    print(f"\n{'='*60}")
    print(f"搜索: {query}")
    print('='*60)

    results = await tavily_search(query, count=5)

    if not results:
        print("未返回结果。可能原因：")
        print("  - TAVILY_SEARCH_ENABLED=false")
        print("  - TAVILY_API_KEY 未配置或无效")
        print("  - 已达免费额度上限（1000次/月）")
        return

    print(f"共返回 {len(results)} 条结果:\n")

    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        source_name = meta.get("source_name", "未知来源")
        title = meta.get("title", "")
        url = meta.get("url", "")
        content = r.get("content", "")

        # 提取正文（去掉开头的【来源：xxx】标记）
        body = content
        if body.startswith("【来源："):
            body = body.split("\n", 1)[1] if "\n" in body else ""

        print(f"--- 结果 {i} ---")
        print(f"[来源: {source_name}]")
        print(f"标题: {title}")
        print(f"链接: {url}")
        print(f"内容: {body[:200]}...")
        print()


async def main():
    print("="*60)
    print("外部搜索交互式测试")
    print("输入问题直接测试 Tavily 搜索，输入 q 退出")
    print("="*60)

    while True:
        try:
            query = input("\n请输入搜索问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not query:
            continue
        if query.lower() in ('q', 'quit', 'exit'):
            print("再见!")
            break

        await search_and_display(query)


if __name__ == "__main__":
    asyncio.run(main())
