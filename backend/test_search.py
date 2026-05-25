#!/usr/bin/env python3
"""
外部搜索测试 - 结果保存到文件
用法: cd backend && python test_search.py "你的搜索问题"
结果会保存到 test_search_result.json
"""
import asyncio
import json
import sys

sys.path.insert(0, '.')

from app.core.mcp_search import tavily_search


async def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "普京访华最新新闻"
    print(f"Searching: {query}")

    results = await tavily_search(query, count=5)

    output = {
        "query": query,
        "count": len(results),
        "results": []
    }

    for r in results:
        meta = r.get("metadata", {})
        content = r.get("content", "")
        # 去掉开头的来源标记，提取正文
        body = content
        if body.startswith("【来源："):
            parts = body.split("\n", 1)
            body = parts[1] if len(parts) > 1 else ""

        output["results"].append({
            "source_name": meta.get("source_name", "未知"),
            "title": meta.get("title", ""),
            "url": meta.get("url", ""),
            "tavily_score": meta.get("tavily_score", 0),
            "content_preview": body[:300],
        })

    with open("test_search_result.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Done! Saved {len(results)} results to test_search_result.json")


if __name__ == "__main__":
    asyncio.run(main())
