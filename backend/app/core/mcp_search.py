"""
外部搜索工具 —— Tavily Search（唯一）

职责：
1. 当内部知识库检索不足时，通过 Tavily 补充外部信息
2. 获取公司最新新闻、融资动态、组织架构调整等实时信息
3. 将搜索结果转换为与 kb_retrieve chunks 兼容的格式

使用方式：
- 由 Plan 模块或 ReAct 执行器动态决定是否调用
- 不是由意图硬编码触发

Tavily 注册: https://tavily.com/
- 国内可直接访问 api.tavily.com
- 免费额度 1000 次/月
- 输出自带网页摘要，直接适配 RAG
"""

import json
import logging
from typing import Dict, List, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

# 搜索调用计数（用于免费额度熔断）
_SEARCH_CALL_COUNT = 0
_TAVILY_MAX_FREE_CALLS = 950  # Tavily 免费额度 1000 次/月，留 50 次 buffer


# ═══════════════════════════════════════════════════════
# 1. Tavily Search（推荐，国内可访问）
# ═══════════════════════════════════════════════════════

async def tavily_search(query: str, count: int = None) -> List[Dict]:
    """
    调用 Tavily Search API。
    返回与 kb_retrieve chunks 兼容的格式。

    Args:
        query: 搜索关键词
        count: 返回结果数量，默认读取配置

    Returns:
        chunks 列表，每个 chunk 格式与 kb_retrieve 输出一致
    """
    global _SEARCH_CALL_COUNT

    if not settings.TAVILY_SEARCH_ENABLED:
        logger.debug("[Tavily] 已禁用，返回空结果")
        return []

    if not settings.TAVILY_API_KEY:
        logger.warning("[Tavily] API Key 未配置，返回空结果")
        return []

    # 熔断检查
    if _SEARCH_CALL_COUNT >= _TAVILY_MAX_FREE_CALLS:
        logger.warning("[Tavily] 已达免费额度上限，返回空结果")
        return []

    if count is None:
        count = settings.TAVILY_MAX_RESULTS

    # 截断过长 query
    max_len = 400  # Tavily 建议 query 长度
    if len(query) > max_len:
        query = query[:max_len]

    try:
        results = await _search_via_tavily_http(query, count)
    except Exception as e:
        logger.error(f"[Tavily] 搜索失败: {e}，返回空结果")
        return []

    _SEARCH_CALL_COUNT += 1
    logger.info(
        f"[Tavily] 返回 {len(results)} 条结果 | "
        f"query='{query[:40]}...' | 累计调用 {_SEARCH_CALL_COUNT} 次"
    )
    return results


async def _search_via_tavily_http(query: str, count: int) -> List[Dict]:
    """通过 HTTP 调用 Tavily Search API"""
    import httpx

    url = "https://api.tavily.com/search"
    payload = {
        "api_key": settings.TAVILY_API_KEY,
        "query": query,
        "search_depth": settings.TAVILY_SEARCH_DEPTH,
        "include_answer": settings.TAVILY_INCLUDE_ANSWER,
        "max_results": count,
    }

    # 可选：限定/排除域名
    if settings.TAVILY_INCLUDE_DOMAINS:
        payload["include_domains"] = settings.TAVILY_INCLUDE_DOMAINS
    if settings.TAVILY_EXCLUDE_DOMAINS:
        payload["exclude_domains"] = settings.TAVILY_EXCLUDE_DOMAINS

    async with httpx.AsyncClient(timeout=settings.TAVILY_TIMEOUT) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

    results = []
    for item in data.get("results", []):
        results.append(_format_tavily_result(item))

    return results


def _format_tavily_result(item: Dict) -> Dict:
    """将 Tavily 搜索结果格式化为与 JD chunk 兼容的格式"""
    title = item.get("title", "")
    url = item.get("url", "")
    # Tavily 的 content 已经是网页内容提取/摘要，直接可用
    content = item.get("content", "")
    score = item.get("score", 0.0)

    # 如果没有 content，用 title 兜底
    display_content = content if content else title

    full_content = f"【来源：网络搜索】{title}\n{display_content}\n参考链接：{url}"

    return {
        "chunk_id": f"tavily_{hash(url) & 0xFFFFFFFF}",
        "content": full_content,
        "metadata": {
            "source": "tavily_search",
            "url": url,
            "title": title,
            "company": _extract_company_from_text(title + " " + display_content),
            "position": "",
            "section": "external_search",
            "jd_id": f"ext_{hash(url) & 0xFFFFFFFF}",
            "tavily_score": score,
        },
        "distance": None,
        "bm25_score": 0.0,
        "hybrid_score": 0.5,  # 外部搜索结果给一个中等偏上的统一分数
        "rerank_score": None,
    }


# ═══════════════════════════════════════════════════════
# 2. Brave Search（备用，国内不可访问）
# ═══════════════════════════════════════════════════════

async def brave_search(query: str, count: int = None) -> List[Dict]:
    """
    通过 MCP 调用 Brave Search API（备用方案）。
    返回与 kb_retrieve chunks 兼容的格式。
    """
    if not settings.BRAVE_SEARCH_MCP_ENABLED:
        logger.debug("[Brave] 已禁用")
        return []

    if not settings.BRAVE_SEARCH_API_KEY:
        logger.warning("[Brave] API Key 未配置，跳过外部搜索")
        return []

    if count is None:
        count = settings.BRAVE_SEARCH_RESULT_COUNT

    # 截断过长 query
    if len(query) > settings.BRAVE_SEARCH_MAX_QUERY_LENGTH:
        query = query[:settings.BRAVE_SEARCH_MAX_QUERY_LENGTH]

    try:
        results = await _search_via_mcp(query, count)
    except ImportError:
        logger.warning("[Brave] mcp SDK 未安装，尝试直接 HTTP 调用")
        results = await _search_via_brave_http(query, count)
    except Exception as e:
        logger.error(f"[Brave] 搜索调用失败: {e}")
        return []

    logger.info(f"[Brave] 返回 {len(results)} 条结果")
    return results


async def _search_via_mcp(query: str, count: int) -> List[Dict]:
    """通过 MCP SDK 调用 Brave Search"""
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        raise ImportError("mcp SDK 未安装")

    import os

    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-brave-search"],
        env={"BRAVE_API_KEY": settings.BRAVE_SEARCH_API_KEY, **dict(os.environ)},
    )

    results = []
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            search_result = await session.call_tool(
                "brave_web_search",
                arguments={"query": query, "count": count}
            )

            for item in search_result.content:
                data = json.loads(item.text) if hasattr(item, "text") else {}
                for web_result in data.get("web", {}).get("results", []):
                    results.append(_format_brave_result(web_result))

    return results


async def _search_via_brave_http(query: str, count: int) -> List[Dict]:
    """直接通过 HTTP 调用 Brave Search API"""
    import httpx

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": settings.BRAVE_SEARCH_API_KEY,
    }
    params = {
        "q": query,
        "count": count,
        "offset": 0,
        "mkt": "zh-CN",
        "safesearch": "moderate",
        "freshness": "pw",
        "text_decorations": False,
        "spellcheck": True,
    }

    async with httpx.AsyncClient(timeout=settings.BRAVE_SEARCH_TIMEOUT) as client:
        resp = await client.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()

    results = []
    for web_result in data.get("web", {}).get("results", []):
        results.append(_format_brave_result(web_result))

    return results


def _format_brave_result(web_result: Dict) -> Dict:
    """将 Brave 搜索结果格式化为与 JD chunk 兼容的格式"""
    title = web_result.get("title", "")
    desc = web_result.get("description", "")
    url = web_result.get("url", "")

    content = f"【来源：网络搜索】{title}\n{desc}\n参考链接：{url}"

    return {
        "chunk_id": f"brave_{hash(url) & 0xFFFFFFFF}",
        "content": content,
        "metadata": {
            "source": "brave_search",
            "url": url,
            "title": title,
            "company": _extract_company_from_text(title + " " + desc),
            "position": "",
            "section": "external_search",
            "jd_id": f"ext_{hash(url) & 0xFFFFFFFF}",
        },
        "distance": None,
        "bm25_score": 0.0,
        "hybrid_score": 0.5,
        "rerank_score": None,
    }


def _extract_company_from_text(text: str) -> str:
    """从搜索结果文本中提取公司名"""
    companies = [
        "字节跳动", "百度", "阿里", "腾讯", "美团", "京东", "小米",
        "拼多多", "网易", "华为", "快手", "滴滴", "B站", "哔哩哔哩",
        "蚂蚁", "支付宝", "阿里云", "腾讯云", "高德", "知乎",
        "小红书", "贝壳", "携程", "饿了么", "得物", "OPPO", "vivo",
        "Shopee", "Google", "Meta", "Amazon", "Microsoft", "Apple",
    ]
    for c in companies:
        if c in text:
            return c
    return "未知"


# ═══════════════════════════════════════════════════════
# 3. ExternalSearchTool 封装（供 ToolRegistry / ReAct 执行器调用）
# ═══════════════════════════════════════════════════════

class ExternalSearchTool:
    """外部搜索工具（供 react_executor 直接调用）"""

    @property
    def name(self) -> str:
        return "external_search"

    @property
    def description(self) -> str:
        return (
            "通过 Tavily Search 搜索外部网络信息（国内可直接访问）。"
            "用于获取内部知识库没有的最新/实时信息，"
            "如公司新闻、融资动态、组织架构调整、行业趋势等。"
            "Tavily 会自动提取网页内容摘要，直接适配 RAG 场景。"
        )

    @property
    def input_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，优先使用 search_keywords 或 rewritten_query",
                },
                "count": {
                    "type": "integer",
                    "default": 5,
                    "description": "返回结果数量，默认 5 条",
                },
            },
            "required": ["query"],
        }

    async def execute(self, params: Dict) -> Dict:
        query = params.get("query", "")
        count = params.get("count", settings.TAVILY_MAX_RESULTS)

        if not query:
            return {
                "success": False,
                "error": "query 为空",
                "data": {"chunks": []},
            }

        # 优先 Tavily，失败自动 fallback 到 Brave
        chunks = await tavily_search(query, count)
        source = "tavily_search" if chunks and chunks[0].get("metadata", {}).get("source") == "tavily_search" else "brave_search"

        return {
            "success": True,
            "data": {
                "chunks": chunks,
                "query": query,
                "source": source,
            },
            "observation": f"外部搜索（{source}）返回 {len(chunks)} 条结果",
        }
