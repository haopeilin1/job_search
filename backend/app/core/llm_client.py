"""
OpenAI 兼容 LLM 客户端

支持用户自定义的 base_url、api_key、model，用于：
1. 意图识别（LLMClassifier）
2. 对话回复生成（各 Handler）

特性：
- 指数退避重试（最多 2 次额外重试）
- 分层超时：轻量 15s / 标准 30s / 重型 60s
"""

import asyncio
import json
import logging
from typing import Dict, Optional

import httpx

from app.core.config import settings
from app.core.state import llm_config_store

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════
# 分层超时常量（秒）
# ═══════════════════════════════════════════════════════

TIMEOUT_LIGHT = 60.0     # 轻量调用：token 少、快速分类（本地模型加载慢，适当放宽）
TIMEOUT_STANDARD = 180.0 # 标准调用：常规生成（本地模型可能部分层在CPU，给足时间）
TIMEOUT_HEAVY = 300.0    # 重型调用：最终聚合回复（本地27B模型推理较慢，给足时间）

# 可重试的错误类型：超时、连接错误、5xx 服务端错误
_RETRYABLE_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.NetworkError,
)


# 全局连接池：按 base_url 复用 AsyncClient，减少 TCP 握手开销
_async_client_pool: Dict[str, httpx.AsyncClient] = {}


def _get_async_client(base_url: str, timeout: float) -> httpx.AsyncClient:
    """获取或创建复用的 AsyncClient"""
    key = f"{base_url}:{timeout}"
    if key not in _async_client_pool:
        _async_client_pool[key] = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )
    return _async_client_pool[key]


class LLMClient:
    """
    轻量 LLM 客户端，直接通过 httpx 调用 OpenAI 兼容 Chat Completions API。
    
    支持指数退避重试和分层超时。
    使用连接池复用，减少每次请求的 TCP 连接开销。
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = TIMEOUT_HEAVY,
    ):
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or ""
        self.model = model or "gpt-4o"
        self.timeout = timeout

    @classmethod
    def from_chat_config(cls) -> "LLMClient":
        """从全局状态读取用户配置的问答模型参数创建客户端"""
        cfg = llm_config_store.chat
        return cls(base_url=cfg.base_url, api_key=cfg.api_key, model=cfg.model)

    @classmethod
    def from_vision_config(cls) -> "LLMClient":
        """从全局状态读取用户配置的多模态模型参数创建客户端"""
        cfg = llm_config_store.vision
        return cls(base_url=cfg.base_url, api_key=cfg.api_key, model=cfg.model)

    @classmethod
    def from_config(cls, name: str) -> "LLMClient":
        """
        按功能分层从全局状态读取模型配置创建客户端。

        模型分层与 fallback 关系（配置层面）：
        - chat:   基础层，最终聚合、通用对话、L3 仲裁
        - core:   match_analyze / interview_gen / attribute_verify 等核心工具调用
                  若 BASE_URL/API_KEY/MODEL 留空，自动复用 chat 配置
        - planner: 动态任务规划（TaskPlanner.create_graph）
                  若留空，自动复用 chat 配置
        - memory:  记忆轮转、摘要生成
                  若留空，自动复用 chat 配置
        - vision:  多模态理解（简历图片 OCR）

        因此当前环境中：match_analyze（core 层）和 interview_gen（core 层）
        实际与 chat 层使用完全相同的模型和 endpoint。

        Args:
            name: 配置层名称 — "chat" | "core" | "planner" | "memory" | "vision"

        Returns:
            对应分层的 LLMClient 实例
        """
        name = name.lower()
        if name == "chat":
            cfg = llm_config_store.chat
        elif name == "core":
            cfg = llm_config_store.core
        elif name == "planner":
            cfg = llm_config_store.planner
        elif name == "rewrite":
            cfg = llm_config_store.rewrite
        elif name == "memory":
            cfg = llm_config_store.memory
        elif name == "vision":
            cfg = llm_config_store.vision
        elif name == "judge":
            cfg = llm_config_store.judge
        else:
            logger.warning(f"[LLMClient] 未知配置层 '{name}'，fallback 到 chat")
            cfg = llm_config_store.chat
        return cls(base_url=cfg.base_url, api_key=cfg.api_key, model=cfg.model)

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        timeout: Optional[float] = None,
    ) -> str:
        """
        调用 Chat Completions API（带指数退避重试）。

        Args:
            messages: OpenAI 格式消息列表
            temperature: 采样温度
            max_tokens: 最大生成 token 数
            json_mode: 是否请求 JSON 格式输出
            timeout: 本次调用的超时秒数，覆盖实例默认值。
                     建议使用 TIMEOUT_LIGHT / TIMEOUT_STANDARD / TIMEOUT_HEAVY。

        Returns:
            LLM 生成的文本内容
        """
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        # Ollama 本地模型：设置 keep_alive 避免 5 分钟卸载
        if "localhost" in self.base_url or "127.0.0.1" in self.base_url:
            payload["keep_alive"] = "1h"

        effective_timeout = timeout if timeout is not None else self.timeout
        max_retries = 2  # 最多 2 次额外重试（共 3 次尝试）
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                client = _get_async_client(self.base_url, effective_timeout)
                resp = await client.post(url, headers=headers, json=payload, timeout=effective_timeout)
                resp.raise_for_status()
                data = resp.json()

                msg = data["choices"][0]["message"]
                content = msg.get("content", "")
                # Qwen3 系列模型将思考内容放在 reasoning 字段，content 可能为空
                if not content and msg.get("reasoning"):
                    content = msg["reasoning"]
                if attempt > 0:
                    logger.info(
                        f"[LLMClient] 第 {attempt + 1} 次尝试成功 | model={self.model}"
                    )
                else:
                    logger.info(
                        f"[LLMClient] model={self.model} "
                        f"prompt_tokens={data.get('usage', {}).get('prompt_tokens', '?')} "
                        f"completion_tokens={data.get('usage', {}).get('completion_tokens', '?')}"
                    )
                return content.strip()

            except httpx.HTTPStatusError as e:
                # 4xx 客户端错误不重试（如 401 认证失败、429 配额超限）
                if 400 <= e.response.status_code < 500:
                    logger.error(
                        f"[LLMClient] HTTP {e.response.status_code} 客户端错误，不重试: {e.response.text[:200]}"
                    )
                    raise RuntimeError(f"LLM API 错误 ({e.response.status_code}): {e.response.text}")
                # 5xx 服务端错误可重试
                logger.warning(
                    f"[LLMClient] HTTP {e.response.status_code} 服务端错误，"
                    f"尝试 {attempt + 1}/{max_retries + 1}: {e.response.text[:200]}"
                )
                last_exception = e

            except _RETRYABLE_EXCEPTIONS as e:
                logger.warning(
                    f"[LLMClient] 可重试异常 ({type(e).__name__})，"
                    f"尝试 {attempt + 1}/{max_retries + 1}: {e}"
                )
                last_exception = e

            except Exception as e:
                logger.error(f"[LLMClient] 不可重试异常: {e}")
                raise RuntimeError(f"LLM 请求失败: {e}")

            # 指数退避：第1次重试等1s，第2次等2s
            if attempt < max_retries:
                delay = 2 ** attempt  # 1s, 2s
                logger.info(f"[LLMClient] {delay}s 后重试...")
                await asyncio.sleep(delay)

        # 所有重试耗尽
        logger.error(f"[LLMClient] {max_retries + 1} 次尝试均失败，放弃")
        raise RuntimeError(f"LLM 请求失败（已重试 {max_retries} 次）: {last_exception}")

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> str:
        """
        简化接口：单条 prompt -> 单条回复（带重试和分层超时）。

        Args:
            prompt: 用户提示词
            system: 系统提示词
            timeout: 本次调用的超时秒数，覆盖实例默认值。
                     建议使用 TIMEOUT_LIGHT / TIMEOUT_STANDARD / TIMEOUT_HEAVY。
            **kwargs: 传递给 chat() 的其他参数

        Returns:
            LLM 生成的文本内容
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return await self.chat(messages, timeout=timeout, **kwargs)

    async def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: Optional[int] = 1500,
        timeout: Optional[float] = None,
    ):
        """
        流式生成：返回异步生成器，逐 token yield 文本片段。
        用于 SSE 流式输出最终聚合回复。
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        # Ollama 本地模型：设置 keep_alive 避免 5 分钟卸载
        if "localhost" in self.base_url or "127.0.0.1" in self.base_url:
            payload["keep_alive"] = "1h"

        effective_timeout = timeout if timeout is not None else self.timeout
        client = _get_async_client(self.base_url, effective_timeout)

        async with client.stream(
            "POST", url, headers=headers, json=payload, timeout=effective_timeout
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except Exception:
                        continue

    async def vision_chat(
        self,
        system_prompt: str,
        image_data_uris: list[str],
        temperature: float = 0.3,
        max_tokens: Optional[int] = 4096,
        timeout: Optional[float] = None,
    ) -> str:
        """
        多模态 LLM 调用（异步，带重试）。

        Args:
            system_prompt: 系统提示词 / 用户文本提示
            image_data_uris: base64 data URI 列表
            temperature: 采样温度
            max_tokens: 最大生成 token 数
            timeout: 本次调用的超时秒数，覆盖实例默认值

        Returns:
            LLM 生成的文本内容
        """
        content = [{"type": "text", "text": system_prompt}]
        for uri in image_data_uris:
            content.append({"type": "image_url", "image_url": {"url": uri}})
        messages = [{"role": "user", "content": content}]

        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        effective_timeout = timeout if timeout is not None else self.timeout
        max_retries = 2
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                client = _get_async_client(self.base_url, effective_timeout)
                resp = await client.post(url, headers=headers, json=payload, timeout=effective_timeout)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                if attempt > 0:
                    logger.info(
                        f"[LLMClient] vision 第 {attempt + 1} 次尝试成功 | model={self.model}"
                    )
                else:
                    logger.info(
                        f"[LLMClient] vision model={self.model} "
                        f"prompt_tokens={data.get('usage', {}).get('prompt_tokens', '?')} "
                        f"completion_tokens={data.get('usage', {}).get('completion_tokens', '?')}"
                    )
                return content.strip()

            except httpx.HTTPStatusError as e:
                if 400 <= e.response.status_code < 500:
                    logger.error(
                        f"[LLMClient] vision HTTP {e.response.status_code} 客户端错误，不重试"
                    )
                    raise RuntimeError(f"Vision LLM API 错误 ({e.response.status_code}): {e.response.text}")
                logger.warning(
                    f"[LLMClient] vision HTTP {e.response.status_code} 服务端错误，"
                    f"尝试 {attempt + 1}/{max_retries + 1}"
                )
                last_exception = e

            except _RETRYABLE_EXCEPTIONS as e:
                logger.warning(
                    f"[LLMClient] vision 可重试异常 ({type(e).__name__})，"
                    f"尝试 {attempt + 1}/{max_retries + 1}: {e}"
                )
                last_exception = e

            except Exception as e:
                logger.error(f"[LLMClient] vision 不可重试异常: {e}")
                raise RuntimeError(f"Vision LLM 请求失败: {e}")

            if attempt < max_retries:
                delay = 2 ** attempt
                logger.info(f"[LLMClient] vision {delay}s 后重试...")
                await asyncio.sleep(delay)

        logger.error(f"[LLMClient] vision {max_retries + 1} 次尝试均失败，放弃")
        raise RuntimeError(f"Vision LLM 请求失败（已重试 {max_retries} 次）: {last_exception}")

    # ──────────────────────────── 同步静态方法（供简历解析等同步场景使用） ────────────────────────────

    @staticmethod
    def _sync_chat(
        base_url: str,
        api_key: str,
        model: str,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        """同步调用 Chat Completions API（供静态方法内部使用）"""
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

    @staticmethod
    def call_text(
        base_url: str,
        api_key: str,
        model: str,
        system_prompt: str,
        user_text: str,
    ) -> str:
        """文本 LLM 调用（简历解析等场景使用）"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        return LLMClient._sync_chat(base_url, api_key, model, messages, temperature=0.3)

    @staticmethod
    def call_vision(
        base_url: str,
        api_key: str,
        model: str,
        system_prompt: str,
        image_data_uris: list[str],
    ) -> str:
        """多模态 LLM 调用（PDF 图片兜底解析场景使用）"""
        content = [{"type": "text", "text": system_prompt}]
        for uri in image_data_uris:
            content.append({"type": "image_url", "image_url": {"url": uri}})
        messages = [{"role": "user", "content": content}]
        return LLMClient._sync_chat(base_url, api_key, model, messages, temperature=0.3)

    @staticmethod
    def safe_parse_json(raw: str) -> Optional[dict]:
        """安全解析 JSON，失败返回 None"""
        if not raw or not raw.strip():
            return None
        try:
            # 尝试直接解析
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            pass
        # 尝试从 markdown 代码块中提取
        try:
            text = raw.strip()
            if "```json" in text:
                start = text.find("```json") + 7
                end = text.find("```", start)
                if end != -1:
                    return json.loads(text[start:end].strip())
            if "```" in text:
                start = text.find("```") + 3
                end = text.find("```", start)
                if end != -1:
                    return json.loads(text[start:end].strip())
            # 尝试找第一个 { 到最后一个 }
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            pass
        return None
