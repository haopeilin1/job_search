"""
模型 Token 成本估算器。

设计原则：
1. 不依赖真实账单 API，基于字符数粗略估算
2. 支持通义千问系列定价
3. 中文字符 ≈ 1.5 token（经验值）
"""

# 单价：USD / 1M tokens（通义千问参考价，2024Q4）
_MODEL_PRICES = {
    "qwen-turbo": {"input": 0.3, "output": 0.6},
    "qwen-plus": {"input": 2.0, "output": 6.0},
    "qwen-vl": {"input": 4.0, "output": 8.0},
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    "gpt-4": {"input": 30.0, "output": 60.0},
    "default": {"input": 1.0, "output": 2.0},
}

# 中文字符 → token 的粗略换算系数
_CHARS_PER_TOKEN = 1.5


def estimate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """
    估算单次 LLM 调用的成本（USD）。

    Args:
        model_name: 模型标识
        input_tokens: 输入 token 数
        output_tokens: 输出 token 数

    Returns:
        估算成本（美元）
    """
    prices = _MODEL_PRICES.get(model_name, _MODEL_PRICES["default"])
    input_cost = (input_tokens / 1_000_000) * prices["input"]
    output_cost = (output_tokens / 1_000_000) * prices["output"]
    return round(input_cost + output_cost, 6)


def estimate_tokens(text: str, chars_per_token: float = _CHARS_PER_TOKEN) -> int:
    """
    粗略估算文本对应的 token 数。

    Args:
        text: 原始文本
        chars_per_token: 每个 token 平均字符数（中文 1.5，英文 3-4）

    Returns:
        估算 token 数
    """
    if not text:
        return 0
    return int(len(text) / chars_per_token)


def estimate_from_messages(messages: list, output_text: str = "", model_name: str = "default") -> dict:
    """
    从 messages 列表估算成本（OpenAI 兼容格式）。

    Returns:
        {"input_tokens": int, "output_tokens": int, "cost_usd": float}
    """
    input_text = ""
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            input_text += content
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    input_text += part["text"]

    input_tokens = estimate_tokens(input_text)
    output_tokens = estimate_tokens(output_text)
    cost = estimate_cost(model_name, input_tokens, output_tokens)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost,
    }
