#!/usr/bin/env python3
"""
一键对比两种 Chunk 切分策略的切分结果

支持两种模式：
1. Mock 模式（默认）：粗略构造 JD schema，直接对比切分策略
2. LLM 解析模式（--parse）：先调用配置的 LLM 解析 JD 原文，再用真实 schema 对比

用法:
    cd backend

    # Mock 模式（内置 Demo JD）
    python scripts/compare_chunk_strategies.py

    # Mock 模式 + 自定义文本
    python scripts/compare_chunk_strategies.py --file path/to/jd.txt
    python scripts/compare_chunk_strategies.py --text "负责推荐算法优化。要求3年以上经验..."

    # LLM 解析模式（需先在 .env 中配置 LLM）
    python scripts/compare_chunk_strategies.py --parse --file path/to/jd.txt
    python scripts/compare_chunk_strategies.py --parse --text "岗位职责：...\n硬性要求：..."
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Windows 控制台默认 GBK，强制设置 UTF-8 避免中文乱码
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# 把 backend 加入路径，确保能 import app.*
BACKEND_ROOT = Path(__file__).parent.parent.resolve()
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.chunking import chunk_fixed_size, chunk_semantic


def build_jd_schema(raw_text: str, title: str = "测试岗位") -> dict:
    """粗略地把原始文本包装成 JDSchema 格式的字典（Mock 模式用）"""
    return {
        "jd_id": "demo-001",
        "company": "演示公司",
        "position": title,
        "location": "北京",
        "salary_range": "30k-50k",
        "raw_text": raw_text,
        "sections": {
            "responsibilities": raw_text[: len(raw_text) // 2],
            "hard_requirements": [
                raw_text[i : i + 40] for i in range(0, min(len(raw_text), 200), 40)
            ],
            "soft_requirements": ["有顶会论文优先", "熟悉 TensorFlow/PyTorch"],
        },
        "keywords": ["Python", "算法", "机器学习"],
    }


async def parse_with_llm(raw_text: str) -> dict:
    """调用 LLM 解析 JD 原文为结构化 Schema"""
    from app.core.llm_client import LLMClient
    from app.core.jd_parser import JDParser

    llm = LLMClient.from_chat_config()
    if not llm.api_key:
        print("[ERROR] LLM API Key 未配置，请先在 backend/.env 中填写 CHAT_API_KEY 等参数")
        sys.exit(1)

    parser = JDParser(llm_client=llm)
    parsed = await parser.parse(raw_text, source_type="test")
    return parsed


def print_divider(title: str, width: int = 70):
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_schema(parsed: dict):
    """打印解析后的 Schema"""
    print_divider("LLM 解析后的 JD Schema", width=70)
    # 精简打印，避免过长
    preview = {
        "jd_id": parsed.get("jd_id"),
        "company": parsed.get("company"),
        "position": parsed.get("position"),
        "location": parsed.get("location"),
        "salary_range": parsed.get("salary_range"),
        "keywords": parsed.get("keywords"),
        "sections": parsed.get("sections"),
    }
    for line in json.dumps(preview, ensure_ascii=False, indent=2).split("\n"):
        print(f"  {line}")


def print_chunks(chunks, strategy_name: str):
    print_divider(f"策略: {strategy_name}  (共 {len(chunks)} 个 chunk)")

    for i, c in enumerate(chunks):
        meta = c.metadata
        content = c.content
        lines = content.split("\n")
        preview_lines = []
        char_count = 0
        for line in lines:
            if char_count + len(line) > 300:
                remaining = 300 - char_count
                preview_lines.append(line[:remaining] + " ...")
                break
            preview_lines.append(line)
            char_count += len(line) + 1  # +1 for newline

        preview = "\n              ".join(preview_lines)

        section = meta.get("section", "-")
        priority = meta.get("priority", "-")
        index = meta.get("index", i)

        print(f"\n  ┌─ chunk[{index}]  section={section:<20} priority={priority:<6} len={len(content)}")
        print(f"  │  {preview}")
        print(f"  └─")


def compare(jd: dict, show_schema: bool = True):
    raw_text = jd.get("raw_text", "")

    if show_schema:
        print_schema(jd)
    else:
        print_divider("原始 JD 文本", width=70)
        print(f"  总长度: {len(raw_text)} 字符")
        print(f"  内容预览:")
        preview = raw_text[:500] + "..." if len(raw_text) > 500 else raw_text
        for line in preview.split("\n"):
            print(f"    {line}")

    # 策略 A: 固定大小
    chunks_fixed = chunk_fixed_size(jd, size=512, overlap=50)
    print_chunks(chunks_fixed, "fixed_size (固定大小滑动窗口)")

    # 策略 B: 语义切分
    chunks_semantic = chunk_semantic(jd)
    print_chunks(chunks_semantic, "semantic (按结构化字段语义切分)")

    # 汇总对比
    print_divider("汇总对比", width=70)
    print(f"  {'指标':<30} {'fixed_size':>15} {'semantic':>15}")
    print(f"  {'-'*30} {'-'*15} {'-'*15}")
    print(f"  {'chunk 数量':<30} {len(chunks_fixed):>15} {len(chunks_semantic):>15}")
    print(f"  {'平均 chunk 长度':<30} {sum(len(c.content) for c in chunks_fixed)//max(len(chunks_fixed),1):>15} {sum(len(c.content) for c in chunks_semantic)//max(len(chunks_semantic),1):>15}")
    print(f"  {'最大 chunk 长度':<30} {max((len(c.content) for c in chunks_fixed), default=0):>15} {max((len(c.content) for c in chunks_semantic), default=0):>15}")
    print(f"  {'最小 chunk 长度':<30} {min((len(c.content) for c in chunks_fixed), default=0):>15} {min((len(c.content) for c in chunks_semantic), default=0):>15}")

    def _cut_in_sentence(chunks) -> int:
        """粗略统计有多少 chunk 在非标点处结束"""
        cuts = 0
        for c in chunks:
            text = c.content.strip()
            if text and text[-1] not in "。！？.!?,;；、\n":
                cuts += 1
        return cuts

    print(f"  {'疑似句中切断数量':<30} {_cut_in_sentence(chunks_fixed):>15} {_cut_in_sentence(chunks_semantic):>15}")
    print()


# ────────── 内置 Demo JD ──────────
DEMO_JD = """岗位职责：
1. 负责公司核心推荐算法的设计与优化，提升用户点击率与停留时长；
2. 深入理解业务场景，将机器学习技术落地到搜索、推荐、广告等场景；
3. 跟踪学术界与工业界最新进展，推动算法迭代与技术创新；
4. 与产品、工程团队紧密协作，推动算法方案从 0 到 1 上线。

硬性要求：
1. 计算机、数学、统计等相关专业本科及以上学历，3 年以上算法经验；
2. 扎实的编程能力，精通 Python/C++，熟悉 TensorFlow/PyTorch 等深度学习框架；
3. 熟悉主流推荐算法（协同过滤、矩阵分解、深度神经网络等），有大规模推荐系统实战经验；
4. 具备良好的数据敏感度，能够独立完成数据分析、特征工程、模型训练与效果评估。

软性要求/加分项：
1. 在 KDD、NeurIPS、ICML、SIGIR 等顶会有论文发表者优先；
2. 有 LLM 相关项目经验，了解 RAG、Agent 等技术栈者优先；
3. 具备优秀的跨团队沟通能力和技术影响力。"""


def main():
    parser = argparse.ArgumentParser(description="对比两种 chunk 切分策略")
    parser.add_argument("--file", "-f", type=str, help="从文件读取 JD 文本")
    parser.add_argument("--text", "-t", type=str, help="直接从命令行传入 JD 文本")
    parser.add_argument("--title", type=str, default="测试岗位", help="岗位名称（仅影响 mock 模式）")
    parser.add_argument("--parse", "-p", action="store_true", help="先调用 LLM 解析 JD，再用真实 schema 对比")
    args = parser.parse_args()

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"[ERROR] 文件不存在: {path}")
            sys.exit(1)
        raw_text = path.read_text(encoding="utf-8")
    elif args.text:
        raw_text = args.text
    else:
        raw_text = DEMO_JD
        if args.parse:
            print("[INFO] LLM 解析模式下未提供 --file 或 --text，使用内置 Demo JD 进行演示\n")
        else:
            print("[INFO] 未提供 --file 或 --text，使用内置 Demo JD 进行演示\n")

    if args.parse:
        # LLM 解析模式
        print("[INFO] 正在调用 LLM 解析 JD 原文...")
        jd = asyncio.run(parse_with_llm(raw_text))
        compare(jd, show_schema=True)
    else:
        # Mock 模式
        jd = build_jd_schema(raw_text, title=args.title)
        compare(jd, show_schema=False)


if __name__ == "__main__":
    main()
