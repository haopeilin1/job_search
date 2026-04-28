#!/usr/bin/env python3
"""
终端直接测试 /api/v1/knowledge-base/chunk-test 接口

用法：
    # 1. 先启动后端（另一个终端）
    cd backend && python -m app.main

    # 2. 在本终端运行测试脚本
    cd backend
    python scripts/test_chunk_api.py

    # 从文件读取 JD
    python scripts/test_chunk_api.py --file path/to/jd.txt

    # 指定后端地址（如果改了端口）
    python scripts/test_chunk_api.py --url http://localhost:8000 --file jd.txt
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import httpx

# Windows 控制台默认 GBK，强制设置 UTF-8 避免中文乱码
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

DEFAULT_URL = "http://localhost:8000/api/v1/knowledge-base/chunk-test"

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


def print_divider(title: str, width: int = 70):
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_schema(parsed: dict):
    print_divider("LLM 解析后的 JD Schema")
    preview = {
        "company": parsed.get("company"),
        "position": parsed.get("position"),
        "location": parsed.get("location"),
        "salary_range": parsed.get("salary_range"),
        "keywords": parsed.get("keywords"),
        "sections": parsed.get("sections"),
    }
    for line in json.dumps(preview, ensure_ascii=False, indent=2).split("\n"):
        print(f"  {line}")


def print_strategy(name: str, data: dict):
    stats = data.get("stats", {})
    chunks = data.get("chunks", [])
    print_divider(f"{name}  |  chunks={stats.get('count')}  avg={stats.get('avg_len')}  max={stats.get('max_len')}  min={stats.get('min_len')}")

    for c in chunks:
        idx = c.get("index", 0)
        section = c.get("section", "-")
        priority = c.get("priority", "-")
        length = c.get("length", 0)
        content = c.get("content", "")
        preview = content.replace("\n", " ")[:120]
        if len(content) > 120:
            preview += " ..."
        print(f"  [{idx:2}] {section:<20} {priority:<6} len={length:<4} | {preview}")


def print_summary(strategies: dict):
    print_divider("汇总对比")
    names = list(strategies.keys())
    col_w = max(16, max(len(n) for n in names) + 2)
    print(f"  {'指标':<14}" + "".join(f"{n:>{col_w}}" for n in names))
    print(f"  {'-'*14}" + "".join(f"{'-'*col_w}" for _ in names))

    for metric, label in [("count", "chunk 数量"), ("avg_len", "平均长度"), ("max_len", "最大长度"), ("min_len", "最小长度")]:
        values = [strategies[n].get("stats", {}).get(metric, 0) for n in names]
        print(f"  {label:<14}" + "".join(f"{v:>{col_w}}" for v in values))
    print()


def read_stdin() -> str:
    """从标准输入读取多行文本"""
    print("请粘贴 JD 文本，输入结束后按 Ctrl+Z 然后回车（Windows）或 Ctrl+D（Mac/Linux）：")
    print("-" * 50)
    lines = sys.stdin.read()
    print("-" * 50)
    return lines


def get_clipboard_text() -> str:
    """从 Windows 剪贴板读取文本（无需额外安装包）"""
    try:
        result = subprocess.run(
            ["powershell", "-command", "Get-Clipboard"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0:
            raw = result.stdout
            # 尝试多种编码解码（Windows 剪贴板可能是 GBK 或 UTF-8）
            for enc in ("utf-8", "gbk", "utf-16-le", "cp936"):
                try:
                    return raw.decode(enc)
                except UnicodeDecodeError:
                    continue
            return raw.decode("utf-8", errors="ignore")
    except Exception:
        pass
    return ""


def run_test(raw_text: str, url: str):
    """执行一次测试并打印结果"""
    print(f"[INFO] 请求地址: {url}")
    print(f"[INFO] JD 长度: {len(raw_text)} 字符")

    try:
        resp = httpx.post(url, json={"raw_text": raw_text}, timeout=120.0)
        resp.raise_for_status()
        result = resp.json()
    except httpx.ConnectError:
        print(f"\n[ERROR] 无法连接到后端，请确认服务已启动: python -m app.main")
        return False
    except httpx.HTTPStatusError as e:
        print(f"\n[ERROR] API 返回错误: {e.response.status_code}")
        print(e.response.text)
        return False

    parsed = result.get("parsed_schema", {})
    print_schema(parsed)

    strategies = {
        "fixed_size": result.get("fixed_size", {}),
        "semantic": result.get("semantic", {}),
        "recursive_512": result.get("recursive_512", {}),
        "recursive_256": result.get("recursive_256", {}),
        "section": result.get("section", {}),
    }

    for name, data in strategies.items():
        print_strategy(name, data)

    print_summary(strategies)
    return True


def main():
    parser = argparse.ArgumentParser(description="终端测试 chunk-test API")
    parser.add_argument("--file", "-f", type=str, help="从文件读取 JD 文本")
    parser.add_argument("--text", "-t", type=str, help="直接从命令行传入 JD 文本")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互式输入：直接在终端粘贴 JD 文本")
    parser.add_argument("--clipboard", "-c", action="store_true", help="从 Windows 剪贴板读取 JD 文本（复制后直接运行）")
    parser.add_argument("--loop", "-l", action="store_true", help="循环模式：测完后自动读取新剪贴板内容继续测")
    parser.add_argument("--url", "-u", type=str, default=DEFAULT_URL, help="API 地址")
    args = parser.parse_args()

    # 非循环模式：单次执行
    if not args.loop:
        if args.file:
            path = Path(args.file)
            if not path.exists():
                print(f"[ERROR] 文件不存在: {path}")
                sys.exit(1)
            raw_text = path.read_text(encoding="utf-8")
        elif args.text:
            raw_text = args.text
        elif args.interactive:
            raw_text = read_stdin()
        elif args.clipboard:
            raw_text = get_clipboard_text()
            if not raw_text.strip():
                print("[ERROR] 剪贴板为空，请先复制 JD 文本")
                sys.exit(1)
        else:
            raw_text = DEMO_JD
            print("[INFO] 未提供输入参数，使用内置 Demo JD")
            print("提示：可以加上 -c 参数从剪贴板读取，或 -i 交互式输入\n")

        run_test(raw_text, args.url)
        return

    # 循环模式
    print("=" * 60)
    print("  循环测试模式")
    print("=" * 60)
    print("操作方式：")
    print("  1. 复制 JD 文本到剪贴板")
    print("  2. 在本终端按回车执行测试")
    print("  3. 测试完成后，复制新的 JD，再按回车继续")
    print("  4. 输入 q 或 quit 退出")
    print("=" * 60)

    last_text = ""
    while True:
        user_input = input("\n请复制 JD 后按回车开始测试 (q 退出): ").strip().lower()
        if user_input in ("q", "quit", "exit"):
            print("[INFO] 已退出循环测试")
            break

        raw_text = get_clipboard_text()
        if not raw_text.strip():
            print("[WARN] 剪贴板为空，请复制 JD 文本后再试")
            continue

        if raw_text.strip() == last_text.strip():
            print("[WARN] 剪贴板内容未变化，请复制新的 JD 后再试")
            continue

        last_text = raw_text
        print()
        run_test(raw_text, args.url)


if __name__ == "__main__":
    main()
