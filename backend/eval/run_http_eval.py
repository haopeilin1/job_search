#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HTTP 端到端延迟测试（黑盒评测）

模拟真实前端调用，测量用户感知延迟。
与 frontend/lib/api.ts 和 src/api/client.js 使用完全相同的请求格式。

用法：
    cd backend && python eval/run_http_eval.py                    # 跑全部 55 条
    cd backend && python eval/run_http_eval.py --stream            # 包含 SSE 流式测试
    cd backend && python eval/run_http_eval.py --concurrent 5 10   # 并发压力测试
    cd backend && python eval/run_http_eval.py --case eval_chen_02 # 单条调试
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# httpx 在 venv 中
import httpx

# ──────────────────────────── 配置 ────────────────────────────

EVAL_DIR = Path(__file__).parent
sys.path.insert(0, str(EVAL_DIR.parent))

BASE_URL = "http://localhost:8001"
API_PREFIX = "/api/v1"
CHAT_TIMEOUT = 120.0      # 非流式超时（首次请求可能较长，留足余量）
STREAM_TIMEOUT = 60.0     # SSE 流式超时
HEALTH_TIMEOUT = 5.0      # 健康检查超时
SERVICE_START_WAIT = 30   # 等待服务启动的最大秒数


# ──────────────────────────── 数据模型 ────────────────────────────

@dataclass
class HttpLatencyRecord:
    """单条 HTTP 请求的完整延迟记录"""
    case_id: str
    endpoint: str
    method: str
    is_streaming: bool = False

    # 计时（毫秒）
    total_latency_ms: float = 0.0
    ttfb_ms: float = 0.0          # Time To First Byte

    # SSE 特有
    first_token_ms: float = 0.0
    streaming_duration_ms: float = 0.0
    token_count: int = 0

    # 响应信息
    http_status: int = 0
    response_size_bytes: int = 0
    json_parse_ok: bool = False
    response_body: str = ""       # 完整响应（限 10KB），用于人工校验

    # 前端一致性断言
    schema_valid: bool = False
    schema_errors: List[str] = field(default_factory=list)

    # 错误
    error: str = ""
    timeout_occurred: bool = False


@dataclass
class HttpEvalReport:
    """完整评测报告"""
    timestamp: str
    total_cases: int
    successful_cases: int
    failed_cases: int
    records: List[HttpLatencyRecord] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)
    concurrent_results: List[Dict] = field(default_factory=list)


# ──────────────────────────── 服务管理 ────────────────────────────

async def check_service_health() -> bool:
    """检查后端服务是否已启动"""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{BASE_URL}/docs", timeout=HEALTH_TIMEOUT)
            return resp.status_code == 200
    except Exception:
        return False


async def wait_for_service(max_wait: int = SERVICE_START_WAIT) -> bool:
    """等待服务就绪，每 2 秒检查一次"""
    for i in range(max_wait // 2):
        if await check_service_health():
            return True
        await asyncio.sleep(2)
    return False


def start_backend_server() -> subprocess.Popen:
    """启动后端服务（作为子进程）"""
    backend_dir = EVAL_DIR.parent
    print(f"[HTTP Eval] 启动后端服务: uvicorn app.main:app --host 127.0.0.1 --port 8001")
    
    # Windows 需要特殊处理
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
    
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8001"],
        cwd=str(backend_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **kwargs,
    )
    return proc


async def ensure_backend_running() -> Optional[subprocess.Popen]:
    """确保后端服务在运行，返回子进程（如果是新启动的）"""
    if await check_service_health():
        print("[HTTP Eval] 后端服务已就绪")
        return None
    
    print("[HTTP Eval] 后端服务未启动，正在启动...")
    proc = start_backend_server()
    
    if await wait_for_service():
        print("[HTTP Eval] 后端服务启动成功")
        return proc
    else:
        print("[HTTP Eval] 后端服务启动失败，请手动检查")
        if proc:
            proc.terminate()
        return None


# ──────────────────────────── 简历管理 ────────────────────────────

async def ensure_active_resume(client: httpx.AsyncClient) -> bool:
    """确保有活跃简历。如果没有，激活陈雨桐"""
    try:
        resp = await client.get(f"{BASE_URL}{API_PREFIX}/resumes/current", timeout=10.0)
        if resp.status_code == 200:
            data = resp.json()
            if data and data.get("parsed_schema", {}).get("basic_info", {}).get("name"):
                name = data["parsed_schema"]["basic_info"]["name"]
                print(f"[HTTP Eval] 简历已激活: {name}")
                return True
    except Exception:
        pass
    
    # 获取简历列表并激活第一个
    try:
        resp = await client.get(f"{BASE_URL}{API_PREFIX}/resumes/", timeout=10.0)
        if resp.status_code == 200:
            resumes = resp.json()
            if resumes and len(resumes) > 0:
                resume_id = resumes[0].get("id")
                if resume_id:
                    activate_resp = await client.put(
                        f"{BASE_URL}{API_PREFIX}/resumes/{resume_id}/activate",
                        timeout=10.0
                    )
                    if activate_resp.status_code == 200:
                        print(f"[HTTP Eval] 已自动激活简历: {resume_id}")
                        return True
    except Exception as e:
        print(f"[HTTP Eval] 简历激活失败: {e}")
    
    print("[HTTP Eval] 警告: 无法激活简历，部分测试可能失败")
    return False


# ──────────────────────────── 核心测试 ────────────────────────────

# ──────────────────────────── 前端一致性断言 ────────────────────────────

def validate_chat_response(body: dict) -> List[str]:
    """
    校验响应 JSON 是否符合 frontend/lib/api.ts 中 ChatResponse 的 schema。
    返回错误列表，空列表表示校验通过。
    """
    errors = []
    
    # 顶层必填字段
    if not isinstance(body, dict):
        errors.append("响应体不是 JSON object")
        return errors
    
    # session_id: string
    if "session_id" not in body:
        errors.append("缺少 session_id")
    elif not isinstance(body["session_id"], str):
        errors.append("session_id 类型错误，期望 string")
    
    # intent: string (IntentType)
    if "intent" not in body:
        errors.append("缺少 intent")
    elif not isinstance(body["intent"], str):
        errors.append("intent 类型错误，期望 string")
    
    # route_meta: object
    route_meta = body.get("route_meta")
    if not isinstance(route_meta, dict):
        errors.append("缺少 route_meta 或类型错误")
    else:
        if "intent" not in route_meta:
            errors.append("route_meta 缺少 intent")
        if "confidence" not in route_meta:
            errors.append("route_meta 缺少 confidence")
        if "layer" not in route_meta:
            errors.append("route_meta 缺少 layer")
        else:
            valid_layers = {"rule", "llm", "llm_fallback", "clarification"}
            if route_meta["layer"] not in valid_layers:
                errors.append(f"route_meta.layer 值异常: {route_meta['layer']}")
    
    # reply: object (ChatReply)
    reply = body.get("reply")
    if not isinstance(reply, dict):
        errors.append("缺少 reply 或类型错误")
    else:
        if "type" not in reply:
            errors.append("reply 缺少 type")
        elif not isinstance(reply["type"], str):
            errors.append("reply.type 类型错误")
        else:
            valid_types = {"match_report", "global_ranking", "rag_answer", "text"}
            if reply["type"] not in valid_types:
                errors.append(f"reply.type 值异常: {reply['type']}")
        
        if "content" not in reply:
            errors.append("reply 缺少 content")
        elif not isinstance(reply["content"], str):
            errors.append("reply.content 类型错误，期望 string")
    
    # 可选字段类型校验（不报错，只记录）
    if "memory" in body and not isinstance(body["memory"], dict):
        errors.append("memory 字段类型错误，期望 object")
    
    return errors


async def test_chat_non_stream(
    client: httpx.AsyncClient,
    case: dict,
    session_id: str,
) -> HttpLatencyRecord:
    """测试非流式 chat 端点，模拟 frontend/lib/api.ts 的 sendChatMessage"""
    record = HttpLatencyRecord(
        case_id=case["session_id"],
        endpoint=f"{API_PREFIX}/chat",
        method="POST",
        is_streaming=False,
    )
    
    # 与前端完全一致的请求体
    payload = {
        "session_id": session_id,
        "message": case["message"],
        "type": "text",
        "attachments": None,
        "context": None,
    }
    
    start_ns = time.perf_counter_ns()
    
    try:
        resp = await client.post(
            f"{BASE_URL}{API_PREFIX}/chat",
            json=payload,
            timeout=CHAT_TIMEOUT,
        )
        
        end_ns = time.perf_counter_ns()
        record.total_latency_ms = (end_ns - start_ns) / 1e6
        record.http_status = resp.status_code
        record.response_size_bytes = len(resp.content)
        
        # 保存完整响应（限 10KB）
        body_json = None
        try:
            body_json = resp.json()
            record.json_parse_ok = True
            record.response_body = json.dumps(body_json, ensure_ascii=False)[:10000]
            
            # 提取最终回复文本（用于人工快速浏览）
            reply = body_json.get("reply", {})
            if isinstance(reply, dict):
                record.response_body = reply.get("content", "")[:200] + " | " + record.response_body[:8000]
            
            # ── 前端一致性断言 ──
            schema_errors = validate_chat_response(body_json)
            record.schema_errors = schema_errors
            record.schema_valid = len(schema_errors) == 0
        except Exception:
            record.json_parse_ok = False
            record.schema_valid = False
            record.schema_errors = ["JSON 解析失败"]
            record.response_body = resp.text[:5000]
        
        if resp.status_code != 200:
            record.error = f"HTTP {resp.status_code}"
    
    except httpx.TimeoutException:
        record.timeout_occurred = True
        record.error = "Timeout"
    except Exception as e:
        record.error = str(e)[:200]
    
    return record


async def test_chat_stream(
    client: httpx.AsyncClient,
    case: dict,
    session_id: str,
) -> HttpLatencyRecord:
    """测试 SSE 流式 chat 端点"""
    record = HttpLatencyRecord(
        case_id=case["session_id"],
        endpoint=f"{API_PREFIX}/chat/stream",
        method="POST",
        is_streaming=True,
    )
    
    payload = {
        "session_id": session_id,
        "message": case["message"],
        "type": "text",
        "attachments": None,
        "context": None,
    }
    
    start_ns = time.perf_counter_ns()
    first_token_ns = None
    first_byte_ns = None
    token_count = 0
    chunks = []
    
    try:
        async with client.stream(
            "POST",
            f"{BASE_URL}{API_PREFIX}/chat/stream",
            json=payload,
            timeout=STREAM_TIMEOUT,
        ) as resp:
            # TTFB：收到 HTTP 响应头第一个字节
            first_byte_ns = time.perf_counter_ns()
            record.ttfb_ms = (first_byte_ns - start_ns) / 1e6
            
            async for line in resp.aiter_lines():
                chunks.append(line)
                line = line.strip()
                if not line:
                    continue
                
                # 解析 SSE data: {...}
                if line.startswith("data:"):
                    try:
                        data = json.loads(line[5:].strip())
                        event_type = data.get("event") or data.get("type")
                        
                        # 统计 delta 事件（token 流）
                        if event_type in ("delta",):
                            token_count += 1
                            if first_token_ns is None:
                                first_token_ns = time.perf_counter_ns()
                        
                        # done 事件包含完整响应
                        if event_type in ("done", "complete"):
                            pass
                    except json.JSONDecodeError:
                        pass
            
            end_ns = time.perf_counter_ns()
            record.total_latency_ms = (end_ns - start_ns) / 1e6
            record.first_token_ms = (first_token_ns - start_ns) / 1e6 if first_token_ns else record.ttfb_ms
            record.streaming_duration_ms = (end_ns - first_byte_ns) / 1e6 if first_byte_ns else 0
            record.token_count = token_count
            record.http_status = resp.status_code
            record.response_size_bytes = sum(len(c.encode("utf-8")) for c in chunks)
            record.response_body = "\n".join(chunks)[:10000]
    
    except httpx.TimeoutException:
        record.timeout_occurred = True
        record.error = "Timeout"
    except Exception as e:
        record.error = str(e)[:200]
    
    return record


# ──────────────────────────── 并发压力测试 ────────────────────────────

async def test_concurrent(
    client: httpx.AsyncClient,
    cases: List[dict],
    concurrency: int,
) -> Dict[str, Any]:
    """并发压力测试"""
    sem = asyncio.Semaphore(concurrency)
    
    async def run_one(case: dict):
        async with sem:
            session_id = f"concurrent_{case['session_id']}_{int(time.time()*1000)}"
            return await test_chat_non_stream(client, case, session_id)
    
    overall_start = time.perf_counter_ns()
    records = await asyncio.gather(*[run_one(c) for c in cases])
    overall_ms = (time.perf_counter_ns() - overall_start) / 1e6
    
    ok_records = [r for r in records if not r.error]
    latencies = [r.total_latency_ms for r in ok_records]
    
    return {
        "concurrency": concurrency,
        "total_cases": len(records),
        "successful": len(ok_records),
        "failed": len([r for r in records if r.error]),
        "timeout_count": sum(1 for r in records if r.timeout_occurred),
        "overall_time_ms": round(overall_ms, 2),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "p50_latency_ms": round(sorted(latencies)[len(latencies)//2], 2) if latencies else 0,
        "p99_latency_ms": round(sorted(latencies)[int(len(latencies)*0.99)], 2) if latencies else 0,
        "max_latency_ms": round(max(latencies), 2) if latencies else 0,
        "records": [asdict(r) for r in records],
    }


# ──────────────────────────── 报告生成 ────────────────────────────

def generate_report(records: List[HttpLatencyRecord], concurrent_results: List[Dict] = None) -> HttpEvalReport:
    """生成评测报告"""
    successful = [r for r in records if not r.error]
    failed = [r for r in records if r.error]
    
    latencies = [r.total_latency_ms for r in successful]
    
    # 按 endpoint 分组
    by_endpoint = defaultdict(list)
    stream_latencies = []
    for r in successful:
        by_endpoint[r.endpoint].append(r.total_latency_ms)
        if r.is_streaming:
            stream_latencies.append(r)
    
    # Schema 一致性统计（仅非流式且 HTTP 200 的用例）
    schema_check_records = [r for r in successful if not r.is_streaming]
    schema_valid_count = sum(1 for r in schema_check_records if r.schema_valid)
    schema_invalid_count = len(schema_check_records) - schema_valid_count
    schema_error_examples = []
    for r in schema_check_records:
        if not r.schema_valid and r.schema_errors:
            schema_error_examples.append({
                "case_id": r.case_id,
                "errors": r.schema_errors,
            })
            if len(schema_error_examples) >= 5:
                break
    
    summary = {
        "total_cases": len(records),
        "successful": len(successful),
        "failed": len(failed),
        "timeout_count": sum(1 for r in records if r.timeout_occurred),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "p50_latency_ms": round(sorted(latencies)[len(latencies)//2], 2) if latencies else 0,
        "p99_latency_ms": round(sorted(latencies)[int(len(latencies)*0.99)], 2) if latencies else 0,
        "max_latency_ms": round(max(latencies), 2) if latencies else 0,
        "by_endpoint": {
            ep: {
                "count": len(ls),
                "avg_ms": round(sum(ls)/len(ls), 2),
                "p50_ms": round(sorted(ls)[len(ls)//2], 2),
                "max_ms": round(max(ls), 2),
            }
            for ep, ls in by_endpoint.items()
        },
        "schema_consistency": {
            "checked_count": len(schema_check_records),
            "valid_count": schema_valid_count,
            "invalid_count": schema_invalid_count,
            "valid_rate": round(schema_valid_count / len(schema_check_records), 4) if schema_check_records else 1.0,
            "error_examples": schema_error_examples,
        },
    }
    
    # SSE 流式统计
    if stream_latencies:
        ttfbs = [r.ttfb_ms for r in stream_latencies if r.ttfb_ms > 0]
        first_tokens = [r.first_token_ms for r in stream_latencies if r.first_token_ms > 0]
        summary["streaming"] = {
            "count": len(stream_latencies),
            "avg_ttfb_ms": round(sum(ttfbs)/len(ttfbs), 2) if ttfbs else 0,
            "avg_first_token_ms": round(sum(first_tokens)/len(first_tokens), 2) if first_tokens else 0,
            "avg_token_count": round(sum(r.token_count for r in stream_latencies)/len(stream_latencies), 1),
        }
    
    return HttpEvalReport(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        total_cases=len(records),
        successful_cases=len(successful),
        failed_cases=len(failed),
        records=records,
        summary=summary,
        concurrent_results=concurrent_results,
    )


def print_report(report: HttpEvalReport):
    """控制台输出报告"""
    print("\n" + "=" * 70)
    print("HTTP 端到端延迟测试报告（黑盒评测）")
    print("=" * 70)
    
    s = report.summary
    print(f"\n【总体】")
    print(f"  总用例: {s['total_cases']} | 成功: {s['successful']} | 失败: {s['failed']} | 超时: {s.get('timeout_count', 0)}")
    print(f"  平均延迟: {s['avg_latency_ms']:.0f}ms | P50: {s['p50_latency_ms']:.0f}ms | P99: {s['p99_latency_ms']:.0f}ms | Max: {s['max_latency_ms']:.0f}ms")
    
    print(f"\n【按端点】")
    for ep, stat in s.get("by_endpoint", {}).items():
        print(f"  {ep}: {stat['count']}次 | avg={stat['avg_ms']:.0f}ms | p50={stat['p50_ms']:.0f}ms | max={stat['max_ms']:.0f}ms")
    
    if "streaming" in s:
        st = s["streaming"]
        print(f"\n【SSE 流式】")
        print(f"  测试条数: {st['count']} | avg_TTFB={st['avg_ttfb_ms']:.0f}ms | avg_首Token={st['avg_first_token_ms']:.0f}ms | avg_tokens={st['avg_token_count']}")
    
    # Schema 一致性
    sc = s.get("schema_consistency", {})
    if sc.get("checked_count", 0) > 0:
        print(f"\n【前端一致性断言】")
        print(f"  校验条数: {sc['checked_count']} | 通过: {sc['valid_count']} | 失败: {sc['invalid_count']} | 通过率: {sc['valid_rate']*100:.1f}%")
        if sc.get("error_examples"):
            print(f"  错误样例（前5条）:")
            for ex in sc["error_examples"]:
                print(f"    {ex['case_id']}: {', '.join(ex['errors'])}")
    
    print("=" * 70)


# ──────────────────────────── 数据加载 ────────────────────────────

def load_dataset(batch: Optional[str] = None, case_id: Optional[str] = None) -> List[dict]:
    """加载测试集"""
    dataset_file = EVAL_DIR / "test_dataset.jsonl"
    cases = []
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    
    if case_id:
        cases = [c for c in cases if c["session_id"] == case_id]
    elif batch:
        batches = [b.strip().lower() for b in batch.split(",")]
        cases = [c for c in cases if any(
            c["session_id"].startswith(f"eval_{b}_") or c["session_id"].startswith(f"eval_{b}")
            for b in batches
        )]
    
    return cases


# ──────────────────────────── 主函数 ────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="HTTP 端到端延迟测试（黑盒评测）")
    parser.add_argument("--batch", default=None, help="指定批次，如 chen,li,wang,gen,sup")
    parser.add_argument("--case", default=None, help="指定单条 case_id")
    parser.add_argument("--stream", action="store_true", help="包含 SSE 流式测试（测首token延迟）")
    parser.add_argument("--output", default=None, help="输出 JSON 报告路径")
    parser.add_argument("--no-start-server", action="store_true", help="不自动启动后端（假设已在运行）")
    args = parser.parse_args()
    
    # 加载测试集
    cases = load_dataset(batch=args.batch, case_id=args.case)
    print(f"[HTTP Eval] 加载 {len(cases)} 条测试用例")
    
    # 确保后端运行
    proc = None
    if not args.no_start_server:
        proc = await ensure_backend_running()
        if proc is None and not await check_service_health():
            print("[HTTP Eval] 后端服务不可用，退出")
            return
    elif not await check_service_health():
        print("[HTTP Eval] 后端服务未运行，请启动后重试，或使用 --no-start-server 跳过检查")
        return
    
    try:
        async with httpx.AsyncClient() as client:
            # 确保简历已激活
            await ensure_active_resume(client)
            
            records: List[HttpLatencyRecord] = []
            
            # Session group 管理：同 group 的 case 共享 session_id
            group_session_map: Dict[str, str] = {}
            
            # 1. 非流式 chat 测试（全部用例）
            print("\n[HTTP Eval] === 非流式 Chat 端到端测试 ===")
            for i, case in enumerate(cases, 1):
                sid = case["session_id"]
                msg = case.get("message", "")[:45]
                group = case.get("session_group")
                group_tag = f"[{group}] " if group else ""
                print(f"  [{i}/{len(cases)}] {group_tag}{sid}: {msg}...")
                
                # 多轮 case 共享 session_id
                if group and group in group_session_map:
                    session_id = group_session_map[group]
                    print(f"    (复用 session: {session_id[:30]}...)")
                else:
                    session_id = f"http_{sid}_{int(time.time()*1000)}"
                    if group:
                        group_session_map[group] = session_id
                
                record = await test_chat_non_stream(client, case, session_id)
                records.append(record)
                
                status = "OK" if not record.error else f"ERR({record.error})"
                print(f"    [{status}] {record.total_latency_ms:.0f}ms | HTTP {record.http_status} | {record.response_size_bytes}B")
            
            # 2. SSE 流式测试（选代表性用例，独立 session）
            if args.stream:
                print("\n[HTTP Eval] === SSE 流式 Chat 端到端测试 ===")
                stream_cases = []
                for intent in ["explore", "assess", "verify", "prepare", "chat"]:
                    for c in cases:
                        if intent in c.get("eval_context", {}).get("gold_intents", []):
                            stream_cases.append(c)
                            break
                stream_cases = stream_cases[:5]
                
                for i, case in enumerate(stream_cases, 1):
                    sid = case["session_id"]
                    print(f"  [{i}/{len(stream_cases)}] {sid}: SSE 流式...")
                    
                    session_id = f"stream_{sid}_{int(time.time()*1000)}"
                    record = await test_chat_stream(client, case, session_id)
                    records.append(record)
                    
                    status = "OK" if not record.error else f"ERR({record.error})"
                    print(f"    [{status}] TTFB={record.ttfb_ms:.0f}ms | FirstToken={record.first_token_ms:.0f}ms | "
                          f"Total={record.total_latency_ms:.0f}ms | Tokens={record.token_count}")
            
            # 生成报告
            report = generate_report(records, [])
            
            # 保存
            output_path = Path(args.output) if args.output else EVAL_DIR / f"http_eval_report_{int(time.time())}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(asdict(report), f, ensure_ascii=False, indent=2)
            print(f"\n[HTTP Eval] 详细报告已保存: {output_path}")
            
            # 控制台输出
            print_report(report)
    
    finally:
        if proc:
            proc.terminate()
            print("[HTTP Eval] 后端服务子进程已终止")


if __name__ == "__main__":
    asyncio.run(main())
