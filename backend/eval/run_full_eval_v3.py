"""
全量流式测试 v3：三轮稳定性测试 + 动态 Judge + 流式输出

功能：
1. 三轮循环，每轮 55 条 case
2. 流式 SSE 请求，模拟真实用户体验
3. 每轮计算意图正确率、工具 F1、Judge 完成度
4. 三轮结束后对比稳定性
5. 失败自动重试，保证 55 条全部执行

输出：
- eval/results/round_{1,2,3}.json
- eval/results/stability_report.md
- eval/results/run.log
"""
import asyncio
import json
import time
import sys
import traceback
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.llm_client import LLMClient
from app.core.config import settings

# ── 配置 ──
BASE_URL = "http://127.0.0.1:8001"
CHAT_URL = f"{BASE_URL}/api/v1/chat/stream"
TIMEOUT_TOTAL = 600.0
RETRY_MAX = 1
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# 加载测试集
with open(Path(__file__).resolve().parent / "test_dataset.jsonl", "r", encoding="utf-8") as f:
    TEST_CASES = [json.loads(line) for line in f]

with open(Path(__file__).resolve().parent.parent / "data" / "resumes.json", "r", encoding="utf-8") as f:
    RESUMES = json.load(f)


def get_resume_text(resume_id: str) -> str:
    for r in RESUMES:
        if r["id"] == resume_id:
            return r.get("parsed_schema", {}).get("meta", {}).get("raw_text", "")
    return ""


def build_judge_criteria(gold_intents: list) -> str:
    """根据 gold_intents 动态生成 Judge 评价标准"""
    criteria_parts = []
    if "clarification" in gold_intents:
        criteria_parts.append(
            "- 对于【澄清】意图：系统是否识别到信息缺失，并提出了合理的澄清问题引导用户补充信息？"
            "如果系统直接猜测并回答了，反而应判为未完成。"
        )
    if "explore" in gold_intents:
        criteria_parts.append(
            "- 对于【探索】意图：系统是否推荐了与简历/查询匹配的岗位，并给出了有依据的推荐理由？"
            "如果只是泛泛而谈没有具体岗位，判为未完成。"
        )
    if "assess" in gold_intents:
        criteria_parts.append(
            "- 对于【评估】意图：系统是否分析了简历与指定 JD 的匹配度，指出了匹配点和差距？"
        )
    if "verify" in gold_intents:
        criteria_parts.append(
            "- 对于【核实】意图：系统是否基于知识库证据回答了属性问题，并引用了来源？"
            "如果只是猜测没有证据，判为未完成。"
        )
    if "prepare" in gold_intents:
        criteria_parts.append(
            "- 对于【准备】意图：系统是否生成了有针对性的面试题，且题目与匹配 gap 相关？"
        )
    if "chat" in gold_intents:
        criteria_parts.append(
            "- 对于【聊天】意图：系统是否给出了完整、准确、有帮助的回答？"
        )
    
    if not criteria_parts:
        criteria_parts = ["- 系统是否直接回答了用户的问题，回答完整且无事实错误？"]
    
    return "\n".join(criteria_parts)


async def llm_judge(case: dict, reply: str, tools_used: list) -> dict:
    """调用 JUDGE 模型判断任务完成度"""
    try:
        criteria = build_judge_criteria(case["eval_context"]["gold_intents"])
        system_prompt = f"""你是一位严格的测试评估员。请判断以下系统回复是否完成了用户的真实需求。

【评价标准】
{criteria}

【通用规则】
- resolved=true：满足上述对应意图的成功标准
- resolved=false：偏题、信息缺失、事实错误、或未达到对应意图的成功标准

请输出严格 JSON：
{{"resolved": true/false, "reason": "评价理由"}}"""

        user_prompt = (
            f"【用户消息】\n{case['message']}\n\n"
            f"【期望意图】{case['eval_context']['gold_intents']}\n\n"
            f"【实际调用工具】{tools_used}\n\n"
            f"【系统回复】\n{reply[:2000]}\n\n"
            f"请输出JSON："
        )
        
        llm = LLMClient.from_config("judge")
        raw = await llm.generate(
            prompt=user_prompt,
            system=system_prompt,
            temperature=0.1,
            max_tokens=500,
            timeout=60.0,
        )
        data = json.loads(raw.strip())
        return {
            "resolved": data.get("resolved", False),
            "reason": data.get("reason", "")[:200],
        }
    except Exception as e:
        return {"resolved": False, "reason": f"Judge 调用失败: {str(e)[:100]}"}


async def stream_chat(
    message: str,
    resume_id: str,
    session_id: str,
    session_group: str = None,
) -> dict:
    """
    发送流式请求，收集完整响应。
    返回: {success, reply, intent, tools, latency, ttfb, error}
    """
    payload = {
        "message": message,
        "resume_id": resume_id,
        "session_id": session_id,
        "stream": True,
    }
    if session_group:
        payload["session_group"] = session_group
    
    t0 = time.time()
    ttfb = None
    chunks = []
    
    try:
        # 每个请求使用独立的 session，避免 SSE 长连接影响连接池
        async with aiohttp.ClientSession() as session:
            async with session.post(CHAT_URL, json=payload, timeout=aiohttp.ClientTimeout(total=TIMEOUT_TOTAL)) as resp:
                ttfb = time.time() - t0
                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            chunks.append(data)
                            # 流式端点的 done 事件包含 intent，读到后主动退出
                            if "intent" in data:
                                break
                        except:
                            pass
            
            latency = time.time() - t0
        
        # 从最后一个非空 chunk 提取结构化信息
        last_chunk = None
        for c in reversed(chunks):
            if c.get("type") in ("final", "result") or "intent" in c:
                last_chunk = c
                break
        
        if not last_chunk:
            # 如果没有结构化 chunk，拼接所有 text chunk
            reply_text = "".join(c.get("content", "") for c in chunks if c.get("type") == "text")
            return {
                "success": True,
                "reply": reply_text,
                "intent": None,
                "tools": [],
                "latency": latency,
                "ttfb": ttfb,
            }
        
        # 流式端点 done 事件中 reply 是 {"text": "..."} 对象
        reply_raw = last_chunk.get("reply", last_chunk.get("content", ""))
        if isinstance(reply_raw, dict):
            reply_text = reply_raw.get("text", str(reply_raw))
        else:
            reply_text = reply_raw
        
        # 从 route_meta 中提取 tools 信息
        route_meta = last_chunk.get("route_meta", {})
        tools = last_chunk.get("tools", [])
        if not tools and "agent" in last_chunk:
            tools = last_chunk["agent"].get("tools", [])
        
        return {
            "success": True,
            "reply": reply_text,
            "intent": last_chunk.get("intent"),
            "tools": tools,
            "route_meta": route_meta,
            "latency": latency,
            "ttfb": ttfb,
        }
    except Exception as e:
        return {
            "success": False,
            "reply": "",
            "intent": None,
            "tools": [],
            "latency": time.time() - t0,
            "ttfb": ttfb,
            "error": str(e)[:200],
        }


def calc_intent_f1(pred: list, gold: list) -> float:
    """多意图 F1"""
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    pred_set = set(pred)
    gold_set = set(gold)
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0


def calc_tool_f1(pred: list, gold: list) -> float:
    """工具选择 F1"""
    if not gold:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    pred_set = set(pred)
    gold_set = set(gold)
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0


async def run_single_case(
    case: dict,
    case_idx: int,
    round_idx: int,
    group_session_map: dict,
) -> dict:
    """执行单条 case，失败时重试 1 次"""
    sid = case["session_id"]
    resume_id = case["resume_id"]
    msg = case["message"]
    gold_intents = case["eval_context"]["gold_intents"]
    expected_tools = case["eval_context"].get("expected_tools", [])
    gold_slots = case["eval_context"].get("gold_slots", {})
    session_group = case.get("session_group")
    
    # 多轮会话：如果属于某个 group，复用 session_id
    actual_session_id = sid
    if session_group and session_group in group_session_map:
        actual_session_id = group_session_map[session_group]
    
    for attempt in range(RETRY_MAX + 1):
        result = await stream_chat(msg, resume_id, actual_session_id, session_group)
        
        if result["success"]:
            # 保存 group_session_map
            if session_group:
                group_session_map[session_group] = actual_session_id
            break
        else:
            if attempt < RETRY_MAX:
                await asyncio.sleep(2)
            else:
                result["error"] = f"重试{RETRY_MAX+1}次后仍失败: {result.get('error', '')}"
    
    # 计算指标
    pred_intents = [result["intent"]] if result["intent"] else []
    intent_f1 = calc_intent_f1(pred_intents, gold_intents)
    tool_f1 = calc_tool_f1(result.get("tools", []), expected_tools)
    
    # 判断意图是否完全正确（多意图需全部匹配）
    intent_exact = set(pred_intents) == set(gold_intents)
    
    # Judge
    judge_result = await llm_judge(case, result.get("reply", ""), result.get("tools", []))
    
    record = {
        "session_id": sid,
        "message": msg,
        "round": round_idx,
        "attempt": attempt + 1,
        "gold_intents": gold_intents,
        "pred_intent": result.get("intent"),
        "intent_exact": intent_exact,
        "intent_f1": round(intent_f1, 3),
        "gold_tools": expected_tools,
        "pred_tools": result.get("tools", []),
        "tool_f1": round(tool_f1, 3),
        "reply_length": len(result.get("reply", "")),
        "latency": round(result.get("latency", 0), 2),
        "ttfb": round(result.get("ttfb", 0), 2) if result.get("ttfb") else None,
        "judge_resolved": judge_result["resolved"],
        "judge_reason": judge_result["reason"],
        "error": result.get("error"),
    }
    
    # 实时日志
    status = "✅" if result["success"] else "❌"
    marker = "LLM" if judge_result["resolved"] else "未"
    print(
        f"[R{round_idx}] [{case_idx+1:02d}/55] {status} {sid} | "
        f"intent={result.get('intent','?')}({intent_f1:.1f}) tools={tool_f1:.1f} "
        f"judge={marker} | {result.get('latency',0):.1f}s"
    )
    
    return record


async def run_round(round_idx: int) -> list:
    """执行一轮 55 条 case"""
    print(f"\n{'='*60}", flush=True)
    print(f"【Round {round_idx}/3】开始执行 {len(TEST_CASES)} 条 case")
    print(f"{'='*60}\n", flush=True)
    
    results = []
    group_session_map = {}
    
    for i, case in enumerate(TEST_CASES):
        t0 = time.time()
        record = await run_single_case(case, i, round_idx, group_session_map)
        results.append(record)
        
        # 进度统计
        elapsed = time.time() - t0
        done = len(results)
        success = sum(1 for r in results if not r.get("error"))
        intent_ok = sum(1 for r in results if r.get("intent_exact"))
        judge_ok = sum(1 for r in results if r.get("judge_resolved"))
        
        if (i + 1) % 10 == 0 or i == len(TEST_CASES) - 1:
            print(f"  >> 进度: {done}/55 | 成功:{success} | 意图正确:{intent_ok} | Judge通过:{judge_ok} | 本轮耗时:{elapsed:.1f}s", flush=True)
    
    # 保存本轮结果
    out_path = RESULTS_DIR / f"round_{round_idx}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Round {round_idx} 完成，结果已保存: {out_path}", flush=True)
    
    return results


def analyze_stability(all_results: list) -> str:
    """分析三轮稳定性，生成 Markdown 报告"""
    # all_results: [round1_results, round2_results, round3_results]
    report_lines = [
        "# 全量流式测试稳定性报告",
        f"\n生成时间: {datetime.now().isoformat()}",
        f"总 case 数: {len(TEST_CASES)}",
        f"测试轮数: 3",
        "",
    ]
    
    # 每轮汇总
    report_lines.append("## 每轮指标汇总\n")
    report_lines.append("| Round | 意图正确率 | 意图 F1 | 工具 F1 | Judge 通过率 | 平均耗时 | 失败数 |")
    report_lines.append("|-------|-----------|---------|---------|-------------|---------|--------|")
    
    for round_idx, results in enumerate(all_results, 1):
        total = len(results)
        intent_ok = sum(1 for r in results if r.get("intent_exact"))
        intent_f1s = [r["intent_f1"] for r in results]
        tool_f1s = [r["tool_f1"] for r in results]
        judge_ok = sum(1 for r in results if r.get("judge_resolved"))
        latencies = [r["latency"] for r in results if r.get("latency")]
        errors = sum(1 for r in results if r.get("error"))
        
        report_lines.append(
            f"| {round_idx} | {intent_ok}/{total} ({intent_ok/total*100:.1f}%) | "
            f"{sum(intent_f1s)/len(intent_f1s):.3f} | {sum(tool_f1s)/len(tool_f1s):.3f} | "
            f"{judge_ok}/{total} ({judge_ok/total*100:.1f}%) | {sum(latencies)/len(latencies):.1f}s | {errors} |"
        )
    
    # 不稳定的 case
    report_lines.append("\n## 不稳定 Case（三轮结果不一致）\n")
    unstable = []
    
    for i in range(len(TEST_CASES)):
        sid = TEST_CASES[i]["session_id"]
        intents = [all_results[r][i].get("intent_exact") for r in range(3)]
        judges = [all_results[r][i].get("judge_resolved") for r in range(3)]
        
        # 如果意图或 judge 结果在三轮中不一致
        if len(set(intents)) > 1 or len(set(judges)) > 1:
            unstable.append({
                "session_id": sid,
                "message": TEST_CASES[i]["message"],
                "intents": intents,
                "judges": judges,
            })
    
    if unstable:
        report_lines.append(f"共发现 {len(unstable)} 条不稳定 case:\n")
        for u in unstable:
            intent_markers = ["✅" if x else "❌" for x in u["intents"]]
            judge_markers = ["✅" if x else "❌" for x in u["judges"]]
            report_lines.append(
                f"- `{u['session_id']}`: {u['message'][:40]}...\n"
                f"  - 意图正确: R1{intent_markers[0]} R2{intent_markers[1]} R3{intent_markers[2]}\n"
                f"  - Judge通过: R1{judge_markers[0]} R2{judge_markers[1]} R3{judge_markers[2]}"
            )
    else:
        report_lines.append("🎉 所有 case 三轮结果完全一致，系统高度稳定！\n")
    
    # 耗时分析
    report_lines.append("\n## 耗时分析\n")
    for round_idx, results in enumerate(all_results, 1):
        latencies = [r["latency"] for r in results if r.get("latency")]
        if latencies:
            report_lines.append(
                f"- Round {round_idx}: 平均 {sum(latencies)/len(latencies):.1f}s, "
                f"最快 {min(latencies):.1f}s, 最慢 {max(latencies):.1f}s"
            )
    
    # 失败 case
    report_lines.append("\n## 失败 Case（HTTP 错误或超时）\n")
    failed = []
    for round_idx, results in enumerate(all_results, 1):
        for r in results:
            if r.get("error"):
                failed.append((round_idx, r["session_id"], r["error"]))
    
    if failed:
        for rnd, sid, err in failed:
            report_lines.append(f"- Round {rnd} `{sid}`: {err[:100]}")
    else:
        report_lines.append("无失败 case\n")
    
    report_lines.append("\n---\n报告生成完毕")
    
    return "\n".join(report_lines)


async def main():
    print(f"全量流式测试 v3 启动", flush=True)
    print(f"测试集: {len(TEST_CASES)} 条")
    print(f"后端: {BASE_URL}", flush=True)
    print(f"结果目录: {RESULTS_DIR}", flush=True)
    print(f"开始时间: {datetime.now().isoformat()}\n")
    
    all_results = []
    
    for round_idx in range(1, 4):
        results = await run_round(round_idx)
        all_results.append(results)
        
        if round_idx < 3:
            print(f"\n⏳ Round {round_idx} 完成，休息 10 秒后继续下一轮...\n", flush=True)
            await asyncio.sleep(10)
    
    # 生成稳定性报告
    report = analyze_stability(all_results)
    report_path = RESULTS_DIR / "stability_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\n{'='*60}", flush=True)
    print("【全部完成】三轮测试 + 稳定性分析", flush=True)
    print(f"报告: {report_path}", flush=True)
    print(f"结束时间: {datetime.now().isoformat()}")
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
