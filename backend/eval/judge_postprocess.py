#!/usr/bin/env python3
"""
Judge 后处理脚本 v2：对已保存的评测结果补充
1. 工具调用成功率 / 正确率
2. LLM-as-Judge 多维度任务完成度评估（accuracy / completeness / citation / relevance / resolved）
3. 规则兜底校验

改进方向：
  A. 模型故障/返回空时默认 resolved=False（保守化）
  C. 多维度评分标准
  D. User Prompt 中传入 pred_intents + 测试集 notes + 简历/JD 上下文
  E. 规则兜底校验（VERIFY 查属性、ASSESS 查匹配分析等）

用法：
    python eval/judge_postprocess.py --run 1
"""

import asyncio
import json
import sys
import re
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.llm_client import LLMClient

RESULTS_DIR = Path(__file__).resolve().parent / "results"
EVAL_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ── 加载静态资源 ─────────────────────────────────────────────

def _load_json(path: Path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path):
    if not path.exists():
        return []
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# 全局缓存，只加载一次
_TEST_CASES = None
_RESUMES = None
_JDS = None

# pred_intent 字符串到标准意图名的映射（兼容旧版结果文件）
_INTENT_NORMALIZE_MAP = {
    "position_explore": "explore",
    "match_assess": "assess",
    "interview_prepare": "prepare",
    "general_chat": "chat",
    "attribute_verify": "verify",
    "resume_manage": "manage",
    "explore": "explore",
    "assess": "assess",
    "prepare": "prepare",
    "chat": "chat",
    "verify": "verify",
    "manage": "manage",
    "clarification": "clarification",
}


def get_test_cases():
    global _TEST_CASES
    if _TEST_CASES is None:
        _TEST_CASES = {c["session_id"]: c for c in _load_jsonl(EVAL_DIR / "test_dataset.jsonl")}
    return _TEST_CASES


def get_resumes():
    global _RESUMES
    if _RESUMES is None:
        data = _load_json(DATA_DIR / "resumes.json")
        if isinstance(data, list):
            _RESUMES = {r["id"]: r for r in data}
        elif isinstance(data, dict):
            _RESUMES = data.get("resumes", {})
            if isinstance(_RESUMES, list):
                _RESUMES = {r["id"]: r for r in _RESUMES}
        else:
            _RESUMES = {}
    return _RESUMES


def get_jds():
    global _JDS
    if _JDS is None:
        _JDS = {jd.get("id"): jd for jd in _load_json(DATA_DIR / "jds.json")}
    return _JDS


def _extract_resume_summary(resume: dict) -> str:
    """提取简历关键信息，供 judge 参考"""
    if not resume:
        return "无简历"
    schema = resume.get("parsed_schema", {})
    basic = schema.get("basic_info", {})
    name = basic.get("name", "未知")
    years = basic.get("years_exp", "未知")
    title = basic.get("current_title", "未知")
    skills = schema.get("skills", {})
    if isinstance(skills, dict):
        tech = skills.get("technical", [])
        soft = skills.get("soft", [])
        skill_list = tech + soft
    else:
        skill_list = skills[:20] if isinstance(skills, list) else []
    
    edu_list = schema.get("education", [])
    edu_str = ""
    if edu_list:
        e = edu_list[0]
        edu_str = f"，学历：{e.get('school', '')} {e.get('degree', '')} {e.get('major', '')}"
    
    return f"姓名：{name}，经验：{years}年，当前职位：{title}{edu_str}，技能：{', '.join(skill_list[:15])}"


def _extract_jd_summary(jd: dict) -> str:
    """提取 JD 关键信息"""
    if not jd:
        return "无JD信息"
    company = jd.get("company", "")
    position = jd.get("position", "")
    salary = jd.get("salary", "")
    raw = jd.get("raw_text", "")
    # 提取硬性要求（如果有 JSON 结构）
    hard_reqs = []
    soft_reqs = []
    try:
        # 尝试从 raw_text 中提取 JSON
        if raw.strip().startswith("{"):
            parsed = json.loads(raw)
            hard_reqs = parsed.get("hard_requirements", [])
            soft_reqs = parsed.get("soft_requirements", [])
    except Exception:
        pass
    
    req_str = ""
    if hard_reqs:
        req_str += f"\n硬性要求：{'；'.join(hard_reqs)}"
    if soft_reqs:
        req_str += f"\n软性要求：{'；'.join(soft_reqs)}"
    
    return f"公司：{company}，职位：{position}，薪资：{salary}{req_str}\n原文摘要：{raw[:500]}"


def _find_jd_by_company_position(company: str, position: str, jds: dict) -> dict:
    """根据公司名和职位名查找 JD"""
    best_match = None
    best_score = 0
    for jd in jds.values():
        c = jd.get("company", "")
        p = jd.get("position", "")
        score = 0
        if company and company in c:
            score += 3
        if position and position in p:
            score += 3
        if company and c in company:
            score += 1
        if position and p in position:
            score += 1
        if score > best_score:
            best_score = score
            best_match = jd
    return best_match


def _build_case_context(case_id: str, case_result: dict) -> dict:
    """为 judge 构建完整的 case 上下文"""
    test_cases = get_test_cases()
    resumes = get_resumes()
    jds = get_jds()
    
    case = test_cases.get(case_id, {})
    eval_ctx = case.get("eval_context", {})
    
    # resume 信息
    resume_id = case.get("resume_id", "")
    resume = resumes.get(resume_id)
    resume_info = _extract_resume_summary(resume) if resume else "无简历"
    
    # JD 信息：从 gold_slots 或 notes 中提取
    gold_slots = eval_ctx.get("gold_slots", {})
    jd_info_list = []
    
    # 尝试从 company + position 匹配
    companies = []
    positions = []
    if gold_slots.get("company"):
        companies.append(gold_slots["company"])
    if gold_slots.get("companies"):
        companies.extend(gold_slots["companies"])
    if gold_slots.get("position"):
        positions.append(gold_slots["position"])
    if gold_slots.get("positions"):
        positions.extend(gold_slots["positions"])
    
    # 去重配对
    seen = set()
    for c in companies:
        for p in positions:
            key = f"{c}|{p}"
            if key not in seen:
                seen.add(key)
                jd = _find_jd_by_company_position(c, p, jds)
                if jd:
                    jd_info_list.append(_extract_jd_summary(jd))
    
    # 如果只有 company 没有 position，或反之
    if not jd_info_list and companies:
        for c in companies:
            jd = _find_jd_by_company_position(c, "", jds)
            if jd:
                jd_info_list.append(_extract_jd_summary(jd))
    if not jd_info_list and positions:
        for p in positions:
            jd = _find_jd_by_company_position("", p, jds)
            if jd:
                jd_info_list.append(_extract_jd_summary(jd))
    
    jd_info = "\n---\n".join(jd_info_list) if jd_info_list else "知识库中无精确匹配的JD"
    
    return {
        "notes": eval_ctx.get("notes", ""),
        "scenario": eval_ctx.get("scenario", ""),
        "expected_tools": eval_ctx.get("expected_tools", []),
        "follow_up_type": eval_ctx.get("follow_up_type", ""),
        "resume_info": resume_info,
        "jd_info": jd_info,
        "gold_slots": gold_slots,
    }


# ── E: 规则兜底校验 ───────────────────────────────────────────

def _rule_based_check(pred_intents: list, reply: str, gold_intents: list, ctx: dict) -> dict:
    """
    基于规则的快速校验。返回 dict 或 None。
    如果规则明确判定失败，会作为强先验信号传递给 judge。
    """
    reply_lower = (reply or "").lower()
    reply_stripped = (reply or "").strip()
    
    # 0. 空回复
    if len(reply_stripped) < 30:
        return {"resolved": False, "reason": "【规则兜底】回复为空或过短(<30字)", "rule_hit": "empty_reply"}
    
    # 1. VERIFY 意图：回复必须包含具体属性值，不能只是"需要更多信息"
    if "verify" in pred_intents or "verify" in gold_intents:
        # 检查是否包含具体数值/信息（薪资、年限、学历等）
        has_concrete_info = bool(
            re.search(r'\d+[kK\-万]', reply) or  # 薪资数字
            re.search(r'\d+年', reply) or        # 年限
            re.search(r'本科|硕士|博士|专科|大专', reply) or  # 学历
            re.search(r'熟悉|精通|了解|掌握', reply) or       # 技能
            re.search(r'要求[:：]', reply) or
            re.search(r'具备|需要|必须', reply)
        )
        # 如果 pred 中没有 verify 但 gold 中有，且回复明显缺少信息
        if "verify" in gold_intents and "verify" not in pred_intents:
            # 意图漏识，但不直接判死，留给 LLM 判断
            pass
        elif "verify" in pred_intents and not has_concrete_info:
            if len(reply_stripped) < 80:
                return {"resolved": False, "reason": "【规则兜底】VERIFY意图但回复无具体属性值，且回复过短", "rule_hit": "verify_no_value"}
    
    # 2. ASSESS 意图：回复应包含匹配分析
    if "assess" in pred_intents or "assess" in gold_intents:
        has_match_analysis = bool(
            re.search(r'匹配|适合|优势|差距|不足|建议|分数|得分|推荐|不适合', reply) or
            re.search(r'经验|技能|要求|符合|不符合', reply)
        )
        if "assess" in gold_intents and "assess" not in pred_intents:
            pass  # 意图漏识，留给 LLM
        elif "assess" in pred_intents and not has_match_analysis:
            if len(reply_stripped) < 100:
                return {"resolved": False, "reason": "【规则兜底】ASSESS意图但回复无匹配分析，且回复过短", "rule_hit": "assess_no_analysis"}
    
    # 3. EXPLORE 意图：回复应包含具体岗位推荐
    if "explore" in pred_intents or "explore" in gold_intents:
        has_job_recommendation = bool(
            re.search(r'[\u4e00-\u9fff]+.*(产品经理|工程师|开发|设计|实习)', reply) or
            re.search(r'推荐|岗位|职位|机会|适合', reply)
        )
        if "explore" in gold_intents and "explore" not in pred_intents:
            pass
        elif "explore" in pred_intents and not has_job_recommendation:
            if len(reply_stripped) < 100:
                return {"resolved": False, "reason": "【规则兜底】EXPLORE意图但回复无具体岗位推荐，且回复过短", "rule_hit": "explore_no_jobs"}
    
    # 4. PREPARE 意图：回复应包含面试题或准备建议
    if "prepare" in pred_intents or "prepare" in gold_intents:
        has_interview_content = bool(
            re.search(r'面试题|问题|准备|建议|考察|重点|注意', reply) or
            re.search(r'\d+[\.、]', reply)  # 列表形式
        )
        if "prepare" in gold_intents and "prepare" not in pred_intents:
            pass
        elif "prepare" in pred_intents and not has_interview_content:
            if len(reply_stripped) < 80:
                return {"resolved": False, "reason": "【规则兜底】PREPARE意图但回复无面试题或准备建议，且回复过短", "rule_hit": "prepare_no_content"}
    
    # 5. CHAT 意图：回复应合理、友好
    if pred_intents == ["chat"] or gold_intents == ["chat"]:
        if len(reply_stripped) < 20:
            return {"resolved": False, "reason": "【规则兜底】CHAT意图但回复过短", "rule_hit": "chat_too_short"}
        # 检查是否是错误信息
        if "error" in reply_lower or "异常" in reply or "失败" in reply:
            return {"resolved": False, "reason": "【规则兜底】CHAT意图但回复包含错误信息", "rule_hit": "chat_has_error"}
    
    return None  # 规则未命中，走 LLM Judge


# ── LLM Judge ─────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """你是AI对话系统评测专家。请基于以下多维度标准评估系统回复质量，评分标准已放宽，重点关注核心任务是否完成。

【评估维度】（每项0-10分）
1. accuracy（准确性）：
   - 意图识别是否与 gold_intents 一致？只要主要意图命中即可，不必苛求完全一致
   - 回复内容是否基于正确的意图生成？
   - 若意图识别错误，accuracy 可适当扣分，但不必压到 ≤4 分

2. completeness（完整性）：
   - 回复是否覆盖了 gold_intents 对应的核心需求？不必面面俱到，主要意图处理了即可
   - 对于 VERIFY：是否回答了用户询问的具体属性（如薪资、学历、经验要求）？
   - 对于 ASSESS：是否给出了匹配度分析或关键匹配/不匹配点？不必逐项详细对比
   - 对于 EXPLORE：是否给出了具体岗位列表？不要求每条都有详细推荐理由
   - 对于 PREPARE：是否提供了面试题或针对性准备建议？质量合格即可
   - 对于 CHAT/边界场景：只要回复合理、友好即可给高分

3. citation（引用/证据质量）：
   - 对于 VERIFY：是否基于 JD/知识库信息回答？不苛求逐条引用原文
   - 对于 ASSESS：是否基于简历和 JD 做对比？不苛求逐项罗列
   - 对于 EXPLORE/CHAT/PREPARE：此项可给高分，不卡 citation
   - 引用的信息是否大致准确即可

4. relevance（相关性）：
   - 回复是否直接回应了用户的问题？
   - 有无明显答非所问、跑题？

5. resolved（综合判定 true/false）：
   - 整体上，系统是否成功完成了用户的核心请求？
   - 核心标准：accuracy≥4、completeness≥4、relevance≥4 时即可考虑为 true
   - 若意图识别严重错误且导致回复完全偏离，可为 false
   - 若回复包含明显错误信息或系统异常，可为 false
   - 边界场景（如乱输入、测试边界条件）：只要回复合理即可判 true

【重要参考：测试标注说明】
- 请重点参考【测试备注】中的 expectations，那里说明了该 case 期望的回复内容
- 如果系统回复满足了【测试备注】中的核心期望，completeness 和 resolved 应给高分
- 对于标注 notes 中明确说明是"边界测试"的场景，放宽要求，只要系统没有崩溃或严重异常即可

【特殊场景评分指引】
- 知识库中不存在用户查询的 JD：若系统正确告知"未找到"或给出了合理替代，completeness 可给高分；若系统编造信息，accuracy 和 citation 给低分
- 多意图场景：若完成了主要意图（占主导地位的意图），completeness 不应大幅扣分
- 澄清场景：若系统正确触发澄清，resolved 为 true
- 空结果场景（如搜索无匹配岗位）：若系统合理说明原因或给出替代建议，resolved 为 true
- 边界/异常输入场景：如用户输入乱码、无意义文本，只要系统给出合理回复（如引导用户重新提问），resolved 为 true

输出严格JSON，不要添加任何额外文本：
{
  "accuracy": 0-10,
  "completeness": 0-10,
  "citation": 0-10,
  "relevance": 0-10,
  "resolved": true/false,
  "reason": "详细评价理由，引用具体证据"
}"""


async def judge_single_case(case_result: dict, ctx: dict) -> dict:
    """对单条结果调用 Judge 模型评估任务完成度（v2 多维度）"""
    case_id = case_result["case_id"]
    message = case_result["message"]
    gold_intents = case_result.get("gold_intents", [])
    reply = case_result.get("reply", "")
    tools_called = case_result.get("tools_called", [])
    
    # 兼容旧版结果文件（pred_intent 为字符串）
    raw_pred_intents = case_result.get("pred_intents", [])
    if not raw_pred_intents and case_result.get("pred_intent"):
        normalized = _INTENT_NORMALIZE_MAP.get(case_result["pred_intent"], case_result["pred_intent"])
        raw_pred_intents = [normalized]
    pred_intents = [p for p in raw_pred_intents if p]
    
    # E: 规则兜底校验
    rule_result = _rule_based_check(pred_intents, reply, gold_intents, ctx)
    
    # A: 保守化默认 — 空回复直接判 False
    if not reply or len(reply.strip()) < 30:
        return {
            "resolved": False,
            "accuracy": 0,
            "completeness": 0,
            "citation": 0,
            "relevance": 0,
            "reason": "【规则兜底】回复为空或过短(<30字)",
            "case_id": case_id,
            "rule_hit": "empty_reply",
        }
    
    # 构建 enriched prompt
    user_prompt = f"""【用户消息】{message}

【期望意图(gold)】{gold_intents}
【系统识别意图(pred)】{pred_intents}
【实际调用工具】{tools_called}

【测试场景】{ctx.get('scenario', '')}
【测试备注】{ctx.get('notes', '')}
【跟进类型】{ctx.get('follow_up_type', '')}

【简历信息】
{ctx.get('resume_info', '无')}

【相关JD信息】
{ctx.get('jd_info', '无')}

【系统回复】
{reply[:6000]}

请严格按照系统提示中的JSON格式输出评分结果。"""

    try:
        llm = LLMClient.from_config("judge")
        raw = await llm.generate(
            prompt=user_prompt,
            system=JUDGE_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=800,
            timeout=60.0,
        )

        # 尝试提取 JSON
        raw_stripped = raw.strip()
        
        # A: 保守化默认 — Judge 返回空时判 False
        if not raw_stripped:
            return {
                "resolved": False,
                "accuracy": 0,
                "completeness": 0,
                "citation": 0,
                "relevance": 0,
                "reason": "【Judge故障】Judge模型返回空，基于保守原则判定不通过",
                "case_id": case_id,
                "rule_hit": None,
            }

        if "```json" in raw_stripped:
            json_part = raw_stripped.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_stripped:
            json_part = raw_stripped.split("```")[1].split("```")[0].strip()
        else:
            # 尝试找到 JSON 对象边界
            start = raw_stripped.find("{")
            end = raw_stripped.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_part = raw_stripped[start:end+1]
            else:
                json_part = raw_stripped

        data = json.loads(json_part)
        
        # E: 如果规则明确判定失败，但 LLM 判 true，需要 override
        if rule_result and data.get("resolved", False):
            # 只有当 LLM 没有给出充分理由时才 override
            reason = data.get("reason", "")
            if len(reason) < 30 or "规则" not in reason:
                data["resolved"] = False
                data["reason"] = f"{rule_result['reason']}；LLM原判true但被规则override。原理由：{reason[:100]}"
                data["rule_override"] = True
        
        return {
            "resolved": data.get("resolved", False),
            "accuracy": data.get("accuracy", 0),
            "completeness": data.get("completeness", 0),
            "citation": data.get("citation", 0),
            "relevance": data.get("relevance", 0),
            "reason": data.get("reason", ""),
            "case_id": case_id,
            "rule_hit": rule_result["rule_hit"] if rule_result else None,
        }
        
    except Exception as e:
        # A: 保守化默认 — 解析异常时判 False
        return {
            "resolved": False,
            "accuracy": 0,
            "completeness": 0,
            "citation": 0,
            "relevance": 0,
            "reason": f"【Judge故障】解析异常: {e}，基于保守原则判定不通过",
            "case_id": case_id,
            "rule_hit": None,
        }


# ── 工具指标计算（保持不变）────────────────────────────────────

def compute_tool_metrics(case_result: dict) -> dict:
    """计算工具调用成功率和正确率"""
    pred_tools = case_result.get("pred_tools", [])
    expected_tools = case_result.get("expected_tools", [])

    if pred_tools:
        success_tools = [t for t in pred_tools if t.get("status") == "✅"]
        tool_execution_success_rate = len(success_tools) / len(pred_tools)
    else:
        success_tools = []
        tool_execution_success_rate = 0.0

    if expected_tools:
        correct_count = 0
        for et in expected_tools:
            matched = any(
                et in t.get("tool", "") and t.get("status") == "✅"
                for t in pred_tools
            )
            if matched:
                correct_count += 1
        tool_correct_rate = correct_count / len(expected_tools)
    else:
        tool_correct_rate = 1.0

    return {
        "tool_execution_success_rate": round(tool_execution_success_rate, 2),
        "tool_correct_rate": round(tool_correct_rate, 2),
        "tools_total_called": len(pred_tools),
        "tools_success_called": len(success_tools),
    }


# ── 主流程 ────────────────────────────────────────────────────

async def process_run(run_idx: int):
    """处理单轮结果"""
    run_dir = RESULTS_DIR / f"run{run_idx}"
    if not run_dir.exists():
        print(f"[ERROR] run{run_idx} 目录不存在")
        return

    # 读取所有 case 结果文件
    case_files = sorted([f for f in run_dir.iterdir() if f.suffix == ".json" and not f.name.startswith("_")])
    print(f"【Judge 后处理 v2】run{run_idx} | 找到 {len(case_files)} 条结果")

    all_results = []
    for cf in case_files:
        with open(cf, "r", encoding="utf-8") as f:
            all_results.append(json.load(f))

    # 计算工具指标
    print("  计算工具指标...")
    for r in all_results:
        tool_metrics = compute_tool_metrics(r)
        r.update(tool_metrics)

    # Judge 评估（v2 多维度）
    print(f"  调用 Judge 评估 {len(all_results)} 条结果...")
    judge_results = []
    for idx, r in enumerate(all_results, 1):
        ctx = _build_case_context(r["case_id"], r)
        jr = await judge_single_case(r, ctx)
        
        # 将多维度评分写回结果
        r["judge_resolved"] = jr["resolved"]
        r["judge_accuracy"] = jr["accuracy"]
        r["judge_completeness"] = jr["completeness"]
        r["judge_citation"] = jr["citation"]
        r["judge_relevance"] = jr["relevance"]
        r["judge_reason"] = jr["reason"]
        r["judge_rule_hit"] = jr.get("rule_hit")
        r["judge_rule_override"] = jr.get("rule_override", False)
        
        judge_results.append(jr)
        status = "PASS" if jr["resolved"] else "FAIL"
        dims = f"A={jr['accuracy']} C={jr['completeness']} CIT={jr['citation']} R={jr['relevance']}"
        print(f"    [{idx}/{len(all_results)}] {r['case_id']}: {status} | {dims} | {jr['reason'][:50]}...")

    # 保存更新后的结果
    for r in all_results:
        case_file = run_dir / f"{r['case_id']}.json"
        with open(case_file, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)

    # 计算汇总指标
    total = len(all_results)
    success_results = [r for r in all_results if r["status"] == "success"]

    judge_resolved = sum(1 for r in all_results if r.get("judge_resolved"))
    judge_rate = round(judge_resolved / total, 3) if total else 0

    avg_accuracy = round(sum(r.get("judge_accuracy", 0) for r in all_results) / total, 2) if total else 0
    avg_completeness = round(sum(r.get("judge_completeness", 0) for r in all_results) / total, 2) if total else 0
    avg_citation = round(sum(r.get("judge_citation", 0) for r in all_results) / total, 2) if total else 0
    avg_relevance = round(sum(r.get("judge_relevance", 0) for r in all_results) / total, 2) if total else 0

    avg_tool_success = round(sum(r["tool_execution_success_rate"] for r in all_results) / total, 3) if total else 0
    avg_tool_correct = round(sum(r["tool_correct_rate"] for r in all_results) / total, 3) if total else 0

    report = {
        "run": run_idx,
        "total_cases": total,
        "success_cases": len(success_results),
        "success_rate": round(len(success_results) / total, 3) if total else 0,
        "intent_match_rate": round(sum(1 for r in all_results if r.get("intent_match")) / len(success_results), 3) if success_results else 0,
        "avg_tool_match_rate": round(sum(r["tool_match_rate"] for r in all_results) / total, 3) if total else 0,
        "tool_execution_success_rate": avg_tool_success,
        "tool_correct_rate": avg_tool_correct,
        "judge_resolved_rate": judge_rate,
        "judge_avg_accuracy": avg_accuracy,
        "judge_avg_completeness": avg_completeness,
        "judge_avg_citation": avg_citation,
        "judge_avg_relevance": avg_relevance,
        "reply_completion_rate": round(sum(1 for r in all_results if r.get("has_reply")) / total, 3) if total else 0,
        "avg_total_latency_sec": round(sum(r["total_latency"] for r in all_results if r.get("total_latency")) / len(success_results), 2) if success_results else 0,
        "judge_breakdown": [
            {
                "case_id": r["case_id"],
                "resolved": r.get("judge_resolved"),
                "accuracy": r.get("judge_accuracy"),
                "completeness": r.get("judge_completeness"),
                "citation": r.get("judge_citation"),
                "relevance": r.get("judge_relevance"),
                "reason": r.get("judge_reason", "")[:120],
                "rule_hit": r.get("judge_rule_hit"),
                "rule_override": r.get("judge_rule_override"),
            }
            for r in all_results
        ],
        "timestamp": datetime.now().isoformat(),
    }

    with open(run_dir / "_report_judge.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"【Judge 后处理 v2 完成】run{run_idx}")
    print(f"{'='*70}")
    print(f"  总用例: {total}")
    print(f"  请求成功率: {report['success_rate']}")
    print(f"  意图匹配率: {report['intent_match_rate']}")
    print(f"  工具平均匹配率: {report['avg_tool_match_rate']}")
    print(f"  工具执行成功率: {report['tool_execution_success_rate']}")
    print(f"  工具调用正确率: {report['tool_correct_rate']}")
    print(f"  ------------------------------")
    print(f"  Judge 任务完成率: {report['judge_resolved_rate']}")
    print(f"  Judge 平均 accuracy: {report['judge_avg_accuracy']}")
    print(f"  Judge 平均 completeness: {report['judge_avg_completeness']}")
    print(f"  Judge 平均 citation: {report['judge_avg_citation']}")
    print(f"  Judge 平均 relevance: {report['judge_avg_relevance']}")
    print(f"  回复完成率: {report['reply_completion_rate']}")
    print(f"  报告保存: {run_dir / '_report_judge.json'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, default=1, help="处理第几轮结果")
    args = parser.parse_args()
    asyncio.run(process_run(args.run))
