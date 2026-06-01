#!/usr/bin/env python3
"""
Judge 后处理脚本 v3：v3.5 Judge 体系
1. 工具调用成功率 / 正确率
2. LLM-as-Judge 多维度任务完成度评估（10维 0-5分制）
3. 规则兜底校验（否决项）
4. Code 辅助判断（不覆盖 LLM 结论，双输出供人工参考）

改进方向：
  A. 评分从 4维0-10分 → 10维0-5分
  B. 覆盖过程全链路（意图/槽位/工具/执行/回复/引用/连贯/语气/效率）
  C. 关键维度（intent_accuracy, response_accuracy, response_completeness）决定通过
  D. 辅助维度（citation_quality, coherence, tone, efficiency）参考不否决
  E. 否决项：严重编造、简历识别错误、检索张冠李戴、空回复、编造引用
  F. Code 判断与 LLM 判断双输出，不覆盖

用法：
    python eval/judge_postprocess.py --run 1
"""

import asyncio
import json
import sys
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.llm_client import LLMClient

logger = logging.getLogger(__name__)

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
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"[Judge] 跳过 malformed JSON 行 {path.name}:{i} | {e}")
                continue
    return items


# 预加载测试集、简历、JD
test_cases = {c["session_id"]: c for c in _load_jsonl(EVAL_DIR / "test_dataset.jsonl")}
resumes = {r["id"]: r for r in _load_json(DATA_DIR / "resumes.json")}
jds = {j["id"]: j for j in _load_json(DATA_DIR / "jds.json")}


# ── 意图标准化映射 ───────────────────────────────────────────

_INTENT_NORMALIZE_MAP = {
    "position_explore": "explore",
    "match_assess": "assess",
    "attribute_verify": "verify",
    "interview_prepare": "prepare",
    "resume_manage": "manage",
    "general_chat": "chat",
}


# ── 上下文构建 ───────────────────────────────────────────────

def _build_case_context(case_id: str, case_result: dict) -> dict:
    """构建 Judge 需要的上下文信息"""
    eval_ctx = case_result.get("eval_context", {})

    resume_id = case_result.get("resume_id", "")
    resume_info = "无简历"
    if resume_id and resume_id in resumes:
        r = resumes[resume_id]
        ps = r.get("parsed_schema", {})
        resume_info = f"姓名：{ps.get('name', '未知')} | 经验：{ps.get('total_years', '未知')}年 | 技能：{', '.join(ps.get('skills', [])[:5])}"

    gold_slots = eval_ctx.get("gold_slots", {})

    # 关联 JD 信息
    jd_info = ""
    company = gold_slots.get("company", "")
    position = gold_slots.get("position", "")
    if company:
        matched_jds = [j for j in jds.values() if j.get("company") == company]
        if matched_jds:
            jd_info = f"知识库中 {company} 的JD：" + "；".join(
                [f"{j['position']}(ID={j['id']})" for j in matched_jds[:3]]
            )
        else:
            jd_info = f"知识库中无 {company} 的JD"

    return {
        "scenario": eval_ctx.get("scenario", ""),
        "notes": eval_ctx.get("notes", ""),
        "expected_tools": eval_ctx.get("expected_tools", []),
        "follow_up_type": eval_ctx.get("follow_up_type", ""),
        "resume_info": resume_info,
        "jd_info": jd_info,
        "gold_slots": gold_slots,
    }


def _needs_rag(case_result: dict) -> bool:
    """判断该 case 是否需要 RAG（kb_retrieve 是预期工具之一）"""
    expected = case_result.get("expected_tools", []) or []
    # 兼容 eval_context 中的 expected_tools
    if not expected:
        eval_ctx = case_result.get("eval_context", {})
        expected = eval_ctx.get("expected_tools", []) or []
    return "kb_retrieve" in expected


def _extract_retrieved_chunks(case_result: dict) -> List[Dict]:
    """从 case_result 中提取 kb_retrieve 实际召回的 chunks"""
    chunks = []
    # 来源1：tool_executions_full（run_eval_v3.py 运行时传入）
    tool_execs = case_result.get("tool_executions_full", []) or case_result.get("tool_details", [])
    for t in tool_execs:
        if isinstance(t, dict) and t.get("tool_name") == "kb_retrieve" or t.get("tool") == "kb_retrieve":
            output = t.get("output", {}) or t.get("result", {})
            if isinstance(output, dict):
                chunks = output.get("chunks", [])
                if chunks:
                    break
    # 来源2：run_eval_v3.py 保存的 case_result 中直接有 kb_chunks
    if not chunks and case_result.get("kb_chunks"):
        chunks = case_result["kb_chunks"]
    return chunks[:15]  # 最多给 judge 看 15 条，防止 prompt 过长


# ── E: 规则兜底校验（否决项）──────────────────────────────────

def _rule_based_check(pred_intents: list, reply: str, gold_intents: list, ctx: dict) -> dict:
    """
    基于规则的快速校验。返回 dict 或 None。
    如果规则明确判定失败，会作为否决项信号传递给 judge。
    """
    reply_lower = (reply or "").lower()
    reply_stripped = (reply or "").strip()
    
    # 0. 空回复 / 极短回复（否决项）
    if len(reply_stripped) < 30:
        return {"resolved": False, "reason": "【规则否决】回复为空或过短(<30字)", "rule_hit": "empty_reply", "veto": True}
    
    # 1. 声称缺少简历但实际应有（否决项）
    no_resume_phrases = ['缺少您的简历', '缺少简历', '没有简历', '未提供简历', '简历信息不足', '请先上传简历', '未检测到简历']
    if any(p in reply for p in no_resume_phrases):
        resume_info = ctx.get("resume_info", "")
        if resume_info and resume_info != "无简历":
            return {"resolved": False, "reason": "【规则否决】系统声称缺少简历，但测试场景已提供简历", "rule_hit": "fake_no_resume", "veto": True}
    
    # 2. VERIFY 意图：回复必须包含具体属性值
    if "verify" in pred_intents or "verify" in gold_intents:
        has_concrete_info = bool(
            re.search(r'\d+[kK\-万]', reply) or
            re.search(r'\d+年', reply) or
            re.search(r'本科|硕士|博士|专科|大专', reply) or
            re.search(r'熟悉|精通|了解|掌握', reply) or
            re.search(r'要求[:：]', reply) or
            re.search(r'具备|需要|必须', reply)
        )
        if "verify" in gold_intents and "verify" not in pred_intents:
            pass
        elif "verify" in pred_intents and not has_concrete_info:
            if len(reply_stripped) < 80:
                return {"resolved": False, "reason": "【规则兜底】VERIFY意图但回复无具体属性值，且回复过短", "rule_hit": "verify_no_value", "veto": False}
    
    # 3. ASSESS 意图：回复应包含匹配分析
    if "assess" in pred_intents or "assess" in gold_intents:
        has_match_analysis = bool(
            re.search(r'匹配|适合|优势|差距|不足|建议|分数|得分|推荐|不适合', reply) or
            re.search(r'经验|技能|要求|符合|不符合', reply)
        )
        if "assess" in gold_intents and "assess" not in pred_intents:
            pass
        elif "assess" in pred_intents and not has_match_analysis:
            if len(reply_stripped) < 100:
                return {"resolved": False, "reason": "【规则兜底】ASSESS意图但回复无匹配分析，且回复过短", "rule_hit": "assess_no_analysis", "veto": False}
    
    # 4. EXPLORE 意图：回复应包含具体岗位推荐
    if "explore" in pred_intents or "explore" in gold_intents:
        has_job_recommendation = bool(
            re.search(r'[\u4e00-\u9fff]+.*(产品经理|工程师|开发|设计|实习)', reply) or
            re.search(r'推荐|岗位|职位|机会|适合', reply)
        )
        if "explore" in gold_intents and "explore" not in pred_intents:
            pass
        elif "explore" in pred_intents and not has_job_recommendation:
            if len(reply_stripped) < 100:
                return {"resolved": False, "reason": "【规则兜底】EXPLORE意图但回复无具体岗位推荐，且回复过短", "rule_hit": "explore_no_jobs", "veto": False}
    
    # 5. PREPARE 意图：回复应包含面试题或准备建议
    if "prepare" in pred_intents or "prepare" in gold_intents:
        has_interview_content = bool(
            re.search(r'面试题|问题|准备|建议|考察|重点|注意', reply) or
            re.search(r'\d+[\.、]', reply)
        )
        if "prepare" in gold_intents and "prepare" not in pred_intents:
            pass
        elif "prepare" in pred_intents and not has_interview_content:
            if len(reply_stripped) < 80:
                return {"resolved": False, "reason": "【规则兜底】PREPARE意图但回复无面试题或准备建议，且回复过短", "rule_hit": "prepare_no_content", "veto": False}
    
    # 6. CHAT 意图：回复应合理、友好
    if pred_intents == ["chat"] or gold_intents == ["chat"]:
        if len(reply_stripped) < 20:
            return {"resolved": False, "reason": "【规则兜底】CHAT意图但回复过短", "rule_hit": "chat_too_short", "veto": False}
        if "error" in reply_lower or "异常" in reply or "失败" in reply:
            return {"resolved": False, "reason": "【规则兜底】CHAT意图但回复包含错误信息", "rule_hit": "chat_has_error", "veto": False}
    
    return None


# ── LLM Judge ─────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """你是AI对话系统评测专家。请基于【标注信息】和【运行过程】，对系统表现进行多维度评估。

## 评分维度（每项 0-5 分）

### 关键维度（决定通过与否）

1. **intent_accuracy（意图识别准确性）**
   - 系统识别的 pred_intents 与 gold_intents 是否一致？
   - 多意图是否全部识别？是否遗漏主要意图？
   - 是否错误触发澄清（gold 不含 clarification 时）？
   - 是否该澄清却没澄清（gold 含 clarification 时）？
   - 0分：意图完全错误；3分：及格（主要意图命中）；5分：完全命中

2. **slot_accuracy（槽位提取准确性）**
   - company/position/attributes 等关键槽位是否正确提取？
   - 是否从上下文（工作记忆）正确补全槽位？
   - 0分：全部错误；3分：主要槽位正确；5分：完全正确

3. **tool_correctness（工具调用正确性）**
   - 是否调用了 expected_tools 中的必要工具？
   - 是否遗漏关键工具（如 assess 场景未调用 match_analyze）？
   - 是否调用了不必要的工具？
   - 0分：工具完全错误；3分：主要工具命中；5分：完全正确

4. **tool_execution（工具执行效果）**
   - 检索工具召回的结果是否相关？（如查"快手"不应只召回"美团"）
   - match_analyze 是否基于简历和JD做了分析？
   - interview_gen 是否针对具体岗位生成面试题？
   - 工具失败后是否有重试/恢复/插入 external_search/模型降级？
   - 0分：执行完全失败；3分：基本可用；5分：完美执行

5. **response_accuracy（回复内容准确性）**
   - 回复是否基于正确的证据？有无编造信息？
   - 有无张冠李戴（如用"某小公司"JD回答"阿里巴巴"问题）？
   - 有无声称"缺少简历"但实际简历已传入？
   - 有无编造学历/经验/技能要求？
   - 0分：完全编造；3分：基本真实；5分：完全准确

6. **response_completeness（回复完整性）**
   - 是否覆盖用户核心需求？
   - VERIFY：是否回答了具体属性？
   - ASSESS：是否给出匹配分析（不是只列要求）？
   - EXPLORE：是否给出岗位列表？
   - PREPARE：是否给出具体面试题/建议？
   - 0分：完全未回答；3分：基本覆盖；5分：完全覆盖

### 辅助维度（参考，不直接否决）

7. **citation_quality（引用标注质量）**
   - 检索类任务（VERIFY/ASSESS/EXPLORE）是否有引用标注？
   - 标注是否准确对应实际来源？
   - **请特别注意判断：系统是否编造了引用标记**（如标注[1][2]但实际无对应来源）？
   - 非检索类任务此项可给满分
   - 0分：无标注或编造；3分：有标注但不够精确；5分：标注准确完整

8. **coherence（连贯性）**
   - 多轮对话中是否保持上下文连贯？
   - 是否正确处理指代消解（如"上面那个岗""第一个推荐"）？
   - 0分：完全脱节；3分：基本连贯；5分：完美连贯

9. **tone（语气）**
   - 是否专业、友好、自然？
   - 是否过于机械/模板化（如"根据您的问题，我将从以下几个方面进行分析"）？
   - 是否像真人顾问在对话？
   - 0分：机械/不友好；3分：尚可；5分：自然专业有亲和力

10. **efficiency（效率）**
    - 延迟是否合理？
    - Token 消耗是否合理？
    - 是否有多余的 LLM 调用或重试？
    - 0分：极慢/浪费；3分：正常；5分：高效

### RAG 专属维度（仅检索类任务评判，非检索类任务可给满分）

11. **faithfulness（忠实度）**
    - 系统回复中的每个事实声明，是否都能在【检索到的证据 chunks】中找到支撑？
    - **核心原则**：不看"是否答全"，只看"是否说假话"。即使检索结果缺少某些信息，只要回复没有编造未在证据中出现的内容，忠实度就应高。
    - 如果检索结果为空或极少，系统回复"抱歉，未找到相关信息"或"根据现有信息无法确定"——这是高忠实度的表现。
    - 如果系统回复中包含了检索证据中没有的信息（如捏造薪资数字、编造不存在的技能要求），这是低忠实度。
    - 0分：大量编造，事实声明大多无证据支撑；3分：基本忠实，偶有小幅推测；5分：完全忠实，每个事实都有证据支撑

12. **answer_relevance（答案相关性）**
    - 系统回复是否直接回答了用户的 query？是否答非所问？
    - 用户问"字节算法岗的薪资"，回复却讲字节跳动的公司文化——这是低相关性。
    - 用户问"我匹配吗"，回复给出了匹配分析和差距——这是高相关性。
    - 与忠实度不同：此项只看"回答是否对准问题"，不看"回答是否有证据支撑"。
    - 0分：完全答非所问；3分：基本相关但有所偏离；5分：精准回答用户问题

## 否决项（任一触发，resolved 强制为 false，无视其他分数）

1. response_accuracy ≤ 1：严重编造或事实错误
2. 声称"缺少简历"但实际简历已传入
3. 检索到完全错误的公司/岗位并作为回答依据
4. 编造引用标记
5. 工具执行异常导致空回复或极短回复（<30字）

## 通过标准

**必须同时满足：**
- intent_accuracy ≥ 3
- response_accuracy ≥ 3
- response_completeness ≥ 3

**否决项任一触发 → 不通过**

## 输出格式

请严格按 JSON 输出，不要有任何其他内容：
{
  "scores": {
    "intent_accuracy": 0-5,
    "slot_accuracy": 0-5,
    "tool_correctness": 0-5,
    "tool_execution": 0-5,
    "response_accuracy": 0-5,
    "response_completeness": 0-5,
    "citation_quality": 0-5,
    "coherence": 0-5,
    "tone": 0-5,
    "efficiency": 0-5,
    "faithfulness": 0-5,
    "answer_relevance": 0-5
  },
  "resolved": true/false,
  "reason": "详细评价理由，说明通过/不通过的原因，引用具体证据"
}"""


async def judge_single_case(case_result: dict, ctx: dict) -> dict:
    """对单条结果调用 Judge 模型评估任务完成度（v3 多维度）"""
    case_id = case_result["case_id"]
    message = case_result["message"]
    gold_intents = case_result.get("gold_intents", [])
    reply = case_result.get("reply", "")
    tools_called = case_result.get("tools_called", [])
    
    # 兼容旧版结果文件
    raw_pred_intents = case_result.get("pred_intents", [])
    if not raw_pred_intents and case_result.get("pred_intent"):
        normalized = _INTENT_NORMALIZE_MAP.get(case_result["pred_intent"], case_result["pred_intent"])
        raw_pred_intents = [normalized]
    pred_intents = [p for p in raw_pred_intents if p]
    
    # E: 规则兜底校验
    rule_result = _rule_based_check(pred_intents, reply, gold_intents, ctx)
    
    # 判断是否需要 RAG，提取检索到的 chunks
    needs_rag = _needs_rag(case_result)
    retrieved_chunks = _extract_retrieved_chunks(case_result) if needs_rag else []
    
    # A: 保守化默认 — 空回复直接判 False
    if not reply or len(reply.strip()) < 30:
        return {
            "resolved": False,
            "scores": {k: 0 for k in [
                "intent_accuracy", "slot_accuracy", "tool_correctness", "tool_execution",
                "response_accuracy", "response_completeness", "citation_quality",
                "coherence", "tone", "efficiency", "faithfulness", "answer_relevance"
            ]},
            "reason": "【规则否决】回复为空或过短(<30字)",
            "case_id": case_id,
            "rule_hit": "empty_reply",
            "veto": True,
            "needs_rag": needs_rag,
        }
    
    # 构建 enriched prompt（传入更多信息）
    rag_section = ""
    if needs_rag and retrieved_chunks:
        chunk_texts = []
        for i, ch in enumerate(retrieved_chunks, 1):
            content = ch.get("content", "") if isinstance(ch, dict) else str(ch)
            meta = ch.get("metadata", {}) if isinstance(ch, dict) else {}
            cid = ch.get("chunk_id", f"chunk_{i}") if isinstance(ch, dict) else f"chunk_{i}"
            src = f"[{meta.get('company','')}/{meta.get('position','')}/{meta.get('section','')}]" if meta else ""
            chunk_texts.append(f"[证据{i}] ID={cid} {src}\n{content[:400]}")
        chunks_joined = "\n\n".join(chunk_texts)
        rag_section = f"""
【检索到的证据 chunks】（共{len(retrieved_chunks)}条，用于评判 faithfulness）
{chunks_joined}

【注意】请基于上述【检索到的证据 chunks】评判 faithfulness（忠实度）。
如果系统回复中的事实声明在上述证据中找不到支撑，则 faithfulness 应低分。
如果检索结果为空/极少，系统回复"未找到"或"无法确定"是合理的，faithfulness 应高分。
"""
    elif needs_rag and not retrieved_chunks:
        rag_section = """
【检索到的证据 chunks】（空——系统未召回任何 chunk）

【注意】检索结果为空。若系统回复"未找到相关信息"或"知识库中无相关内容"，
则 faithfulness 应给高分（系统没有编造）；若系统却给出了具体信息，则 faithfulness 应低分。
"""
    
    user_prompt = f"""【用户消息】{message}

【期望意图(gold)】{gold_intents}
【期望槽位(gold_slots)】{json.dumps(ctx.get('gold_slots', {}), ensure_ascii=False)}
【期望工具(expected_tools)】{ctx.get('expected_tools', [])}
【系统识别意图(pred)】{pred_intents}
【实际调用工具】{tools_called}

【测试场景】{ctx.get('scenario', '')}
【测试备注】{ctx.get('notes', '')}
【跟进类型】{ctx.get('follow_up_type', '')}

【简历信息】
{ctx.get('resume_info', '无')}

【相关JD信息】
{ctx.get('jd_info', '无')}
{rag_section}
【系统回复】
{reply[:6000]}

请严格按照系统提示中的JSON格式输出评分结果。"""

    try:
        llm = LLMClient.from_config("judge")
        raw = await llm.generate(
            prompt=user_prompt,
            system=JUDGE_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=1200,
            timeout=60.0,
        )

        # 尝试提取 JSON
        raw_stripped = raw.strip()
        
        if not raw_stripped:
            return _judge_failure(case_id, "Judge模型返回空")

        if "```json" in raw_stripped:
            json_part = raw_stripped.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_stripped:
            json_part = raw_stripped.split("```")[1].split("```")[0].strip()
        else:
            start = raw_stripped.find("{")
            end = raw_stripped.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_part = raw_stripped[start:end+1]
            else:
                json_part = raw_stripped

        data = json.loads(json_part)
        scores = data.get("scores", {})
        
        # 确保所有维度都有值
        default_scores = {
            "intent_accuracy": 0, "slot_accuracy": 0, "tool_correctness": 0,
            "tool_execution": 0, "response_accuracy": 0, "response_completeness": 0,
            "citation_quality": 0, "coherence": 0, "tone": 0, "efficiency": 0,
            "faithfulness": 0, "answer_relevance": 0,
        }
        for k in default_scores:
            if k not in scores:
                scores[k] = default_scores[k]
        
        # 非 RAG 任务：faithfulness 和 answer_relevance 给满分（不参与拉低）
        if not needs_rag:
            scores["faithfulness"] = 5
            scores["answer_relevance"] = 5
        
        # E: 如果规则明确判定失败（否决项），强制 resolved=False
        if rule_result and rule_result.get("veto"):
            data["resolved"] = False
            data["reason"] = f"{rule_result['reason']}。原LLM理由：{data.get('reason', '')[:100]}"
        
        return {
            "resolved": data.get("resolved", False),
            "scores": scores,
            "reason": data.get("reason", ""),
            "case_id": case_id,
            "rule_hit": rule_result["rule_hit"] if rule_result else None,
            "veto": rule_result.get("veto", False) if rule_result else False,
            "needs_rag": needs_rag,
            "judge_prompt": user_prompt,
            "raw_output": raw,
        }
        
    except Exception as e:
        return _judge_failure(case_id, f"解析异常: {e}", needs_rag=needs_rag)


def _judge_failure(case_id: str, reason: str, needs_rag: bool = False) -> dict:
    """Judge 调用/解析失败时的保守默认"""
    scores = {k: 0 for k in [
        "intent_accuracy", "slot_accuracy", "tool_correctness", "tool_execution",
        "response_accuracy", "response_completeness", "citation_quality",
        "coherence", "tone", "efficiency", "faithfulness", "answer_relevance"
    ]}
    # 非 RAG 任务：新维度给满分（不参与拉低）
    if not needs_rag:
        scores["faithfulness"] = 5
        scores["answer_relevance"] = 5
    return {
        "resolved": False,
        "scores": scores,
        "reason": f"【Judge故障】{reason}，基于保守原则判定不通过",
        "case_id": case_id,
        "rule_hit": None,
        "veto": False,
        "needs_rag": needs_rag,
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

    case_files = sorted([f for f in run_dir.iterdir() if f.suffix == ".json" and not f.name.startswith("_")])
    print(f"【Judge 后处理 v3】run{run_idx} | 找到 {len(case_files)} 条结果")

    all_results = []
    for cf in case_files:
        with open(cf, "r", encoding="utf-8") as f:
            all_results.append(json.load(f))

    print("  计算工具指标...")
    for r in all_results:
        tool_metrics = compute_tool_metrics(r)
        r.update(tool_metrics)

    print(f"  调用 Judge 评估 {len(all_results)} 条结果...")
    judge_results = []
    for idx, r in enumerate(all_results, 1):
        ctx = _build_case_context(r["case_id"], r)
        jr = await judge_single_case(r, ctx)
        
        scores = jr.get("scores", {})
        r["judge_resolved"] = jr["resolved"]
        r["judge_scores"] = scores
        r["judge_reason"] = jr["reason"]
        r["judge_rule_hit"] = jr.get("rule_hit")
        r["judge_veto"] = jr.get("veto", False)
        judge_results.append(jr)
        print(f"    [{idx}/{len(all_results)}] {r['case_id']} → resolved={jr['resolved']}, scores={scores}")

    # 保存结果
    output_path = run_dir / "_report_judge.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "cases": all_results,
            "summary": _compute_summary(judge_results),
        }, f, ensure_ascii=False, indent=2)
    print(f"  结果已保存: {output_path}")


def _compute_summary(judge_results: list) -> dict:
    """计算汇总统计"""
    total = len(judge_results)
    if not total:
        return {}
    
    resolved_count = sum(1 for j in judge_results if j["resolved"])
    veto_count = sum(1 for j in judge_results if j.get("veto"))
    
    dim_avg = {}
    dim_keys = [
        "intent_accuracy", "slot_accuracy", "tool_correctness", "tool_execution",
        "response_accuracy", "response_completeness", "citation_quality",
        "coherence", "tone", "efficiency", "faithfulness", "answer_relevance"
    ]
    for k in dim_keys:
        vals = [j.get("scores", {}).get(k, 0) for j in judge_results]
        dim_avg[k] = round(sum(vals) / len(vals), 2) if vals else 0
    
    # RAG 子集统计
    rag_cases = [j for j in judge_results if j.get("needs_rag")]
    non_rag_cases = [j for j in judge_results if not j.get("needs_rag")]
    rag_dim_avg = {}
    if rag_cases:
        for k in ["faithfulness", "answer_relevance"]:
            vals = [j.get("scores", {}).get(k, 0) for j in rag_cases]
            rag_dim_avg[k] = round(sum(vals) / len(vals), 2) if vals else 0
    
    return {
        "total_cases": total,
        "resolved_count": resolved_count,
        "resolved_rate": round(resolved_count / total, 2),
        "veto_count": veto_count,
        "dimension_averages": dim_avg,
        "rag_cases": {
            "count": len(rag_cases),
            "dimension_averages": rag_dim_avg,
        },
        "non_rag_cases": {
            "count": len(non_rag_cases),
        },
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, default=1, help="评测轮次编号")
    args = parser.parse_args()
    asyncio.run(process_run(args.run))
