"""
对话意图 Handler 层 —— 求职雷达 Agent 的业务处理模块

每个 Handler 调用真实 LLM 生成回复。
"""

import json
import logging
from typing import Optional, List, Literal

from pydantic import BaseModel, Field, model_validator

from app.core.intent import IntentResult
from app.core.llm_client import LLMClient
from app.core import state as app_state


logger = logging.getLogger(__name__)


# ──────────────────────────── 请求/响应模型 ────────────────────────────

class ChatAttachment(BaseModel):
    """用户上传的附件元数据"""
    filename: str = Field(..., description="文件名")
    content_type: str = Field(..., description="MIME 类型")
    url: Optional[str] = Field(None, description="文件访问地址（可选）")
    data: Optional[str] = Field(None, description="文件内容 base64 编码（图片/文件直接上传时使用）")


class ChatRequest(BaseModel):
    """统一对话请求模型"""
    session_id: str = Field(..., description="对话会话 ID")
    user_id: Optional[str] = Field(default=None, description="用户唯一标识（用于跨会话长期记忆）")
    message: str = Field(..., description="用户输入的文本内容")
    type: Literal["text", "image", "file"] = Field(default="text", description="消息类型")
    attachments: Optional[List[ChatAttachment]] = Field(default=None, description="附件列表")
    context: Optional[List[dict]] = Field(default=None, description="最近 N 轮对话上下文")
    user_provided_jd: Optional[str] = Field(default=None, description="用户通过图片/附件上传的 JD 文本（OCR提取），存在时跳过知识库检索")
    eval_context: Optional[dict] = Field(default=None, description="评测专用字段（gold_intents/gold_slots/relevance_score）")


class ChatReply(BaseModel):
    """统一对话响应模型"""
    type: Literal["match_report", "global_ranking", "rag_answer", "text"] = Field(..., description="响应类型")
    content: str = Field(..., description="主文本内容")
    data: Optional[dict] = Field(default=None, description="结构化数据")
    sources: Optional[List[str]] = Field(default=None, description="RAG 问答时的引用来源列表")

    # 兼容旧代码中 text 字段的使用
    @model_validator(mode="before")
    @classmethod
    def _accept_text_alias(cls, data):
        if isinstance(data, dict) and "text" in data:
            data = dict(data)
            if "content" not in data:
                data["content"] = data.pop("text")
            if "type" not in data:
                data["type"] = "text"
        return data

    @property
    def text(self) -> str:
        return self.content


# ──────────────────────────── 工具函数 ────────────────────────────

def _get_resume_text() -> str:
    """获取当前生效简历的文本描述"""
    if not app_state.active_resume_id or app_state.active_resume_id not in app_state.resumes_db:
        return "（用户尚未上传简历）"
    item = app_state.resumes_db[app_state.active_resume_id]
    schema = item.get("parsed_schema", {})
    # 优先使用原始文本
    raw = schema.get("meta", {}).get("raw_text", "") if schema.get("meta") else ""
    if raw:
        return raw[:3000]  # 限制长度避免 token 过多
    # 否则拼接结构化信息
    parts = []
    bi = schema.get("basic_info", {})
    if bi:
        parts.append(f"姓名：{bi.get('name', '未知')}，{bi.get('years_exp', '?')}年经验，现任{bi.get('current_company', '?')} {bi.get('current_title', '?')}")
    skills = schema.get("skills", {})
    if skills:
        tech = skills.get("technical", [])
        if tech:
            parts.append(f"技术技能：{', '.join(tech[:15])}")
    exp = schema.get("work_experience", [])
    if exp:
        parts.append("工作经历：")
        for w in exp[:3]:
            parts.append(f"- {w.get('company', '?')} {w.get('title', '?')}：{w.get('description', '')[:100]}")
    return "\n".join(parts) or "（简历信息为空）"


def _get_kb_summary() -> str:
    """获取知识库 JD 摘要"""
    from app.routers.knowledge_base import MOCK_JDS
    if not MOCK_JDS:
        return "（知识库为空）"
    lines = []
    for jd in MOCK_JDS:
        lines.append(f"- {jd['company']} · {jd['title']} | {jd['location']} | {jd['salary']} | {jd['description']}")
    return "\n".join(lines)


# ──────────────────────────── Handler 实现 ────────────────────────────

async def handle_match_single(request: ChatRequest, route_meta: IntentResult) -> ChatReply:
    """单 JD 匹配：分析简历与具体 JD 的契合度"""
    resume_text = _get_resume_text()
    # 优先使用用户通过附件上传的 JD 文本（OCR 提取），否则使用 message
    jd_text = (request.user_provided_jd or request.message).strip()

    if resume_text == "（用户尚未上传简历）":
        return ChatReply(
            type="text",
            content="⚠️ 你还没有上传简历哦！请先点击底部「我的简历」完善信息，我才能帮你做精准匹配。",
            data={"need_resume": True},
        )

    if len(jd_text) < 50:
        return ChatReply(
            type="text",
            content="请粘贴完整的 JD 内容（或上传 JD 截图），我才能帮你分析匹配度。内容越详细，分析越精准 📎",
            data={"need_jd": True},
        )

    system_prompt = """你是一位资深猎头顾问，擅长分析简历与岗位描述的匹配度。
请根据用户简历和 JD 内容，从以下几个维度给出专业分析：
1. 总体匹配分数（0-100）
2. 匹配等级：高度契合 / 基本匹配 / 略有差距 / 不匹配
3. 核心优势（简历中与 JD 要求高度对齐的 2-3 点）
4. 潜在短板（简历中相对 JD 要求不足的 1-2 点）
5. 面试准备建议（预测面试官可能关注的 2 个问题）

请用中文输出，语气专业、客观、有建设性。"""

    user_prompt = f"【简历】\n{resume_text}\n\n【岗位描述】\n{jd_text}\n\n请给出匹配分析："

    try:
        llm = LLMClient.from_config("chat")
        content = await llm.generate(prompt=user_prompt, system=system_prompt, temperature=0.5, max_tokens=1500)
        # 尝试提取分数
        score = 75
        for line in content.split("\n"):
            if "分" in line or "score" in line.lower():
                import re
                nums = re.findall(r"\d+", line)
                if nums:
                    score = min(int(nums[0]), 100)
                    break
        label = "高度契合" if score >= 80 else "基本匹配" if score >= 60 else "略有差距"
        return ChatReply(
            type="match_report",
            content=content,
            data={
                "score": score,
                "label": label,
                "advantage": "详见上方分析",
                "weakness": "详见上方分析",
            },
        )
    except Exception as e:
        logger.error(f"[handle_match_single] LLM error: {e}")
        return ChatReply(
            type="text",
            content=f"❌ 匹配分析失败：{e}",
        )


async def handle_global_match(request: ChatRequest, route_meta: IntentResult) -> ChatReply:
    """全局对比：将简历与知识库中所有 JD 对比，给出排序和推荐"""
    resume_text = _get_resume_text()
    kb_summary = _get_kb_summary()

    if resume_text == "（用户尚未上传简历）":
        return ChatReply(
            type="text",
            content="⚠️ 你还没有上传简历哦！请先点击底部「我的简历」完善信息，我才能帮你做全局对比分析。",
            data={"need_resume": True},
        )

    system_prompt = """你是一位资深猎头顾问。请根据用户简历，对比知识库中的所有岗位，按匹配度从高到低排序。
对每个岗位给出：
1. 匹配分数（0-100）
2. 一句话推荐理由
3. 投递优先级建议

输出格式要求：
1. 总体概述（1-2 句话）
2. 排序列表（每家公司的分析）
3. 投递策略建议

请用中文输出，语气专业、简洁。"""

    user_prompt = f"【简历】\n{resume_text}\n\n【知识库岗位列表】\n{kb_summary}\n\n请给出全局对比分析和投递建议："

    try:
        llm = LLMClient.from_config("chat")
        content = await llm.generate(prompt=user_prompt, system=system_prompt, temperature=0.5, max_tokens=2000)
        # 构造 rankings 数据结构
        from app.routers.knowledge_base import MOCK_JDS
        rankings = []
        for idx, jd in enumerate(MOCK_JDS):
            rankings.append({
                "rank": idx + 1,
                "company": jd["company"],
                "title": jd["title"],
                "score": 85 - idx * 10,
                "reason": f"详见上方分析（{jd['company']} · {jd['title']}）",
            })
        return ChatReply(
            type="global_ranking",
            content=content,
            data={"rankings": rankings},
        )
    except Exception as e:
        logger.error(f"[handle_global_match] LLM error: {e}")
        return ChatReply(
            type="text",
            content=f"❌ 全局对比失败：{e}",
        )


async def handle_rag_qa(request: ChatRequest, route_meta: IntentResult) -> ChatReply:
    """知识库问答：基于知识库内容回答用户问题"""
    kb_summary = _get_kb_summary()
    question = request.message.strip()

    system_prompt = """你是一位求职信息顾问。请基于下方的知识库岗位信息，准确回答用户的问题。
如果知识库中没有相关信息，请明确告知用户"知识库中暂无该信息"，不要编造。
请用中文输出，回答简洁、准确。"""

    user_prompt = f"【知识库信息】\n{kb_summary}\n\n【用户问题】\n{question}\n\n请基于以上信息回答："

    try:
        llm = LLMClient.from_config("chat")
        content = await llm.generate(prompt=user_prompt, system=system_prompt, temperature=0.3, max_tokens=800)
        from app.routers.knowledge_base import MOCK_JDS
        sources = [f"知识库 — {jd['company']} {jd['title']}" for jd in MOCK_JDS]
        return ChatReply(
            type="rag_answer",
            content=content,
            sources=sources[:3],
        )
    except Exception as e:
        logger.error(f"[handle_rag_qa] LLM error: {e}")
        return ChatReply(
            type="text",
            content=f"❌ 问答失败：{e}",
        )


async def handle_general(request: ChatRequest, route_meta: IntentResult) -> ChatReply:
    """通用对话：求职咨询、行业闲聊、打招呼等"""
    question = request.message.strip()
    resume_text = _get_resume_text()

    system_prompt = """你是「求职雷达」AI 助手小橘 🍊，一位专业的求职顾问。
你的职责：
1. 回答求职相关问题（简历优化、面试准备、行业选择、薪资谈判等）
2. 提供有针对性、可落地的建议
3. 语气友好、专业、有温度
4. 如果用户没有上传简历，可以友好提醒上传简历以获得更精准的分析

请用中文输出。"""

    context = f"\n【用户简历摘要】\n{resume_text}" if resume_text != "（用户尚未上传简历）" else ""
    user_prompt = f"【用户输入】\n{question}{context}\n\n请回复："

    try:
        llm = LLMClient.from_config("chat")
        content = await llm.generate(prompt=user_prompt, system=system_prompt, temperature=0.7, max_tokens=1200)
        return ChatReply(
            type="text",
            content=content,
        )
    except Exception as e:
        logger.error(f"[handle_general] LLM error: {e}")
        return ChatReply(
            type="text",
            content=f"❌ 对话失败：{e}",
        )
