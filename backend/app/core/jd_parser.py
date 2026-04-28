"""
JD 结构化解析器 —— 使用 LLM 将原始 JD 文本解析为结构化 Schema

设计决策：
1. 解析与存储分离：parser 只负责 raw_text -> JDSchema，不关心存储
2. 使用用户配置的 chat LLM 进行解析，降低配置成本
3. Prompt 经过反复调优，要求 LLM 输出严格 JSON，避免解析失败
4. 解析失败时返回基础 fallback，不阻断用户流程
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Optional

from app.core.llm_client import LLMClient

logger = logging.getLogger(__name__)


# ──────────────────────────── Prompt 模板 ────────────────────────────

_JD_PARSE_PROMPT = """你是一位专业的 JD（岗位描述）结构化提取专家。

任务：将下方原始 JD 文本解析为严格的 JSON 格式，提取关键字段。

【输出 JSON 格式要求】
{
  "company": "公司名，如 ByteDance",
  "position": "岗位名，如 AI平台产品经理",
  "location": "工作地点，如 北京/上海",
  "salary_range": "薪资范围，如 30k-60k 或 面议",
  "sections": {
    "responsibilities": "岗位职责的完整原文",
    "hard_requirements": ["硬性要求1", "硬性要求2", ...],
    "soft_requirements": ["软性要求/加分项1", "软性要求/加分项2", ...]
  },
  "keywords": ["关键词1", "关键词2", ...],
  "structured_summary": {
    "min_years": 3,
    "max_years": null,
    "min_education": "本科",
    "category": "技术",
    "domain": "AI"
  }
}

【字段说明】
- company：从文本中提取公司名，如果没有明确写公司名，根据上下文推断
- position：岗位名称
- location：工作地点，如果没有写则返回 null
- salary_range：薪资范围，如果没有写则返回 null
- sections.responsibilities：岗位职责的完整段落原文
- sections.hard_requirements：硬性要求，逐条提取为数组。注意：
  * 必须逐条拆分，不要合并成一句话
  * 每条是一个独立的字符串
  * 如果 JD 中没有明确区分硬性和软性，把"必须""要求""具备"等强条件放入 hard_requirements
- sections.soft_requirements：软性要求/加分项，逐条提取为数组。注意：
  * 把"优先""加分""熟悉""了解"等弱条件放入 soft_requirements
  * 如果没有明确区分，可以返回空数组 []
- keywords：从全文中提取 5-15 个核心关键词，包括技术栈、业务领域、岗位特性等
- structured_summary：从全文中提取结构化摘要，用于快速粗筛匹配：
  * min_years：最低工作年限要求，如"3年以上"→3，"5年+"→5，未提及则 null
  * max_years：最高工作年限上限，如"5年以下"→5，未提及则 null
  * min_education：最低学历要求，如"本科及以上"→"本科"，"硕士优先"→"硕士"，未提及则 null
  * category：岗位大类，限定为 ["技术", "产品", "运营", "设计", "市场", "销售", "职能", "其他"]
  * domain：业务领域，如"AI", "电商", "金融", "教育", "游戏", "SaaS", "自动驾驶", "医疗"等

【约束】
1. 必须返回合法 JSON，不要包含 markdown 代码块标记
2. 不要添加任何解释性文字，只输出 JSON
3. 如果某个字段无法从文本中提取，使用 null 或空数组 []，不要编造
4. hard_requirements 和 soft_requirements 必须拆分为数组，禁止合并成长字符串"""


# ──────────────────────────── 解析器类 ────────────────────────────

class JDParser:
    """
    JD 结构化解析器。

    使用 LLM 将原始文本解析为结构化 JDSchema。
    解析失败时自动降级，返回基础 fallback 结构。
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client

    async def parse(
        self,
        raw_text: str,
        source_type: str = "paste",
    ) -> dict:
        """
        解析原始 JD 文本为结构化 Schema。

        Args:
            raw_text: 用户粘贴或 OCR 后的原始文本
            source_type: 来源类型（paste/image/pdf）

        Returns:
            符合 JDSchema 的字典
        """
        if not raw_text or not raw_text.strip():
            return self._fallback(raw_text, source_type, reason="empty_input")

        # 尝试 LLM 解析
        if self.llm_client is not None:
            try:
                parsed = await self._llm_parse(raw_text)
                if parsed:
                    result = self._enrich(parsed, raw_text, source_type)
                    logger.info(f"[JDParser] LLM parse SUCCESS → company={result.get('company')} position={result.get('position')} keywords={result.get('keywords')}")
                    logger.info(f"[JDParser] LLM parsed schema:\n{json.dumps(result, ensure_ascii=False, indent=2)}")
                    return result
            except Exception as e:
                logger.warning(f"[JDParser] LLM parse failed: {e}, falling back to rule-based")
        else:
            logger.warning("[JDParser] llm_client is None, using rule-based fallback")

        # LLM 不可用或失败，使用规则 fallback
        result = self._rule_based_parse(raw_text, source_type)
        logger.info(f"[JDParser] RULE-BASED parse → company={result.get('company')} position={result.get('position')} keywords={result.get('keywords')}")
        logger.info(f"[JDParser] RULE-BASED parsed schema:\n{json.dumps(result, ensure_ascii=False, indent=2)}")
        return result

    async def _llm_parse(self, raw_text: str) -> Optional[dict]:
        """调用 LLM 进行结构化解析"""
        full_prompt = f"{_JD_PARSE_PROMPT}\n\n【原始 JD 文本】\n{raw_text}\n\n请输出 JSON："

        raw = await self.llm_client.generate(
            prompt=full_prompt,
            system="你是一个专业的 JD 结构化提取专家，只输出合法 JSON，不添加任何解释。",
            temperature=0.3,
            max_tokens=2048,
        )

        logger.info(f"[JDParser] LLM raw response:\n{raw}")

        parsed = self._safe_parse_json(raw)
        if parsed is None:
            logger.warning("[JDParser] LLM returned non-JSON, retrying with json_mode")
            # 二次尝试，启用 json_mode（如果模型支持）
            try:
                raw2 = await self.llm_client.chat(
                    messages=[
                        {"role": "system", "content": "你是一个 JD 结构化提取专家，只输出合法 JSON。"},
                        {"role": "user", "content": full_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=2048,
                )
                parsed = self._safe_parse_json(raw2)
            except Exception:
                pass

        return parsed

    def _rule_based_parse(self, raw_text: str, source_type: str) -> dict:
        """
        规则兜底解析（LLM 不可用时）。
        基于简单规则提取信息，准确度远低于 LLM，但至少能跑通流程。
        """
        import re
        text = raw_text.strip()
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        # 公司名
        company = "未知公司"
        for line in lines[:5]:
            if any(kw in line for kw in ["公司", "科技", "集团", "网络", "信息"]):
                company = line.replace("【", "").replace("】", "").replace("*", "").strip()[:20]
                break
        if company == "未知公司" and lines:
            company = lines[0][:20]

        # 岗位名
        position = "未知岗位"
        for line in lines[:10]:
            if any(kw in line for kw in ["工程师", "产品经理", "产品", "经理", "运营", "开发", "算法", "设计", "测试", "架构师"]):
                position = line.replace("【", "").replace("】", "").replace("*", "").strip()[:30]
                break

        # 地点
        location = None
        cities = ["北京", "上海", "深圳", "杭州", "广州", "成都", "武汉", "西安", "南京", "苏州"]
        for city in cities:
            if city in text:
                location = city
                break

        # 薪资
        salary_range = None
        m = re.search(r'(\d+)[kK]-(\d+)[kK]', text)
        if m:
            salary_range = f"{m.group(1)}k-{m.group(2)}k"

        # 职责、要求
        responsibilities = text[:800] if len(text) <= 800 else text[:800] + "..."
        hard_requirements = []
        soft_requirements = []

        req_lines = [l.strip("-• ").strip() for l in lines if l.strip() and len(l.strip()) > 5]
        hard_requirements = req_lines[1:4] if len(req_lines) > 1 else ["详见职位描述"]
        soft_requirements = req_lines[4:7] if len(req_lines) > 4 else []

        # 关键词
        keywords = []
        tech_kws = ["Python", "Java", "Go", "C++", "JavaScript", "React", "Vue", "Node.js",
                    "MySQL", "Redis", "MongoDB", "Kafka", "Docker", "Kubernetes",
                    "AI", "大模型", "LLM", "NLP", "机器学习", "深度学习", "算法",
                    "产品经理", "数据分析", "用户研究", "运营", "增长"]
        for kw in tech_kws:
            if kw in text:
                keywords.append(kw)
        if not keywords:
            keywords = ["AI"]

        return self._enrich(
            {
                "company": company,
                "position": position,
                "location": location,
                "salary_range": salary_range,
                "sections": {
                    "responsibilities": responsibilities,
                    "hard_requirements": hard_requirements,
                    "soft_requirements": soft_requirements,
                },
                "keywords": keywords[:10],
                "structured_summary": {},
            },
            raw_text,
            source_type,
        )

    def _enrich(self, parsed: dict, raw_text: str, source_type: str) -> dict:
        """补充系统字段，生成完整 JDSchema"""
        now = datetime.now().isoformat()
        return {
            "jd_id": str(uuid.uuid4()),
            "company": parsed.get("company") or "未知公司",
            "position": parsed.get("position") or "未知岗位",
            "location": parsed.get("location"),
            "salary_range": parsed.get("salary_range"),
            "sections": {
                "responsibilities": parsed.get("sections", {}).get("responsibilities", "") or raw_text[:500],
                "hard_requirements": parsed.get("sections", {}).get("hard_requirements", []) or [],
                "soft_requirements": parsed.get("sections", {}).get("soft_requirements", []) or [],
            },
            "keywords": parsed.get("keywords", []) or [],
            "structured_summary": parsed.get("structured_summary") or {},
            "raw_text": raw_text,
            "meta": {
                "source_type": source_type,
                "created_at": now,
                "updated_at": now,
                "chunk_ids": [],
            },
        }

    def _fallback(self, raw_text: str, source_type: str, reason: str) -> dict:
        """完全无法解析时的兜底"""
        now = datetime.now().isoformat()
        result = {
            "jd_id": str(uuid.uuid4()),
            "company": "未知公司",
            "position": "未知岗位",
            "location": None,
            "salary_range": None,
            "sections": {
                "responsibilities": raw_text or "",
                "hard_requirements": [],
                "soft_requirements": [],
            },
            "keywords": [],
            "raw_text": raw_text or "",
            "meta": {
                "source_type": source_type,
                "created_at": now,
                "updated_at": now,
                "chunk_ids": [],
                "fallback_reason": reason,
            },
        }
        logger.info(f"[JDParser] FALLBACK parse (reason={reason}) → company={result.get('company')} position={result.get('position')}")
        logger.info(f"[JDParser] FALLBACK parsed schema:\n{json.dumps(result, ensure_ascii=False, indent=2)}")
        return result

    @staticmethod
    def _safe_parse_json(raw: str) -> Optional[dict]:
        """安全解析 JSON，支持 markdown 代码块包裹"""
        if not raw or not raw.strip():
            return None
        text = raw.strip()

        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 尝试从 markdown 代码块提取
        for marker in ["```json", "```"]:
            if marker in text:
                start = text.find(marker) + len(marker)
                end = text.find("```", start)
                if end != -1:
                    try:
                        return json.loads(text[start:end].strip())
                    except json.JSONDecodeError:
                        pass

        # 尝试找第一个 { 到最后一个 }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

        return None
