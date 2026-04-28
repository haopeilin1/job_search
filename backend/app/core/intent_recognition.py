"""
三层级联意图识别模块 v2

职责：理解用户需求，提取实体，判断语义完整性
不做：绑定工具链、决定澄清措辞以外的对话策略

架构：
L1 规则层：关键词/正则/词典 → candidate_intent + candidate_entities
L2 小模型层：校准意图 + 校准实体 + 置信度 + 必要实体缺失检测
L3 大模型层：兜底仲裁 + 最终澄清决策

输入：rewritten_query + search_keywords + follow_up_info + history_slots + 三层记忆
输出：IntentResult（demands[] + entities + needs_clarification + clarification_question）
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.core.llm_client import LLMClient, TIMEOUT_STANDARD
from app.core.memory import SessionMemory
from app.core.query_rewrite import QueryRewriteResult

logger = logging.getLogger(__name__)


def _strip_markdown_json(raw: str) -> str:
    """去除LLM输出中可能的markdown代码块包裹"""
    if not raw:
        return raw
    text = raw.strip()
    for marker in ["```json", "```"]:
        if marker in text:
            text = text.replace(marker, "")
    return text.strip()


# ═══════════════════════════════════════════════════════
# 1. 意图分类定义
# ═══════════════════════════════════════════════════════

INTENT_TYPES = {
    "position_explore": {
        "description": "用户想探索/发现自己适合什么岗位",
        "required_entities": [],
        "preferred_entities": ["skills", "years_of_experience", "education"],
        "keywords": ["推荐", "适合", "有什么岗位", "能投什么", "找什么工作", "方向", "机会"],
    },
    "match_assess": {
        "description": "用户想评估自己与某个/某些岗位的匹配度",
        "required_entities": ["evaluation_target"],  # company 或 position 至少一个
        "preferred_entities": ["resume_text"],
        "keywords": ["匹配", "够格", "能过吗", "竞争力", "适合吗", "够资格", "差距", "几率"],
    },
    "attribute_verify": {
        "description": "用户想核实某个岗位的具体属性",
        "required_entities": ["company", "attribute"],
        "preferred_entities": ["position"],
        "keywords": ["薪资", "工资", "加班", "福利", "地点", "要求", "hc", "hc多", "hc有多少", "学历要求", "年限"],
    },
    "interview_prepare": {
        "description": "用户想为面试做准备",
        "required_entities": ["position"],
        "preferred_entities": ["company", "match_result"],
        "keywords": ["面试", "面经", "面试题", "怎么准备", "会问什么", "考察", "八股"],
    },
    "resume_manage": {
        "description": "用户想管理自己的简历",
        "required_entities": [],
        "preferred_entities": [],
        "keywords": ["上传简历", "更新简历", "解析简历", "我的简历", "简历内容"],
    },
    "general_chat": {
        "description": "通用咨询/闲聊",
        "required_entities": [],
        "preferred_entities": [],
        "keywords": ["你好", "谢谢", "再见", "帮忙", "建议", "规划", "行业", "趋势"],
    },
}

# 公司名词典（规则提取用）
_COMPANY_DICT = [
    "字节跳动", "字节", "百度", "阿里巴巴", "阿里", "腾讯", "美团", "京东", "小米", "拼多多",
    "网易", "华为", "快手", "滴滴", "B站", "哔哩哔哩", "携程", "OPPO", "vivo", "联想",
    "字节跳动", "抖音", "TikTok", "飞书",
]

# 岗位名词典（规则提取用）
_POSITION_DICT = [
    "算法工程师", "后端开发工程师", "后端开发", "后端", "前端开发", "前端",
    "产品经理", "PM", "测试工程师", "测试", "运营", "设计师", "数据分析师", "数据分析",
    "AI工程师", "大模型工程师", "NLP工程师", "推荐算法", "搜索算法", "计算机视觉",
    "Java开发", "Python开发", "Go开发", "C++开发",
]

# 属性词典
_ATTRIBUTE_DICT = {
    "薪资": ["薪资", "工资", "待遇", "package", "总包", "年薪", "月薪", "base", "股票", "期权"],
    "加班": ["加班", "996", "大小周", "工作强度", "工作时长", "wlb", "work life balance"],
    "地点": ["地点", "base", "城市", "在哪", "办公地点", "远程", "居家办公"],
    "福利": ["福利", "补贴", "餐补", "房补", "五险一金", "六险一金", "年假", "体检"],
    "hc": ["hc", "headcount", "名额", "招几个人", "还招吗", "hc多", "hc有多少"],
    "学历": ["学历要求", "学历", "本科", "硕士", "博士", "985", "211", "双一流"],
    "年限": ["年限", "经验要求", "几年经验", "工作年限", "应届生", "校招", "社招"],
    "技能": ["技能要求", "技术要求", "需要会什么", "技术栈"],
}


# ═══════════════════════════════════════════════════════
# 2. 数据模型
# ═══════════════════════════════════════════════════════

@dataclass
class Demand:
    """单个需求"""
    intent_type: str           # 如 "match_assess"
    entities: Dict[str, any] = field(default_factory=dict)
    confidence: float = 0.0
    priority: int = 1          # 1=主需求, 2=次要需求


@dataclass
class IntentResult:
    """意图识别最终结果"""
    demands: List[Demand] = field(default_factory=list)
    resolved_entities: Dict[str, any] = field(default_factory=dict)  # 合并后的实体
    is_complete: bool = True
    needs_clarification: bool = False
    clarification_question: str = ""
    missing_entities: List[str] = field(default_factory=list)
    raw_intent_text: str = ""  # 自然语言描述的需求（供Plan模块理解）
    skipped_due_to_timeout: bool = False  # L1+L2 双失败时跳过意图识别


# ═══════════════════════════════════════════════════════
# 3. L1 规则层
# ═══════════════════════════════════════════════════════

class RuleExtractor:
    """L1规则层：快速意图识别 + 实体提取"""

    def extract(
        self,
        rewritten_query: str,
        history_slots: Dict[str, str],
        follow_up_type: str,
    ) -> Tuple[str, Dict[str, any]]:
        """
        返回：(candidate_intent, candidate_entities)
        """
        query = rewritten_query.lower()
        entities = {}

        # 1. 实体提取（基于词典和正则）
        entities.update(self._extract_company(query))
        entities.update(self._extract_position(query))
        entities.update(self._extract_education(query))
        entities.update(self._extract_years(query))
        entities.update(self._extract_attribute(query))

        # 2. 根据follow_up_type融合历史槽位
        entities = self._merge_with_history(entities, history_slots, follow_up_type)

        # 3. 意图识别（基于关键词匹配）
        intent_scores = {}
        for intent_key, intent_def in INTENT_TYPES.items():
            score = 0
            for kw in intent_def["keywords"]:
                if kw.lower() in query:
                    score += 1
            if score > 0:
                intent_scores[intent_key] = score

        if not intent_scores:
            # 兜底：含公司/岗位名但无明显意图词 → 默认match_assess
            if "company" in entities or "position" in entities:
                candidate_intent = "match_assess"
            else:
                candidate_intent = "general_chat"
        else:
            candidate_intent = max(intent_scores, key=intent_scores.get)

        logger.info(f"[L1-Rule] intent={candidate_intent} | entities={entities}")
        return candidate_intent, entities

    def _extract_company(self, query: str) -> Dict[str, str]:
        for c in _COMPANY_DICT:
            if c in query:
                # 标准化：字节→字节跳动
                normalized = "字节跳动" if c == "字节" else c
                return {"company": normalized}
        return {}

    def _extract_position(self, query: str) -> Dict[str, str]:
        for p in _POSITION_DICT:
            if p in query:
                return {"position": p}
        return {}

    def _extract_education(self, query: str) -> Dict[str, str]:
        patterns = [("博士", "博士"), ("硕士", "硕士"), ("本科", "本科"), ("大专", "大专")]
        for pat, val in patterns:
            if pat in query:
                return {"education": val}
        return {}

    def _extract_years(self, query: str) -> Dict[str, any]:
        m = re.search(r"(\d+(?:\.\d+)?)\s*年(?:经验|以上)?", query)
        if m:
            return {"years_of_experience": float(m.group(1))}
        m = re.search(r"应届", query)
        if m:
            return {"years_of_experience": 0}
        return {}

    def _extract_attribute(self, query: str) -> Dict[str, str]:
        for attr_name, keywords in _ATTRIBUTE_DICT.items():
            for kw in keywords:
                if kw in query:
                    return {"attribute": attr_name}
        return {}

    def _merge_with_history(
        self, current: Dict[str, any], history: Dict[str, str], follow_up_type: str
    ) -> Dict[str, any]:
        """根据follow_up_type融合历史槽位"""
        if follow_up_type == "none" or not history:
            return current

        merged = {}

        if follow_up_type == "expand":
            # 历史继承，当前覆盖
            merged.update(history)
            merged.update(current)

        elif follow_up_type == "switch":
            # 当前有的替换历史，当前没有的清空
            for key in history:
                if key in current:
                    merged[key] = current[key]
                else:
                    # 特殊：如果当前换了company但没说position，position继承
                    if key == "position" and "company" in current:
                        merged[key] = history[key]
            merged.update(current)

        elif follow_up_type == "clarify":
            # 历史保留，当前修正覆盖
            merged.update(history)
            merged.update(current)

        return merged


# ═══════════════════════════════════════════════════════
# 4. L2 小模型层
# ═══════════════════════════════════════════════════════

class SmallModelRecognizer:
    """L2小模型层：意图校准 + 实体校准 + 置信度 + 实体完整性检测"""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client

    async def recognize(
        self,
        rewritten_query: str,
        l1_intent: str,
        l1_entities: Dict[str, any],
        history_slots: Dict[str, str],
    ) -> Tuple[List[Demand], float, List[str]]:
        """
        返回：(demands[], confidence, missing_entities)
        """
        if self.llm is None:
            # L2 是轻量级审核修正，用 memory(turbo) 足够，成本比 planner(qwen-plus) 低 2-3 倍
            self.llm = LLMClient.from_config("memory")

        # 构建prompt
        system_prompt = """你是一位求职场景意图审核专家。你的任务是**审核并修正**L1规则层的初步结果，输出最终意图和实体。

工作流程：
1. 接收L1规则层的初步结果（意图+实体）
2. 判断L1结果是否正确：
   - 如果正确，直接确认并输出
   - 如果错误，修正意图类型、补充/修正实体
3. 判断是否有缺少的必要实体（基于意图类型的required_entities定义）
4. 输出审核结论、置信度和修正原因

【审核原则】
- position_explore（岗位探索）: 用户表达了"想找/看看/推荐"等探索意愿即可，**不需要**明确的公司名或岗位名。
- general_chat（通用对话）: 问候、闲聊、职业规划咨询等**一律不需要澄清**。
- match_assess（匹配评估）: 需要company或position至少一个。上下文引用或隐含目标不应标记为缺失。
- interview_prepare（面试准备）: 需要position。泛化表达（如"AI产品面试"）可推断position，不应标记为缺失。
- **绝对不要过度澄清！** 只有当核心信息完全无法推断时，才标记missing_entities。

意图定义：
- position_explore: 探索适合什么岗位。required_entities: []（无强制要求）
- match_assess: 评估匹配度。required_entities: ["evaluation_target"]（company或position至少一个）
- attribute_verify: 核实属性。required_entities: ["company", "attribute"]
- interview_prepare: 面试准备。required_entities: ["position"]
- resume_manage: 简历管理。required_entities: []
- general_chat: 通用对话。required_entities: []

【多意图识别】
用户一句话可能包含多个需求，必须同时识别出来。例如：
- "帮我看看字节AI产品我匹配不，再给我准备几道面试题" → demands 应包含 match_assess + interview_prepare
- "推几个我能投的，顺便告诉我面试要准备啥" → demands 应包含 position_explore + interview_prepare
- "字节和百度的产品岗，哪个我更合适？" → demands 应包含两个 match_assess（或一个对比意图）

输出格式（严格JSON）：
{
  "demands": [
    {
      "intent_type": "match_assess",
      "entities": {"company": "字节跳动", "position": "算法工程师"},
      "priority": 1
    },
    {
      "intent_type": "interview_prepare",
      "entities": {"position": "算法工程师"},
      "priority": 2
    }
  ],
  "confidence": 0.92,
  "missing_entities": [],
  "corrected": false,
  "corrected_reason": "L1只识别了match_assess，漏掉了interview_prepare，已补充"
}"""

        user_prompt = f"""【用户query】{rewritten_query}

【L1规则层初步结果】
- 意图：{l1_intent}
- 实体：{json.dumps(l1_entities, ensure_ascii=False)}

【历史槽位】{json.dumps(history_slots, ensure_ascii=False)}

请审核L1结果，输出JSON。如果L1结果正确，corrected=false；如果需要修正，corrected=true并说明原因。"""

        raw = await self.llm.generate(
            prompt=user_prompt,
            system=system_prompt,
            temperature=0.2,
            max_tokens=800,
            timeout=TIMEOUT_STANDARD,  # 20s，turbo 审核很快
        )
        data = json.loads(_strip_markdown_json(raw))

        demands = []
        for d in data.get("demands", []):
            demands.append(Demand(
                intent_type=d.get("intent_type", l1_intent),
                entities=d.get("entities", {}),
                confidence=data.get("confidence", 0.5),
                priority=d.get("priority", 1),
            ))

        confidence = data.get("confidence", 0.5)
        missing = data.get("missing_entities", [])

        logger.info(f"[L2-SmallModel] demands={[d.intent_type for d in demands]} | conf={confidence} | missing={missing}")
        return demands, confidence, missing


# ═══════════════════════════════════════════════════════
# 5. L3 大模型层
# ═══════════════════════════════════════════════════════

class LargeModelArbitrator:
    """L3大模型层：兜底仲裁 + 澄清决策"""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client

    async def arbitrate(
        self,
        rewritten_query: str,
        l1_result: Tuple[str, Dict],
        l2_demands: List[Demand],
        l2_confidence: float,
        l2_missing: List[str],
        history_slots: Dict[str, str],
        session_history: str = "",
    ) -> IntentResult:
        """
        L3兜底仲裁。
        触发条件：
        - L2置信度 < 0.7
        - L2与L1意图不一致
        - L2发现必要实体缺失
        """
        if self.llm is None:
            self.llm = LLMClient.from_config("chat")

        l1_intent, l1_entities = l1_result

        # 构建prompt
        system_prompt = """你是一位资深的求职对话理解专家。请基于完整上下文，做出最终意图判断和澄清决策。

你的任务是：
1. 理解用户的真实需求（可能一句话包含多个需求）
2. 提取/补全所有相关实体
3. 判断是否有缺少的必要实体，决定是否向用户澄清
4. 如果不需要澄清，输出完整的意图和实体

【多意图识别】
用户一句话可能包含多个需求，必须同时输出。例如：
- "帮我看看字节AI产品我匹配不，再给我准备几道面试题" → demands: [match_assess, interview_prepare]
- "推几个我能投的，顺便告诉我工资多少" → demands: [position_explore, attribute_verify]

输出格式（严格JSON）：
{
  "needs_clarification": false,
  "clarification_question": "",
  "clarification_options": [],
  "demands": [
    {
      "intent_type": "match_assess",
      "entities": {"company": "字节跳动", "position": "算法工程师"},
      "priority": 1
    }
  ],
  "resolved_entities": {"company": "字节跳动", "position": "算法工程师"},
  "reasoning": "用户明确问匹配度，提到了字节跳动和算法工程师，信息完整"
}"""

        user_prompt = f"""用户query：{rewritten_query}

L1规则层结果：意图={l1_intent}，实体={json.dumps(l1_entities, ensure_ascii=False)}
L2小模型结果：demands={[d.intent_type for d in l2_demands]}，置信度={l2_confidence}，缺失实体={l2_missing}

历史槽位：{json.dumps(history_slots, ensure_ascii=False)}
{session_history}

请综合判断，输出JSON："""

        try:
            raw = await self.llm.generate(
                prompt=user_prompt,
                system=system_prompt,
                temperature=0.3,
                max_tokens=1000,
                timeout=TIMEOUT_STANDARD,  # 20s，兜底仲裁
            )
            data = json.loads(_strip_markdown_json(raw))

            needs_clarification = data.get("needs_clarification", False)

            if needs_clarification:
                return IntentResult(
                    demands=[],
                    needs_clarification=True,
                    clarification_question=data.get("clarification_question", "能再详细说明一下吗？"),
                    missing_entities=data.get("missing_entities", l2_missing),
                )

            # 不需要澄清，构建最终结果
            demands = []
            for d in data.get("demands", []):
                demands.append(Demand(
                    intent_type=d.get("intent_type", "general_chat"),
                    entities=d.get("entities", {}),
                    priority=d.get("priority", 1),
                    confidence=0.95,
                ))

            resolved_entities = data.get("resolved_entities", {})

            # 检查语义完整性
            is_complete, missing = self._check_completeness(demands, resolved_entities)

            logger.info(f"[L3-LargeModel] demands={[d.intent_type for d in demands]} | complete={is_complete}")

            return IntentResult(
                demands=demands,
                resolved_entities=resolved_entities,
                is_complete=is_complete,
                needs_clarification=not is_complete,
                clarification_question=self._build_clarification_question(missing) if not is_complete else "",
                missing_entities=missing,
                raw_intent_text=self._build_raw_intent_text(demands),
            )

        except Exception as e:
            logger.error(f"[L3-LargeModel] 兜底失败: {e}")
            # 极端fallback：使用L2结果
            return IntentResult(
                demands=l2_demands,
                resolved_entities=l2_demands[0].entities if l2_demands else {},
                is_complete=len(l2_missing) == 0,
                needs_clarification=len(l2_missing) > 0,
                clarification_question=self._build_clarification_question(l2_missing) if l2_missing else "",
                missing_entities=l2_missing,
            )

    def _check_completeness(
        self, demands: List[Demand], entities: Dict[str, any]
    ) -> Tuple[bool, List[str]]:
        """检查所有demand的语义完整性"""
        missing = []
        for demand in demands:
            intent_def = INTENT_TYPES.get(demand.intent_type)
            if not intent_def:
                continue
            required = intent_def.get("required_entities", [])
            for req in required:
                if req == "evaluation_target":
                    # 特殊：company或position至少一个
                    if "company" not in entities and "position" not in entities:
                        missing.append("evaluation_target")
                elif req not in entities or entities[req] is None:
                    missing.append(req)
        return len(missing) == 0, list(set(missing))

    def _build_clarification_question(self, missing: List[str]) -> str:
        """根据缺失实体构建澄清问题"""
        if not missing:
            return ""
        if "evaluation_target" in missing:
            return "请问您指的是哪家公司的哪个岗位呢？"
        if "company" in missing:
            return "请问您指的是哪家公司呢？"
        if "position" in missing:
            return "请问您指的是什么岗位呢？"
        if "attribute" in missing:
            return "您想了解这个岗位的哪方面信息呢？（如薪资、加班、地点等）"
        return "能再详细说明一下吗？"

    def _build_raw_intent_text(self, demands: List[Demand]) -> str:
        """构建自然语言描述的需求文本"""
        parts = []
        for d in demands:
            desc = INTENT_TYPES.get(d.intent_type, {}).get("description", d.intent_type)
            parts.append(desc)
        return "；".join(parts)


# ═══════════════════════════════════════════════════════
# 6. 三层级联合入口
# ═══════════════════════════════════════════════════════

class IntentRecognizer:
    """意图识别器：三层级联入口"""

    def __init__(self):
        self.rule_extractor = RuleExtractor()
        self.small_model = SmallModelRecognizer()
        self.large_model = LargeModelArbitrator()

    async def recognize(
        self,
        rewrite_result: QueryRewriteResult,
        session: SessionMemory,
    ) -> IntentResult:
        """
        漏斗模式意图识别主入口。
        流程：L1(规则) → L2(审核修正) → 信任判断 → L3(兜底) 或 跳过
        """
        rewritten_query = rewrite_result.rewritten_query
        follow_up_type = rewrite_result.follow_up_type

        # 获取历史槽位
        history_slots = self._get_history_slots(session)

        # L1: 规则层（本地计算，不会超时）
        l1_intent, l1_entities = self.rule_extractor.extract(
            rewritten_query, history_slots, follow_up_type
        )

        # L2: 漏斗审核层
        l2_ok = False
        l2_demands: List[Demand] = []
        l2_confidence = 0.0
        l2_missing: List[str] = []

        try:
            l2_demands, l2_confidence, l2_missing = await self.small_model.recognize(
                rewritten_query, l1_intent, l1_entities, history_slots
            )
            l2_ok = True
        except Exception as e:
            logger.warning(f"[IntentRecognizer] L2 调用失败: {e}")

        if l2_ok:
            _CRITICAL_ENTITIES = {"company", "position", "evaluation_target", "attribute"}
            l2_missing_critical = [m for m in l2_missing if m in _CRITICAL_ENTITIES]
            l2_trustworthy = l2_confidence >= 0.75 and not l2_missing_critical

            if l2_trustworthy:
                # L2 审核通过，直接信任
                is_complete = len(l2_missing) == 0
                return IntentResult(
                    demands=l2_demands,
                    resolved_entities=l2_demands[0].entities if l2_demands else {},
                    is_complete=is_complete,
                    needs_clarification=not is_complete,
                    clarification_question=self.large_model._build_clarification_question(l2_missing) if not is_complete else "",
                    missing_entities=l2_missing,
                    raw_intent_text=self.large_model._build_raw_intent_text(l2_demands),
                )
            # L2 可信但 confidence < 0.75 或有关键缺失 → 进 L3 兜底
            logger.info(f"[IntentRecognizer] L2 confidence={l2_confidence} 不足或缺失关键实体，进入 L3")
            session_history = self._build_session_history(session)
            return await self.large_model.arbitrate(
                rewritten_query=rewritten_query,
                l1_result=(l1_intent, l1_entities),
                l2_demands=l2_demands,
                l2_confidence=l2_confidence,
                l2_missing=l2_missing,
                history_slots=history_slots,
                session_history=session_history,
            )

        # L2 失败（超时/异常）：跳过意图识别，直接让 Planner 处理
        logger.warning("[IntentRecognizer] L2 失败，跳过意图识别，直接进入 Planner")
        return IntentResult(
            demands=[Demand(intent_type="general_chat", entities={}, confidence=0.0, priority=1)],
            resolved_entities={},
            is_complete=True,
            needs_clarification=False,
            clarification_question="",
            missing_entities=[],
            raw_intent_text="general_chat",
            skipped_due_to_timeout=True,
        )

    def _get_history_slots(self, session: SessionMemory) -> Dict[str, str]:
        """获取上轮已确认的槽位"""
        if not session or not hasattr(session, "working_memory"):
            return {}
        turns = session.working_memory.turns if hasattr(session.working_memory, "turns") else []
        if not turns:
            return {}
        last_turn = turns[-1]
        if hasattr(last_turn, "global_slots"):
            return {k: v for k, v in last_turn.global_slots.items() if v is not None}
        if hasattr(session, "global_slots"):
            return {k: v for k, v in session.global_slots.items() if v is not None}
        return {}

    def _build_session_history(self, session: SessionMemory) -> str:
        """构建最近几轮对话摘要，供L3参考"""
        if not session or not hasattr(session, "working_memory"):
            return ""
        turns = session.working_memory.turns if hasattr(session.working_memory, "turns") else []
        if len(turns) <= 1:
            return ""

        lines = ["\n最近对话历史："]
        for t in turns[-3:]:
            user_msg = getattr(t, "user_message", "")[:50]
            assistant_msg = getattr(t, "assistant_reply", "")[:50]
            lines.append(f"用户：{user_msg}")
            lines.append(f"助手：{assistant_msg}")
        return "\n".join(lines)
