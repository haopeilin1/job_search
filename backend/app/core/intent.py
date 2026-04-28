import json
import logging
from enum import Enum
from typing import Optional, Literal, Callable, List
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# 1. 意图枚举
# ═══════════════════════════════════════════════════════

class IntentType(str, Enum):
    MATCH_SINGLE = "match_single"
    GLOBAL_MATCH = "global_match"
    RAG_QA = "rag_qa"
    GENERAL = "general"


class RouteLayer(str, Enum):
    RULE = "rule"
    LLM = "llm"
    LLM_FALLBACK = "llm_fallback"


# ═══════════════════════════════════════════════════════
# 2. 路由结果模型
# ═══════════════════════════════════════════════════════

class IntentResult(BaseModel):
    intent: IntentType = Field(..., description="识别到的意图类型")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度 0.0-1.0")
    layer: Literal["rule", "llm", "llm_fallback"] = Field(
        ..., description="决策层级：rule=规则命中, llm=LLM命中, llm_fallback=LLM置信度不足降级"
    )
    rule: Optional[str] = Field(None, description="规则命中时的规则名")
    reason: Optional[str] = Field(None, description="判断理由")
    metadata: dict = Field(default_factory=dict, description="规则层或 LLM 层附带的元数据（如公司名、岗位名、属性等）")


# ═══════════════════════════════════════════════════════
# 3. 规则注册表基础设施
# ═══════════════════════════════════════════════════════

@dataclass
class RuleResult:
    intent: IntentType
    confidence: float
    rule_id: str
    layer: RouteLayer = RouteLayer.RULE
    metadata: dict = field(default_factory=dict)


RuleFunc = Callable[[str, list, list, list, list], Optional[RuleResult]]


class RuleRegistry:
    """规则注册表：支持按优先级注册，classify 时按优先级遍历，命中即停"""

    def __init__(self):
        self._rules: List[tuple[int, str, RuleFunc]] = []
        self._hit_counter: dict[str, int] = {}

    def register(self, priority: int, rule_id: str, func: RuleFunc):
        self._rules.append((priority, rule_id, func))
        self._rules.sort(key=lambda x: x[0])
        if rule_id not in self._hit_counter:
            self._hit_counter[rule_id] = 0

    def classify(self, message: str, attachments: list, context: list, kb_companies: list, kb_positions: list) -> Optional[RuleResult]:
        for priority, rule_id, func in self._rules:
            result = func(message, attachments, context, kb_companies, kb_positions)
            if result:
                self._hit_counter[rule_id] += 1
                logger.info(f"[RuleHit] {rule_id} | msg='{message[:40]}...' | intent={result.intent.value} | meta={result.metadata}")
                return result
        return None

    def get_stats(self) -> dict:
        return {"total_hits": sum(self._hit_counter.values()), "by_rule": self._hit_counter.copy()}


# ═══════════════════════════════════════════════════════
# 4. 知识库已知实体（可后续从数据库动态加载）
# ═══════════════════════════════════════════════════════

KB_COMPANIES = [
    "字节跳动", "字节", "ByteDance", "抖音",
    "百度", "Baidu",
    "阿里巴巴", "阿里", "Alibaba", "淘天", "蚂蚁", "蚂蚁金服",
    "腾讯", "Tencent", "微信",
    "美团", "Meituan",
    "京东", "JD",
    "快手", "Kuaishou",
    "小红书", "Red",
    "拼多多", "PDD",
    "小米", "Xiaomi",
    "华为", "Huawei",
    "网易", "NetEase",
    "滴滴", "Didi",
    "携程", "Ctrip",
    "贝壳", "Beike",
    "蔚来", "NIO",
    "理想", "LiAuto",
    "大疆", "DJI",
    "商汤", "SenseTime",
    "科大讯飞", "讯飞",
    "360", "奇安信",
    "B站", "哔哩哔哩", "bilibili",
    "知乎", "Zhihu",
    "微博", "Weibo",
    "OPPO", "VIVO",
    "联想", "Lenovo",
    "平安", "PingAn",
    "招商银行", "招行",
]

KB_POSITIONS = [
    "产品经理", "算法工程师", "后端开发", "前端开发",
    "数据分析师", "运营", "设计师", "AI产品经理", "产品实习生"
]


# ═══════════════════════════════════════════════════════
# 5. L1 结构特征层（priority=10~19）
#    只看格式/结构，不看语义，零成本，零歧义
# ═══════════════════════════════════════════════════════

def rule_attachment_with_match_intent(msg, attachments, ctx, kb_companies, kb_positions):
    """R1-附件+匹配词：用户上传了文件，且明确表达了'匹配/对比/适合'意图 → 单JD匹配"""
    if not attachments or len(attachments) == 0:
        return None
    MATCH_KWS = ["匹配", "适合", "差距", "对比", "我和", "我的简历", "匹配度", "差多少"]
    if any(kw in msg for kw in MATCH_KWS):
        return RuleResult(
            intent=IntentType.MATCH_SINGLE,
            confidence=1.0,
            rule_id="R1-附件+匹配词",
            metadata={"sub_type": "match_intent", "attachment_count": len(attachments)}
        )
    return None


def rule_attachment_with_info_intent(msg, attachments, ctx, kb_companies, kb_positions):
    """R1-附件+查询词：用户上传了文件，问的是'要求什么/需要什么/技能/薪资' → 单JD匹配（预览）"""
    if not attachments or len(attachments) == 0:
        return None
    INFO_KWS = ["要求什么", "需要什么", "技能", "薪资", "福利", "学历", "工作内容", "职责", "岗位介绍", "这家公司", "这个岗位"]
    if any(kw in msg for kw in INFO_KWS):
        return RuleResult(
            intent=IntentType.MATCH_SINGLE,
            confidence=0.95,
            rule_id="R1-附件+查询词",
            metadata={"sub_type": "jd_preview", "attachment_count": len(attachments)}
        )
    return None


def rule_attachment_no_intent(msg, attachments, ctx, kb_companies, kb_positions):
    """R1-附件+无明确意图：用户只丢了附件，query为空或极短 → 默认JD预览"""
    if not attachments or len(attachments) == 0:
        return None
    trimmed = msg.strip()
    if len(trimmed) < 8 or trimmed in ["看看", "分析一下", "帮我看看", ""]:
        return RuleResult(
            intent=IntentType.MATCH_SINGLE,
            confidence=0.9,
            rule_id="R1-附件+无明确意图",
            metadata={"sub_type": "jd_preview", "attachment_count": len(attachments)}
        )
    return None


def rule_jd_long_text(msg, attachments, ctx, kb_companies, kb_positions):
    """R1-JD长文本：用户粘贴了大段JD文本（非附件），含明显JD格式特征 → 单JD匹配"""
    JD_MARKERS = [
        "岗位职责", "任职要求", "硬性要求", "软性要求", "加分项",
        "薪资范围", "工作地点", "五险一金", "带薪年假", "简历投递",
        "工作职责", "任职资格", "岗位描述", "职位要求"
    ]
    if len(msg) > 200 and any(marker in msg for marker in JD_MARKERS):
        return RuleResult(
            intent=IntentType.MATCH_SINGLE,
            confidence=1.0,
            rule_id="R1-JD长文本",
            metadata={"sub_type": "match_intent", "text_length": len(msg)}
        )
    return None


# ═══════════════════════════════════════════════════════
# 6. L2 强关键词层（priority=20~29）
#    单点命中，几乎无歧义
# ═══════════════════════════════════════════════════════

def rule_global_match(msg, attachments, ctx, kb_companies, kb_positions):
    """R2-全局对比：用户明确要求全局对比、推荐排序 → global_match
    排除同时含公司名+匹配意图词的场景（留给R3引用式匹配处理）"""
    # 如果同时包含公司名和匹配意图词，可能是引用式匹配，不在这里拦截
    has_company = any(c in msg for c in kb_companies)
    REF_MATCH_KWS = ["匹配度", "适合吗", "匹配吗", "差距", "差多少", "适合我", "我能去", "够格", "契合度"]
    if has_company and any(kw in msg for kw in REF_MATCH_KWS):
        return None

    GLOBAL_KWS = [
        "推荐", "适合哪家", "匹配度最高", "投递顺序",
        "优先投", "哪些岗位", "哪些公司",
        "我该投哪个", "帮我选一下", "帮我看看适合", "排序",
        "最匹配", "最适合", "投哪家", "投哪个", "全局对比"
    ]
    if any(kw in msg for kw in GLOBAL_KWS):
        hit = next(k for k in GLOBAL_KWS if k in msg)
        return RuleResult(
            intent=IntentType.GLOBAL_MATCH,
            confidence=1.0,
            rule_id="R2-全局对比",
            metadata={"hit_word": hit}
        )

    # "对比"单独判断：如果同时含知识库实体（公司/岗位）和属性词，可能是RAG查询，不拦截
    if "对比" in msg:
        has_entity = any(e in msg for e in kb_companies + kb_positions)
        ATTR_WORDS = ["技能", "薪资", "工资", "要求", "福利", "学历", "经验", "工作年限"]
        has_attr = any(w in msg for w in ATTR_WORDS)
        if has_entity and has_attr:
            return None
        return RuleResult(
            intent=IntentType.GLOBAL_MATCH,
            confidence=1.0,
            rule_id="R2-全局对比",
            metadata={"hit_word": "对比"}
        )
    return None


def rule_greeting(msg, attachments, ctx, kb_companies, kb_positions):
    """R2-问候：纯问候语 → general
    注意：不再用 len<10 兜底，避免截获正常短句"""
    GREETINGS = ["你好", "您好", "嗨", "hello", "hi", "在吗", "早上好", "晚上好", "哈喽", "打扰了", "谢谢", "感谢"]
    trimmed = msg.strip().lower()
    if trimmed in [g.lower() for g in GREETINGS]:
        return RuleResult(
            intent=IntentType.GENERAL,
            confidence=1.0,
            rule_id="R2-问候",
            metadata={"length": len(trimmed)}
        )
    return None


def rule_general_career(msg, attachments, ctx, kb_companies, kb_positions):
    """R2-面试通用：面试技巧、职业规划等通用求职咨询 → general"""
    GENERAL_KWS = [
        "面试技巧", "怎么准备", "职业规划", "发展路径",
        "行业趋势", "如何转行", "怎么跳槽", "谈薪技巧",
        "简历优化", "自我介绍", "离职原因", "简历修改"
    ]
    if any(kw in msg for kw in GENERAL_KWS) and len(msg) < 80:
        hit = next(k for k in GENERAL_KWS if k in msg)
        return RuleResult(
            intent=IntentType.GENERAL,
            confidence=0.95,
            rule_id="R2-面试通用",
            metadata={"hit_word": hit}
        )
    return None


# ═══════════════════════════════════════════════════════
# 7. L3 组合模式层（priority=30~39）
#    需同时满足多个条件，降低误判
# ═══════════════════════════════════════════════════════

def rule_referenced_match(msg, attachments, ctx, kb_companies, kb_positions):
    """R3-引用式匹配：用户提及了知识库中已知的公司名，且表达了匹配/对比意图 → match_single"""
    if attachments and len(attachments) > 0:
        return None
    has_company = any(c in msg for c in kb_companies)

    # 强匹配意图词（明确表达"我和这个岗位匹不匹配"）
    STRONG_MATCH_KWS = [
        "匹配度", "适合吗", "匹配吗", "差距", "差多少",
        "适合我", "我能去", "够格", "契合度"
    ]
    # 弱匹配意图词（"分析一下""对比"等可能指RAG查询）
    WEAK_MATCH_KWS = ["分析一下", "我和", "对比"]

    has_strong_match = any(kw in msg for kw in STRONG_MATCH_KWS)
    has_weak_match = any(kw in msg for kw in WEAK_MATCH_KWS)

    # 如果同时包含属性查询词，且没有强匹配意图，优先走RAG
    ATTR_WORDS = ["要求", "技能", "薪资", "工资", "福利", "学历", "经验"]
    has_attr = any(w in msg for w in ATTR_WORDS)
    if has_company and has_attr and not has_strong_match:
        return None

    has_match_intent = has_strong_match or has_weak_match
    if has_company and has_match_intent:
        hit_company = next(c for c in kb_companies if c in msg)
        return RuleResult(
            intent=IntentType.MATCH_SINGLE,
            confidence=0.95 if has_strong_match else 0.85,
            rule_id="R3-引用式匹配",
            metadata={"company": hit_company, "sub_type": "referenced_match"}
        )
    return None


def rule_rag_attribute_query(msg, attachments, ctx, kb_companies, kb_positions):
    """R3-RAG属性查询：用户在问某岗位/公司的具体属性 → rag_qa
    支持疑问词模式 或 "公司+属性词"结尾的短查询模式"""
    QUESTION_WORDS = ["什么", "多少", "怎么", "吗", "呢", "哪些", "如何", "介绍一下", "怎样", "啥"]
    ATTR_WORDS = ["技能", "薪资", "工资", "要求", "福利", "学历", "经验", "工作年限", "团队规模", "加班", "作息", "技术栈"]
    has_question = any(w in msg for w in QUESTION_WORDS)
    has_attr = any(w in msg for w in ATTR_WORDS)
    has_entity = any(e in msg for e in kb_companies + kb_positions)

    # 模式A：标准疑问句（疑问词 + 属性词 + 实体）
    if has_question and has_attr and has_entity:
        return RuleResult(
            intent=IntentType.RAG_QA,
            confidence=0.9,
            rule_id="R3-RAG属性查询",
            metadata={"attributes": [w for w in ATTR_WORDS if w in msg], "mode": "question"}
        )

    # 模式B：以属性词结尾的短查询（如"对比阿里与字节的要求"）
    ends_with_attr = any(msg.strip().endswith(w) for w in ATTR_WORDS)
    if ends_with_attr and has_attr and has_entity and len(msg) < 60:
        return RuleResult(
            intent=IntentType.RAG_QA,
            confidence=0.85,
            rule_id="R3-RAG属性查询",
            metadata={"attributes": [w for w in ATTR_WORDS if w in msg], "mode": "ending"}
        )
    return None


def rule_rag_company_overview(msg, attachments, ctx, kb_companies, kb_positions):
    """R3-RAG公司查询：用户问某公司在招什么岗位/有什么职位 → rag_qa"""
    has_company = any(c in msg for c in kb_companies)
    POSITION_KWS = ["岗位", "职位", "招聘", "招什么", "HC", "在招", "机会", "缺人", "招人"]
    has_position_kw = any(kw in msg for kw in POSITION_KWS)
    if has_company and has_position_kw:
        return RuleResult(
            intent=IntentType.RAG_QA,
            confidence=0.9,
            rule_id="R3-RAG公司查询",
            metadata={}
        )
    return None


# ═══════════════════════════════════════════════════════
# 8. 规则分类器（使用注册表）
# ═══════════════════════════════════════════════════════

def _load_kb_entities() -> tuple[list[str], list[str]]:
    """从 jds.json 动态加载公司名和岗位名，硬编码作为兜底"""
    companies = set(KB_COMPANIES)
    positions = set(KB_POSITIONS)
    try:
        from pathlib import Path
        jds_file = Path(__file__).resolve().parent.parent.parent / "data" / "jds.json"
        if jds_file.exists():
            jds = json.loads(jds_file.read_text(encoding="utf-8"))
            for jd in jds:
                c = jd.get("company", "")
                if c:
                    companies.add(c)
                p = jd.get("position", "") or jd.get("title", "")
                if p:
                    positions.add(p)
    except Exception:
        pass
    return list(companies), list(positions)


class RuleClassifier:
    """基于分层注册表的规则分类器"""

    def __init__(self):
        self.registry = RuleRegistry()
        # L1: 结构特征层
        self.registry.register(10, "R1-附件+匹配词", rule_attachment_with_match_intent)
        self.registry.register(11, "R1-附件+查询词", rule_attachment_with_info_intent)
        self.registry.register(12, "R1-附件+无明确意图", rule_attachment_no_intent)
        self.registry.register(15, "R1-JD长文本", rule_jd_long_text)
        # L2: 强关键词层
        self.registry.register(20, "R2-全局对比", rule_global_match)
        self.registry.register(21, "R2-问候", rule_greeting)
        self.registry.register(22, "R2-面试通用", rule_general_career)
        # L3: 组合模式层
        self.registry.register(30, "R3-引用式匹配", rule_referenced_match)
        self.registry.register(31, "R3-RAG属性查询", rule_rag_attribute_query)
        self.registry.register(32, "R3-RAG公司查询", rule_rag_company_overview)

    def classify(self, message: str, attachments: list) -> tuple[Optional[IntentType], float, Optional[str], dict]:
        kb_companies, kb_positions = _load_kb_entities()
        result = self.registry.classify(message, attachments or [], [], kb_companies, kb_positions)
        if result:
            return (result.intent, result.confidence, result.rule_id, result.metadata)
        return (None, 0.0, None, {})


# ═══════════════════════════════════════════════════════
# 9. LLM 分类器（保持不变，兜底）
# ═══════════════════════════════════════════════════════

class LLMClassifier:
    _CLASSIFY_PROMPT = """你是一位对话意图识别专家。请根据用户输入判断其意图，并输出严格 JSON。

【可识别的意图类型】
1. match_single —— 用户提供了具体的 JD（岗位描述）内容，希望与简历做一对一匹配分析
2. global_match —— 用户要求从知识库中对比/推荐/筛选多个岗位，找出最适合的或给出投递优先级
3. rag_qa —— 用户询问知识库中某岗位/公司的具体信息，如薪资、要求、福利、团队情况等
4. general —— 打招呼、闲聊、行业泛泛咨询、无法归类到上述三类的情况

【判断指南】
- match_single：用户上传了文件，或粘贴了包含"岗位职责""任职要求"等字样的长文本，或提及具体公司名并问匹配度
- global_match：用户问"我适合什么岗位""帮我推荐几家""对比这些职位"等
- rag_qa：用户问"字节这个岗位薪资多少""百度PM需要什么技能"等具体问题
- general："你好""谢谢""什么是RAG"等泛泛而谈

【输出格式】
必须返回合法 JSON，不要包含 markdown 代码块：
{
  "intent": "match_single|global_match|rag_qa|general",
  "confidence": 0.0-1.0,
  "reason": "简要说明判断理由"
}

【约束】
- confidence > 0.8 时返回对应意图
- confidence <= 0.7 时建议返回 general，避免误判
- 只输出 JSON，不要添加任何解释性文字
- 只做「问题 vs 预设意图集」的二次判断，**不改变用户输入中的任何名词**（公司名、岗位名、技能名保留原文）
- 若输入已做口语降噪/指代消解，请直接基于其语义做判断"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    async def classify(self, message: str, context: Optional[list[dict]] = None, history_context: str = "") -> dict:
        context_text = ""
        if history_context:
            context_text = f"\n【历史对话上下文】\n{history_context[:800]}"
        elif context and len(context) > 0:
            last = context[-1]
            context_text = f"\n【上文】用户：{last.get('message', '')[:100]}"

        full_prompt = f"{self._CLASSIFY_PROMPT}\n{context_text}\n\n【当前输入】\n{message}\n\n请输出 JSON："

        raw_result = None
        for attempt in range(2):
            try:
                if self.llm_client is None:
                    return {"intent": "general", "confidence": 0.0, "reason": "llm_client_not_available"}

                raw_text = await self.llm_client.generate(
                    prompt=full_prompt,
                    temperature=0.3,
                    max_tokens=256,
                )
                raw_result = self._parse_llm_output(raw_text)
                if raw_result is not None:
                    break
            except Exception as e:
                logger.warning(f"[LLMClassifier] attempt {attempt + 1} failed: {e}")
                continue

        if raw_result is None:
            return {"intent": "general", "confidence": 0.0, "reason": "parse_error"}

        if raw_result.get("confidence", 0.0) < 0.7:
            raw_result["intent"] = "general"
            raw_result["reason"] = f"confidence_low: {raw_result.get('reason', '')}"

        return raw_result

    def _parse_llm_output(self, raw_text: str) -> Optional[dict]:
        if not raw_text or not raw_text.strip():
            return None
        try:
            data = json.loads(raw_text.strip())
        except json.JSONDecodeError:
            try:
                start = raw_text.find("{")
                end = raw_text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    data = json.loads(raw_text[start : end + 1])
                else:
                    return None
            except json.JSONDecodeError:
                return None

        intent_str = data.get("intent")
        if intent_str not in {m.value for m in IntentType}:
            return None

        confidence = float(data.get("confidence", 0.0))
        return {
            "intent": intent_str,
            "confidence": confidence,
            "reason": data.get("reason", ""),
        }


# ═══════════════════════════════════════════════════════
# 10. 统一路由入口
# ═══════════════════════════════════════════════════════

class IntentRouter:
    """意图路由统一入口，协调规则层与 LLM 层"""

    def __init__(self, llm_client=None):
        self.rule_classifier = RuleClassifier()
        self.llm_classifier = LLMClassifier(llm_client=llm_client)

    async def route(
        self,
        message: str,
        attachments: list,
        context: Optional[list[dict]] = None,
        history_context: str = "",
    ) -> IntentResult:
        # ── 第一层：规则分类 ──
        rule_intent, rule_conf, rule_name, rule_meta = self.rule_classifier.classify(message, attachments)

        if rule_intent is not None and rule_conf >= 0.85:
            result = IntentResult(
                intent=rule_intent,
                confidence=rule_conf,
                layer="rule",
                rule=rule_name,
                reason=None,
                metadata=rule_meta,
            )
            logger.info(
                f"[IntentRouter] message='{message[:30]}...' -> intent={result.intent.value}, "
                f"layer={result.layer}, rule={result.rule}, meta={result.metadata}"
            )
            return result

        # ── 第二层：LLM 分类 ──
        llm_result = await self.llm_classifier.classify(message, context, history_context=history_context)
        llm_intent_str = llm_result.get("intent", "general")
        llm_confidence = llm_result.get("confidence", 0.0)
        llm_reason = llm_result.get("reason", "")

        if llm_confidence < 0.7:
            result = IntentResult(
                intent=IntentType.GENERAL,
                confidence=llm_confidence,
                layer="llm_fallback",
                rule=None,
                reason=f"confidence={llm_confidence:.2f}, original={llm_intent_str}, {llm_reason}",
                metadata={},
            )
        else:
            result = IntentResult(
                intent=IntentType(llm_intent_str),
                confidence=llm_confidence,
                layer="llm",
                rule=None,
                reason=llm_reason,
                metadata={},
            )

        logger.info(
            f"[IntentRouter] message='{message[:30]}...' -> intent={result.intent.value}, "
            f"layer={result.layer}, confidence={result.confidence:.2f}"
        )
        return result
    