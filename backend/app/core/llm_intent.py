"""
LLM 路线专用意图识别模块 —— 规则引擎 + 小模型校准

采用 JTBD 六类意图体系，不与规则路线的四类意图共用任何组件。

流水线：
    Query 改写 → RuleRegistry（16 条规则，分层遍历）
    → STRONG + 无话题切换 → 直接确认
    → STRONG + 话题切换 / WEAK / MISS → SmallModelCalibrator（小模型校准）
    → confidence >= 0.8 → 采信
    → confidence < 0.8 → 标记 needs_llm_fallback（大模型兜底预留）
"""

import json
import logging
from enum import Enum
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass, field

from app.core.llm_client import LLMClient
from app.core.memory import SessionMemory
from app.core.query_rewrite import QueryRewriteResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# 1. 枚举
# ═══════════════════════════════════════════════════════

class LLMIntentType(str, Enum):
    EXPLORE = "explore"
    ASSESS = "assess"
    VERIFY = "verify"
    PREPARE = "prepare"
    MANAGE = "manage"
    CHAT = "chat"


class RuleStrength(str, Enum):
    STRONG = "STRONG"
    WEAK = "WEAK"
    MISS = "MISS"


# ═══════════════════════════════════════════════════════
# 2. 数据模型
# ═══════════════════════════════════════════════════════

@dataclass
class LLMRuleResult:
    intent: Optional[LLMIntentType]
    strength: RuleStrength
    rule_name: str
    trigger: str
    metadata: dict = field(default_factory=dict)


@dataclass
class CalibrationResult:
    intent: LLMIntentType
    confidence: float
    reason: str
    slots: dict = field(default_factory=dict)
    slot_sources: dict = field(default_factory=dict)
    missing_slots: list = field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str = ""
    clarification_options: list = field(default_factory=list)
    rule_agreement: bool = False
    context_driven: bool = False
    needs_llm_fallback: bool = False


@dataclass
class FallbackResult:
    intent: LLMIntentType
    confidence: float
    reason: str
    slots: dict = field(default_factory=dict)
    slot_sources: dict = field(default_factory=dict)
    missing_slots: list = field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str = ""
    clarification_options: list = field(default_factory=list)
    arbitration: str = ""
    candidate_options: list = field(default_factory=list)



@dataclass
class IntentCandidate:
    """意图候选：校准后的单一意图 + 槽位 + 执行属性"""
    intent_type: LLMIntentType
    confidence: float
    reason: str
    slots: dict = field(default_factory=dict)
    slot_sources: dict = field(default_factory=dict)
    missing_slots: list = field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str = ""
    clarification_options: list = field(default_factory=list)
    source: str = ""
    rule_result: Optional[LLMRuleResult] = None
    rule_agreement: bool = True
    # 执行属性（供编排器使用）
    execution_cost: str = "medium"
    dependencies: list = field(default_factory=list)  # List[LLMIntentType]
    can_parallel: bool = True


@dataclass
class MultiIntentResult:
    """多意图识别最终结果"""
    candidates: list = field(default_factory=list)   # List[IntentCandidate]
    primary_intent: Optional[LLMIntentType] = None
    needs_clarification: bool = False
    clarification_reason: Optional[str] = None
    global_slots: dict = field(default_factory=dict)
    execution_topology: list = field(default_factory=list)  # List[List[LLMIntentType]]

@dataclass
class LLMIntentResult:
    intent: LLMIntentType
    confidence: float
    layer: str
    rule_result: Optional[LLMRuleResult]
    calibration_result: Optional[CalibrationResult]
    reason: str
    slots: dict = field(default_factory=dict)
    slot_sources: dict = field(default_factory=dict)
    missing_slots: list = field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str = ""
    candidate_options: list = field(default_factory=list)


# ═══════════════════════════════════════════════════════
# 3. 规则注册表基础设施
# ═══════════════════════════════════════════════════════

RuleFunc = Callable[[str, list, list, list, list], Optional[LLMRuleResult]]


class LLMRuleRegistry:
    def __init__(self):
        self._rules: List[tuple[int, str, RuleFunc]] = []
        self._hit_counter: dict[str, int] = {}

    def register(self, priority: int, rule_id: str, func: RuleFunc):
        self._rules.append((priority, rule_id, func))
        self._rules.sort(key=lambda x: x[0])
        if rule_id not in self._hit_counter:
            self._hit_counter[rule_id] = 0

    def classify(self, message: str, attachments: list) -> LLMRuleResult:
        kb_companies, kb_positions = _load_kb_entities()
        for priority, rule_id, func in self._rules:
            result = func(message, attachments or [], [], kb_companies, kb_positions)
            if result:
                self._hit_counter[rule_id] += 1
                logger.info(
                    f"[LLMRuleHit] {rule_id} | msg='{message[:40]}...' | "
                    f"intent={result.intent.value if result.intent else 'None'} | "
                    f"strength={result.strength}"
                )
                return result
        return LLMRuleResult(
            intent=None, strength=RuleStrength.MISS, rule_name="MISS", trigger="", metadata={},
        )

    def classify_all(self, message: str, attachments: list, fallback_message: str = None) -> list:
        """
        多意图改造：收集所有命中规则，返回 List[LLMRuleResult]。
        按 strength(STRONG>WEAK) 和 priority 排序。
        
        Args:
            message: 原始消息（用于规则匹配）
            attachments: 附件列表
            fallback_message: 改写后的消息（当原始消息无匹配时，用此做二次匹配）
        """
        kb_companies, kb_positions = _load_kb_entities()
        matches = []
        seen_intents = set()

        def _run_rules(msg: str, is_fallback: bool = False):
            for priority, rule_id, func in self._rules:
                result = func(msg, attachments or [], [], kb_companies, kb_positions)
                if result and result.intent:
                    self._hit_counter[rule_id] += 1
                    # 去重：同一意图只保留第一个（优先级更高的）
                    if result.intent not in seen_intents:
                        seen_intents.add(result.intent)
                        matches.append(result)
                        logger.info(
                            f"[LLMRuleHit] {rule_id} | msg='{msg[:40]}...' | "
                            f"intent={result.intent.value if result.intent else 'None'} | "
                            f"strength={result.strength}{' | fallback' if is_fallback else ''}"
                        )

        # 第一轮：用原始消息匹配
        _run_rules(message)

        # 第二轮：若原始消息无匹配，且 fallback_message 不同，则用 fallback_message 二次匹配
        if not any(r.intent for r in matches) and fallback_message and fallback_message != message:
            logger.info(f"[LLMRuleRegistry] 原始消息无匹配，尝试改写消息二次匹配")
            _run_rules(fallback_message, is_fallback=True)

        # 排序：STRONG > WEAK > MISS
        strength_order = {RuleStrength.STRONG: 0, RuleStrength.WEAK: 1, RuleStrength.MISS: 2}
        matches.sort(key=lambda r: (strength_order.get(r.strength, 99), r.rule_name))

        if not matches:
            matches.append(LLMRuleResult(
                intent=None, strength=RuleStrength.MISS, rule_name="MISS", trigger="", metadata={},
            ))
        return matches

    def classify(self, message: str, attachments: list) -> LLMRuleResult:
        """单意图兼容接口：返回 classify_all 的第一个结果"""
        all_matches = self.classify_all(message, attachments)
        return all_matches[0]

    def get_stats(self) -> dict:
        return {"total_hits": sum(self._hit_counter.values()), "by_rule": self._hit_counter.copy()}


# ═══════════════════════════════════════════════════════
# 4. 知识库已知实体
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
    "数据分析师", "运营", "设计师", "AI产品经理", "产品实习生",
]


def _load_kb_entities() -> tuple[list[str], list[str]]:
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
                    # 添加常见简称映射
                    if "产品经理" in p and "产品岗" not in positions:
                        positions.add("产品岗")
                    if "AI产品经理" in p and "AI产品岗" not in positions:
                        positions.add("AI产品岗")
                    if "后端" in p and "后端" not in positions:
                        positions.add("后端")
                    if "Java" in p and "Java" not in positions:
                        positions.add("Java")
    except Exception:
        pass
    return list(companies), list(positions)


def _extract_basic_slots(msg: str, intent: LLMIntentType, kb_companies: list, kb_positions: list) -> dict:
    """规则层基础实体提取：根据意图类型提取 company/position/attributes 等"""
    slots = {}
    # 公司和岗位（ASSESS/VERIFY/PREPARE/EXPLORE 都可能需要）
    # 按长度降序匹配，优先命中更长的实体（如 "字节跳动" 优先于 "字节"）
    if intent in (LLMIntentType.ASSESS, LLMIntentType.VERIFY, LLMIntentType.PREPARE, LLMIntentType.EXPLORE):
        company = next((c for c in sorted(kb_companies, key=len, reverse=True) if c in msg), None)
        position = next((p for p in sorted(kb_positions, key=len, reverse=True) if p in msg), None)
        if company:
            slots["company"] = company
        if position:
            slots["position"] = position
    # VERIFY 专属：属性词
    if intent == LLMIntentType.VERIFY:
        ATTR_WORDS = ["薪资", "工资", "要求", "学历", "经验", "技术栈", "福利", "加班", "地点", "区别", "怎么样"]
        attrs = [w for w in ATTR_WORDS if w in msg]
        if attrs:
            slots["attributes"] = attrs
            slots["qa_type"] = "factual"
    # EXPLORE 专属：搜索关键词
    if intent == LLMIntentType.EXPLORE:
        slots["search_keywords"] = msg[:50]
        slots["sort_by"] = "match_score"
        slots["top_k"] = 5
    # ASSESS 专属
    if intent == LLMIntentType.ASSESS:
        slots["attributes"] = ["匹配度"]
        slots["jd_source"] = "kb"
    return slots


# ═══════════════════════════════════════════════════════
# 5. L1 结构特征层（priority=10~14）
# ═══════════════════════════════════════════════════════

def rule_l1_attachment_assess(msg, attachments, ctx, kb_companies, kb_positions):
    if not attachments or len(attachments) == 0:
        return None
    ASSESS_KWS = ["这家", "这个", "该", "匹配", "适合", "差距", "差多少", "契合", "够格", "搭不搭", "我能去", "分析一下"]
    if any(kw in msg for kw in ASSESS_KWS):
        hit = next(kw for kw in ASSESS_KWS if kw in msg)
        return LLMRuleResult(
            intent=LLMIntentType.ASSESS, strength=RuleStrength.STRONG,
            rule_name="L1-附件+单实体评估", trigger=hit,
            metadata={"attachment_count": len(attachments)},
        )
    return None


def rule_l1_attachment_verify(msg, attachments, ctx, kb_companies, kb_positions):
    if not attachments or len(attachments) == 0:
        return None
    VERIFY_KWS = ["要求什么", "技能", "薪资", "福利", "学历", "职责", "岗位介绍"]
    if any(kw in msg for kw in VERIFY_KWS):
        hit = next(kw for kw in VERIFY_KWS if kw in msg)
        return LLMRuleResult(
            intent=LLMIntentType.VERIFY, strength=RuleStrength.STRONG,
            rule_name="L1-附件+核实词", trigger=hit,
            metadata={"attachment_count": len(attachments)},
        )
    return None


def rule_l1_attachment_prepare(msg, attachments, ctx, kb_companies, kb_positions):
    if not attachments or len(attachments) == 0:
        return None
    PREPARE_KWS = ["面试", "题目", "模拟", "怎么准备", "会问什么", "刁钻", "押题"]
    if any(kw in msg for kw in PREPARE_KWS):
        hit = next(kw for kw in PREPARE_KWS if kw in msg)
        return LLMRuleResult(
            intent=LLMIntentType.PREPARE, strength=RuleStrength.STRONG,
            rule_name="L1-附件+准备词", trigger=hit,
            metadata={"attachment_count": len(attachments)},
        )
    return None


def rule_l1_attachment_manage(msg, attachments, ctx, kb_companies, kb_positions):
    if not attachments or len(attachments) == 0:
        return None
    MANAGE_KWS = ["上传", "更新", "删除", "替换", "保存", "清空"]
    if any(kw in msg for kw in MANAGE_KWS):
        hit = next(kw for kw in MANAGE_KWS if kw in msg)
        return LLMRuleResult(
            intent=LLMIntentType.MANAGE, strength=RuleStrength.STRONG,
            rule_name="L1-附件+管理词", trigger=hit,
            metadata={"attachment_count": len(attachments)},
        )
    return None


def rule_l1_jd_long_text(msg, attachments, ctx, kb_companies, kb_positions):
    JD_MARKERS = [
        "岗位职责", "任职要求", "硬性要求", "软性要求", "加分项",
        "薪资范围", "工作地点", "五险一金", "带薪年假", "简历投递",
        "工作职责", "任职资格", "岗位描述", "职位要求",
    ]
    if len(msg) > 200 and any(marker in msg for marker in JD_MARKERS):
        return LLMRuleResult(
            intent=LLMIntentType.ASSESS, strength=RuleStrength.STRONG,
            rule_name="L1-JD长文本", trigger="jd_markers",
            metadata={"text_length": len(msg)},
        )
    return None


# ═══════════════════════════════════════════════════════
# 6. L2 强关键词层（priority=20~25）
# ═══════════════════════════════════════════════════════

def rule_l2_global_explore(msg, attachments, ctx, kb_companies, kb_positions):
    if attachments and len(attachments) > 0:
        return None
    RANGE_KWS = ["所有", "全局", "哪家", "排序", "适合我", "投哪些", "投递哪些", "帮我选", "对比几家", "筛几个", "筛选", "推几个", "挑几个", "看看有什么", "有什么", "帮我找", "帮我看看", "有用的", "合适的", "有价值的", "分析一遍", "对哪个岗", "哪个岗", "相关", "相关岗位"]
    has_range = any(kw in msg for kw in RANGE_KWS)
    ATTR_WORDS = ["薪资", "工资", "要求", "技能", "学历", "福利"]
    has_attr = any(w in msg for w in ATTR_WORDS)
    if has_attr and not has_range:
        return None
    if has_range:
        hit = next(kw for kw in RANGE_KWS if kw in msg)
        slots = _extract_basic_slots(msg, LLMIntentType.EXPLORE, kb_companies, kb_positions)
        return LLMRuleResult(
            intent=LLMIntentType.EXPLORE, strength=RuleStrength.STRONG,
            rule_name="L2-全局探索", trigger=hit, metadata=slots,
        )
    if "对比" in msg:
        has_entity = any(e in msg for e in kb_companies + kb_positions)
        if has_entity and has_attr:
            return None
        slots = _extract_basic_slots(msg, LLMIntentType.EXPLORE, kb_companies, kb_positions)
        return LLMRuleResult(
            intent=LLMIntentType.EXPLORE, strength=RuleStrength.STRONG,
            rule_name="L2-全局探索", trigger="对比", metadata=slots,
        )
    return None


def rule_l2_greeting(msg, attachments, ctx, kb_companies, kb_positions):
    GREETINGS = ["你好", "您好", "嗨", "hello", "hi", "在吗", "早上好", "晚上好", "哈喽", "打扰了", "谢谢", "感谢", "再见"]
    trimmed = msg.strip().lower()
    if trimmed in [g.lower() for g in GREETINGS]:
        return LLMRuleResult(
            intent=LLMIntentType.CHAT, strength=RuleStrength.STRONG,
            rule_name="L2-问候", trigger=trimmed, metadata={"length": len(trimmed)},
        )
    return None


def rule_l2_prepare(msg, attachments, ctx, kb_companies, kb_positions):
    if attachments and len(attachments) > 0:
        return None
    PREPARE_KWS = ["面试题", "模拟面试", "会问什么", "怎么准备", "刁钻", "押题", "准备什么", "问什么", "要准备"]
    if any(kw in msg for kw in PREPARE_KWS):
        hit = next(kw for kw in PREPARE_KWS if kw in msg)
        return LLMRuleResult(
            intent=LLMIntentType.PREPARE, strength=RuleStrength.STRONG,
            rule_name="L2-面试准备", trigger=hit, metadata={},
        )
    return None


def rule_l2_attr_verify(msg, attachments, ctx, kb_companies, kb_positions):
    ATTR_WORDS = ["薪资", "工资", "要求", "学历", "经验", "技术栈", "福利", "加班", "地点", "区别", "怎么样"]
    has_attr = any(w in msg for w in ATTR_WORDS)
    if has_attr:
        hit = next(w for w in ATTR_WORDS if w in msg)
        # 多意图场景下，即使同时存在匹配度词，也允许 VERIFY 被识别（强度降为 WEAK）
        STRONG_MATCH_KWS = ["匹配度", "差距", "差多少", "契合", "够格"]
        strength = RuleStrength.WEAK if any(kw in msg for kw in STRONG_MATCH_KWS) else RuleStrength.STRONG
        slots = _extract_basic_slots(msg, LLMIntentType.VERIFY, kb_companies, kb_positions)
        return LLMRuleResult(
            intent=LLMIntentType.VERIFY, strength=strength,
            rule_name="L2-属性核实", trigger=hit, metadata=slots,
        )
    return None


def rule_l2_introduce_verify(msg, attachments, ctx, kb_companies, kb_positions):
    """岗位综合分析/介绍：分析/介绍一下/了解一下 + 实体 → VERIFY（综合查询）"""
    if attachments and len(attachments) > 0:
        return None
    INTRODUCE_KWS = ["分析一下", "介绍一下", "分析", "介绍", "了解一下", "说说", "讲讲"]
    has_introduce = any(kw in msg for kw in INTRODUCE_KWS)
    has_entity = any(e in msg for e in kb_companies + kb_positions)
    # 排除已被 ASSESS 规则覆盖的匹配度场景
    ASSESS_KWS = ["匹配度", "差距", "差多少", "契合", "够格", "适合吗", "搭不搭"]
    if has_introduce and has_entity and not any(kw in msg for kw in ASSESS_KWS):
        hit = next(kw for kw in INTRODUCE_KWS if kw in msg)
        slots = _extract_basic_slots(msg, LLMIntentType.VERIFY, kb_companies, kb_positions)
        # 综合查询模式：attributes 默认填充为综合情况
        if not slots.get("attributes"):
            slots["attributes"] = ["综合情况"]
        return LLMRuleResult(
            intent=LLMIntentType.VERIFY, strength=RuleStrength.WEAK,
            rule_name="L2-岗位介绍", trigger=hit, metadata=slots,
        )
    return None


def rule_l2_manage(msg, attachments, ctx, kb_companies, kb_positions):
    MANAGE_KWS = ["上传简历", "更新JD", "删除", "清空", "我的简历库", "列出"]
    if any(kw in msg for kw in MANAGE_KWS):
        hit = next(kw for kw in MANAGE_KWS if kw in msg)
        return LLMRuleResult(
            intent=LLMIntentType.MANAGE, strength=RuleStrength.STRONG,
            rule_name="L2-管理操作", trigger=hit, metadata={},
        )
    return None


def rule_l2_general_chat(msg, attachments, ctx, kb_companies, kb_positions):
    GENERAL_KWS = ["职业规划", "行业趋势", "如何转行", "简历优化", "建议"]
    if len(msg) < 80 and any(kw in msg for kw in GENERAL_KWS):
        hit = next(kw for kw in GENERAL_KWS if kw in msg)
        return LLMRuleResult(
            intent=LLMIntentType.CHAT, strength=RuleStrength.WEAK,
            rule_name="L2-通用咨询", trigger=hit, metadata={},
        )
    return None


# ═══════════════════════════════════════════════════════
# 7. L3 组合模式层（priority=30~33）
# ═══════════════════════════════════════════════════════

def rule_l3_referenced_explore(msg, attachments, ctx, kb_companies, kb_positions):
    if attachments and len(attachments) > 0:
        return None
    has_company = any(c in msg for c in kb_companies)
    EXPLORE_KWS = ["还有哪些", "也适合", "别的", "对比几家"]
    has_explore = any(kw in msg for kw in EXPLORE_KWS)
    if has_company and has_explore:
        hit = next(kw for kw in EXPLORE_KWS if kw in msg)
        slots = _extract_basic_slots(msg, LLMIntentType.EXPLORE, kb_companies, kb_positions)
        return LLMRuleResult(
            intent=LLMIntentType.EXPLORE, strength=RuleStrength.WEAK,
            rule_name="L3-引用式探索", trigger=hit,
            metadata=slots,
        )
    return None


def rule_l3_referenced_assess(msg, attachments, ctx, kb_companies, kb_positions):
    if attachments and len(attachments) > 0:
        return None
    has_company = any(c in msg for c in kb_companies)
    has_position = any(p in msg for p in kb_positions)
    MATCH_KWS = ["匹配度", "匹配", "适合吗", "适合", "差距", "差多少", "契合", "够格", "能去", "能投", "转行"]
    has_match = any(kw in msg for kw in MATCH_KWS)
    # 有公司名或岗位名 + 匹配度词 → ASSESS
    if (has_company or has_position) and has_match:
        hit = next(kw for kw in MATCH_KWS if kw in msg)
        slots = _extract_basic_slots(msg, LLMIntentType.ASSESS, kb_companies, kb_positions)
        return LLMRuleResult(
            intent=LLMIntentType.ASSESS, strength=RuleStrength.WEAK,
            rule_name="L3-引用式评估", trigger=hit,
            metadata=slots,
        )
    return None


def rule_l3_referenced_verify(msg, attachments, ctx, kb_companies, kb_positions):
    QUESTION_WORDS = ["什么", "多少", "怎么", "吗", "呢", "哪些", "如何", "介绍一下", "怎样", "啥"]
    ATTR_WORDS = ["技能", "薪资", "工资", "要求", "福利", "学历", "经验", "工作年限", "团队规模", "加班", "作息", "技术栈", "条件", "门槛", "待遇", "前景", "发展", "需要", "用到", "偏"]
    has_question = any(w in msg for w in QUESTION_WORDS)
    has_attr = any(w in msg for w in ATTR_WORDS)
    has_entity = any(e in msg for e in kb_companies + kb_positions)
    if has_question and has_attr and has_entity:
        slots = _extract_basic_slots(msg, LLMIntentType.VERIFY, kb_companies, kb_positions)
        slots["mode"] = "question"
        return LLMRuleResult(
            intent=LLMIntentType.VERIFY, strength=RuleStrength.WEAK,
            rule_name="L3-引用式核实", trigger="question+attr+entity",
            metadata=slots,
        )
    ends_with_attr = any(msg.strip().endswith(w) for w in ATTR_WORDS)
    if ends_with_attr and has_attr and has_entity and len(msg) < 60:
        slots = _extract_basic_slots(msg, LLMIntentType.VERIFY, kb_companies, kb_positions)
        slots["mode"] = "ending"
        return LLMRuleResult(
            intent=LLMIntentType.VERIFY, strength=RuleStrength.WEAK,
            rule_name="L3-引用式核实", trigger="ending_attr",
            metadata=slots,
        )
    return None


def rule_l3_referenced_prepare(msg, attachments, ctx, kb_companies, kb_positions):
    has_entity = any(e in msg for e in kb_companies + kb_positions)
    PREPARE_KWS = ["面试", "题目", "模拟", "会问什么", "怎么准备", "刁钻", "押题", "准备什么", "问什么", "要准备"]
    has_prepare = any(kw in msg for kw in PREPARE_KWS)
    if has_entity and has_prepare:
        # 多意图场景：不再因为存在 assess 词而跳过 PREPARE
        hit = next(kw for kw in PREPARE_KWS if kw in msg)
        return LLMRuleResult(
            intent=LLMIntentType.PREPARE, strength=RuleStrength.WEAK,
            rule_name="L3-引用式准备", trigger=hit, metadata={},
        )
    return None


def rule_l3_existence_verify(msg, attachments, ctx, kb_companies, kb_positions):
    """存在性核实：公司 + '有...岗吗/有没有' → VERIFY（询问岗位是否存在）"""
    if attachments and len(attachments) > 0:
        return None
    has_company = any(c in msg for c in kb_companies)
    EXIST_KWS = ["有没有", "有", "存在"]
    has_exist = any(kw in msg for kw in EXIST_KWS)
    # 必须包含 '有'/'有没有'/'存在' 且包含疑问词，才认定为存在性询问
    QUESTION_MARKERS = ["吗", "么", "没", "不"]
    has_question = any(q in msg for q in QUESTION_MARKERS)
    if has_company and has_exist and has_question:
        slots = _extract_basic_slots(msg, LLMIntentType.VERIFY, kb_companies, kb_positions)
        return LLMRuleResult(
            intent=LLMIntentType.VERIFY, strength=RuleStrength.WEAK,
            rule_name="L3-存在性核实", trigger="有/有没有/存在+疑问",
            metadata=slots,
        )
    return None


# ═══════════════════════════════════════════════════════
# 8. 规则分类器初始化
# ═══════════════════════════════════════════════════════

def _create_rule_registry() -> LLMRuleRegistry:
    registry = LLMRuleRegistry()
    registry.register(10, "L1-附件+单实体评估", rule_l1_attachment_assess)
    registry.register(11, "L1-附件+核实词", rule_l1_attachment_verify)
    registry.register(12, "L1-附件+准备词", rule_l1_attachment_prepare)
    registry.register(13, "L1-附件+管理词", rule_l1_attachment_manage)
    registry.register(14, "L1-JD长文本", rule_l1_jd_long_text)
    registry.register(20, "L2-全局探索", rule_l2_global_explore)
    registry.register(21, "L2-问候", rule_l2_greeting)
    registry.register(22, "L2-面试准备", rule_l2_prepare)
    registry.register(23, "L2-属性核实", rule_l2_attr_verify)
    registry.register(23, "L2-岗位介绍", rule_l2_introduce_verify)
    registry.register(24, "L2-管理操作", rule_l2_manage)
    registry.register(25, "L2-通用咨询", rule_l2_general_chat)
    registry.register(30, "L3-引用式探索", rule_l3_referenced_explore)
    registry.register(31, "L3-引用式评估", rule_l3_referenced_assess)
    registry.register(32, "L3-引用式核实", rule_l3_referenced_verify)
    registry.register(33, "L3-引用式准备", rule_l3_referenced_prepare)
    registry.register(34, "L3-存在性核实", rule_l3_existence_verify)
    return registry


# ═══════════════════════════════════════════════════════
# 9. 小模型校准器
# ═══════════════════════════════════════════════════════

INTENT_CALIBRATION_SYSTEM = """你是一位意图与槽位联合校准专家。基于【规则引擎参考】和【三层记忆】，同时完成意图判断和槽位抽取，并评估置信度。

【六类意图】
- EXPLORE（探索选岗）：从多个岗位中筛选、排序、推荐，如"适合投哪些"、"帮我排序"、"对比几家"
- ASSESS（评估匹配）：分析自己与某个具体岗位的匹配度，如"我和字节匹配吗"、"差距大吗"
- VERIFY（核实细节）：查询某个岗位的具体属性信息，如"薪资多少"、"要求什么学历"
- PREPARE（面试准备）：为面试做准备，如"生成面试题"、"模拟面试"
- MANAGE（管理资料）：操作文件/数据，如"上传简历"、"删除JD"
- CHAT（闲聊咨询）：闲聊、打招呼、通用建议，如"你好"、"职业规划建议"

【槽位 Schema（按意图）】

1. EXPLORE:
   - resume_available: boolean（系统是否有生效简历）
   - filters: {{skills:[string], location:string|null, experience_years:string|null}}
   - sort_by: enum("match_score"|"salary"|"company_size")，默认"match_score"
   - top_k: integer，默认5
   - search_keywords: string

2. ASSESS:
   - company: string|null（必须从query或历史提取，禁止编造）
   - position: string|null
   - jd_source: enum("kb"|"attachment"|"text")，默认"kb"
   - attributes: [string]，默认["匹配度"]
   - resume_available: boolean
   - search_keywords: string

3. VERIFY:
   - company: string|null
   - position: string|null
   - attributes: [string]（至少一个）
   - qa_type: enum("factual"|"comparative"|"temporal"|"definition")
   - search_keywords: string

4. PREPARE:
   - company: string|null（可为null，若有历史match_result）
   - position: string|null
   - has_history_match: boolean（是否有上轮ASSESS结果）
   - focus_area: enum("gap"|"strength"|"general")，默认"gap"
   - count: integer，默认5
   - difficulty: enum("easy"|"medium"|"hard"|"mixed")，默认"mixed"

5. MANAGE:
   - operation: enum("upload_jd"|"upload_jd_image"|"delete_jd"|"update_resume"|"list_jds"|"list_resumes")
   - 注意：简历上传不在对话内处理，请引导用户前往「我的简历」页面
   - file_data: string|null（base64或文件ID）
   - text_data: string|null（粘贴文本）
   - target_id: string|null

6. CHAT:
   - general_type: enum("greeting"|"career_advice"|"industry"|"how_to"|"other")
   - topic_hint: string|null

【槽位补全优先级】
当query未直接提及某槽位时：
1. 工作记忆（最近3轮）→ 2. 压缩记忆（4-10轮摘要）→ 3. 长期记忆（用户画像）→ 4. 附件元数据 → 5. 系统状态
无法补全则标记为null。

【输入信息】
1. 【规则引擎参考】
   - 规则判定意图：{rule_intent}
   - 规则强度：{rule_strength}（STRONG/WEAK/MISS）
   - 触发规则：{rule_name}

2. 【改写后查询】
   - rewritten_query：{rewritten_query}
   - 追问类型：{follow_up_type}（expand/switch/clarify/none）

3. 【三层记忆】
   工作记忆：
   {working_history}

   压缩记忆：
   {compressed_history}

   长期记忆：
   {long_term_profile}

【校准原则】
1. 独立判断：规则结果仅供参考，三层记忆与规则冲突时以记忆为准
2. 话题连贯优先：上轮ASSESS + 本轮追问"薪资呢" → VERIFY，同时company/position从工作记忆补全
3. 管理意图隔离：涉及上传/删除/更新 → 优先MANAGE
4. 追问类型辅助：
   - expand：通常维持上轮意图，深入细节（如"薪资呢""具体怎么做"）
   - switch：切换话题或实体（如"那看看百度呢""换成后端"）
   - clarify：用户在回应系统澄清，补充缺失槽位。请结合上轮澄清的 pending_intent 和上下文推断真实意图，不要仅基于当前简短输入做表面判断
   - none：独立的新问题
5. 【clarify 场景强制规则】当工作记忆中出现【系统状态】显示"上轮触发澄清"时：
   - 本轮用户输入若补充了缺失的 company 和/或 position，这是用户在回应澄清，意图**绝不能**判为 CHAT
   - 必须结合上轮用户query的实际语义推断真实意图：含"分析""匹配""适合"→ASSESS；含"薪资""要求""加班"→VERIFY
   - 槽位必须从当前query中提取，confidence ≥ 0.80

【强度标签对应策略】
- STRONG：原则上直接确认规则意图和槽位，仅当明显话题切换时修正
- WEAK：必须结合三层记忆做独立语义判断，有权推翻规则并重新抽槽
- MISS：完全依赖语义判断，无规则参考

【输出格式】
严格JSON，不要markdown代码块：

{{
  "intent": "EXPLORE/ASSESS/VERIFY/PREPARE/MANAGE/CHAT",
  "confidence": 0.0-1.0,
  "reason": "意图判断依据 + 槽位抽取依据",
  "slots": {{
    "company": "...",
    "position": "...",
    ...
  }},
  "slot_sources": {{
    "company": "working_memory/current_query/long_term/attachment/system_state/default",
    ...
  }},
  "missing_slots": ["字段名"],
  "needs_clarification": true/false,
  "clarification_question": "若缺失必填槽位，生成澄清问题",
  "clarification_options": ["选项1", "选项2"],
  "rule_agreement": true/false,
  "context_driven": true/false,
  "needs_llm_fallback": true/false
}}

【置信度标准】
- 0.9-1.0：语义明确，槽位完整，三层记忆一致支持
- 0.8-0.89：语义较明确，槽位基本完整，存在轻微歧义
- 0.7-0.79：语义模糊，或关键槽位缺失无法补全，建议大模型兜底
- <0.7：信息严重不足，建议大模型兜底或澄清

【冲突处理】
当规则与你的判断不一致时：
- 有强上下文证据 → 采信你的判断，rule_agreement=false
- 规则STRONG且记忆无矛盾 → 采信规则，rule_agreement=true
- 无法确定 → confidence<0.75，触发needs_llm_fallback=true
"""

INTENT_CALIBRATION_EXAMPLES = """
【示例1：ASSESS + expand + 槽位从记忆补全】
规则参考：intent="ASSESS"，strength="STRONG"，rule_name="附件+单实体评估"
工作记忆：
用户：帮我看看字节跳动算法岗匹配吗
助手：匹配度85%
改写查询：具体哪些技能不匹配
追问类型：expand

输出：
{{"intent":"ASSESS","confidence":0.95,"reason":"规则STRONG命中，工作记忆显示上轮为ASSESS，expand追问维持意图。company='字节跳动'从工作记忆补全，position='算法岗'从工作记忆补全，attributes=['技能不匹配']从当前query提取","slots":{{"company":"字节跳动","position":"算法岗","attributes":["技能不匹配"],"jd_source":"kb","resume_available":true,"search_keywords":"字节跳动 算法岗 技能不匹配"}},"slot_sources":{{"company":"working_memory","position":"working_memory","attributes":"current_query","jd_source":"default","resume_available":"system_state","search_keywords":"current_query"}},"missing_slots":[],"needs_clarification":false,"clarification_question":null,"clarification_options":[],"rule_agreement":true,"context_driven":true,"needs_llm_fallback":false}}

【示例2：VERIFY + extend + 槽位补全】
规则参考：intent="VERIFY"，strength="STRONG"，rule_name="属性核实"
工作记忆：
用户：字节跳动算法岗匹配度多少
助手：匹配度85%
改写查询：那薪资和职级要求呢
追问类型：extend

输出：
{{"intent":"VERIFY","confidence":0.92,"reason":"规则STRONG命中属性词，工作记忆显示上轮讨论字节跳动算法岗，extend追问薪资职级。company和position从工作记忆补全，attributes从当前query提取","slots":{{"company":"字节跳动","position":"算法工程师","attributes":["薪资","职级"],"qa_type":"factual","search_keywords":"字节跳动 算法工程师 薪资 职级"}},"slot_sources":{{"company":"working_memory","position":"working_memory","attributes":"current_query","qa_type":"inferred","search_keywords":"current_query"}},"missing_slots":[],"needs_clarification":false,"clarification_question":null,"clarification_options":[],"rule_agreement":true,"context_driven":true,"needs_llm_fallback":false}}

【示例3：ASSESS + 槽位缺失 + 触发澄清】
规则参考：intent="ASSESS"，strength="WEAK"，rule_name="引用式评估(弱匹配词)"
工作记忆：（空）
压缩记忆：（空）
长期记忆：（空）
改写查询：帮我分析一下匹配度
追问类型：none

输出：
{{"intent":"ASSESS","confidence":0.55,"reason":"规则WEAK命中'分析'，但三层记忆全空，无法补全company和position，query中无实体。ASSESS必填槽位缺失","slots":{{"company":null,"position":null,"attributes":["匹配度"],"jd_source":"kb","resume_available":false,"search_keywords":"匹配度"}},"slot_sources":{{"company":null,"position":null,"attributes":"current_query","jd_source":"default","resume_available":"system_state","search_keywords":"current_query"}},"missing_slots":["company","position","resume_available"],"needs_clarification":true,"clarification_question":"您想分析哪个公司、哪个岗位的匹配度？","clarification_options":["从知识库选择已有JD","上传新的JD图片","先上传简历"],"rule_agreement":true,"context_driven":false,"needs_llm_fallback":false}}

【示例4：clarify + 澄清后意图推断】
规则参考：intent="MISS"，strength="MISS"
工作记忆：
用户：分析一下这个岗
助手：您想了解哪个公司的什么信息？请提供公司名称。
改写查询：蚂蚁集团AI产品经理
追问类型：clarify

输出：
{{"intent":"ASSESS","confidence":0.82,"reason":"规则MISS，但工作记忆显示上轮系统触发澄清（意图模糊），用户本轮clarify回复补充公司名'蚂蚁集团'和岗位'AI产品经理'。结合上下文推断用户在补充ASSESS所需的实体信息，意图为ASSESS。company和position从当前query提取","slots":{{"company":"蚂蚁集团","position":"AI产品经理","attributes":["匹配度"],"jd_source":"kb","resume_available":false,"search_keywords":"蚂蚁集团 AI产品经理"}},"slot_sources":{{"company":"current_query","position":"current_query","attributes":"default","jd_source":"default","resume_available":"system_state","search_keywords":"current_query"}},"missing_slots":[],"needs_clarification":false,"clarification_question":null,"clarification_options":[],"rule_agreement":false,"context_driven":true,"needs_llm_fallback":false}}

【示例5：话题切换 + 槽位从长期记忆补全】
规则参考：intent="ASSESS"，strength="STRONG"，rule_name="附件+单实体评估"
工作记忆：
用户：帮我看看字节跳动算法岗匹配吗
助手：匹配度85%
改写查询：阿里巴巴产品经理岗位介绍一下
追问类型：extend

输出：
{{"intent":"VERIFY","confidence":0.88,"reason":"规则STRONG但工作记忆显示话题从字节跳动算法岗切换至阿里巴巴产品经理，应修正为VERIFY。company='阿里巴巴'从当前query提取，position='产品经理'从当前query提取","slots":{{"company":"阿里巴巴","position":"产品经理","attributes":["岗位介绍"],"qa_type":"factual","search_keywords":"阿里巴巴 产品经理 岗位介绍"}},"slot_sources":{{"company":"current_query","position":"current_query","attributes":"current_query","qa_type":"inferred","search_keywords":"current_query"}},"missing_slots":[],"needs_clarification":false,"clarification_question":null,"clarification_options":[],"rule_agreement":false,"context_driven":true,"needs_llm_fallback":false}}

【示例6：规则MISS + 独立判断 + 槽位完整】
规则参考：intent=null，strength="MISS"
工作记忆：
用户：我适合投哪些公司
助手：推荐字节跳动、腾讯、阿里
改写查询：那字节和腾讯哪家更适合我
追问类型：extend

输出：
{{"intent":"EXPLORE","confidence":0.85,"reason":"规则未命中，但工作记忆显示上轮为EXPLORE（全局推荐），本轮extend追问对比两家，维持探索选岗意图。filters从长期记忆补全","slots":{{"resume_available":true,"filters":{{"skills":["Python","Go"],"location":"北京","experience_years":null}},"sort_by":"match_score","top_k":5,"search_keywords":"字节 腾讯 对比 适合"}},"slot_sources":{{"resume_available":"system_state","filters":"long_term","sort_by":"default","top_k":"default","search_keywords":"current_query"}},"missing_slots":[],"needs_clarification":false,"clarification_question":null,"clarification_options":[],"rule_agreement":false,"context_driven":true,"needs_llm_fallback":false}}
"""


class SmallModelCalibrator:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client

    def _build_memory_context(self, session: SessionMemory) -> tuple[str, str, str]:
        working_history = ""
        if session.working_memory.turns:
            working_history = session.working_memory.get_recent_context(3)
        # 注入 pending_clarification 状态，帮助校准器理解 clarify 场景
        if hasattr(session, "pending_clarification") and session.pending_clarification:
            pc = session.pending_clarification
            working_history += f"\n【系统状态】上轮触发澄清，pending_intent={pc.pending_intent}，缺失槽位={pc.missing_slots}"
        if not working_history.strip():
            working_history = "（空，首轮对话）"

        compressed_history = ""
        if session.compressed_memories:
            recent_cm = session.compressed_memories[-10:]
            lines = []
            for cm in recent_cm:
                lines.append(f"轮次{cm.start_turn}-{cm.end_turn}：{cm.summary[:200]}")
                for fact in cm.key_facts[:3]:
                    lines.append(f"  · {fact}")
            compressed_history = "\n".join(lines)
        if not compressed_history.strip():
            compressed_history = "（空）"

        long_term_profile = ""
        if session.long_term:
            lt = session.long_term
            lines = []
            if lt.entities:
                for k, v in lt.entities.items():
                    if v:
                        if isinstance(v, list):
                            lines.append(f"  {k}：{', '.join(v)}")
                        else:
                            lines.append(f"  {k}：{v}")
            if lt.preferences:
                for k, v in lt.preferences.items():
                    if v:
                        lines.append(f"  {k}偏好：{v}")
            long_term_profile = "\n".join(lines) if lines else "（空）"
        else:
            long_term_profile = "（空）"

        return working_history, compressed_history, long_term_profile

    async def calibrate(self, rule_result: LLMRuleResult, rewrite_result: QueryRewriteResult, session: SessionMemory) -> CalibrationResult:
        if self.llm is None:
            try:
                self.llm = LLMClient.from_config("planner")
            except Exception as e:
                logger.warning(f"[SmallModelCalibrator] planner LLM 不可用，fallback: {e}")
                return self._fallback_calibration(rule_result)

        working_history, compressed_history, long_term_profile = self._build_memory_context(session)

        system_prompt = f"{INTENT_CALIBRATION_SYSTEM}\n\n{INTENT_CALIBRATION_EXAMPLES}"
        rule_intent_str = rule_result.intent.value if rule_result.intent else "null"
        system_prompt = system_prompt.format(
            rule_intent=rule_intent_str,
            rule_strength=rule_result.strength.value,
            rule_name=rule_result.rule_name,
            rewritten_query=rewrite_result.rewritten_query,
            follow_up_type=rewrite_result.follow_up_type,
            working_history=working_history,
            compressed_history=compressed_history,
            long_term_profile=long_term_profile,
        )

        try:
            raw = await self.llm.generate(
                prompt="请基于以上信息做意图校准判断，严格输出 JSON：",
                system=system_prompt,
                temperature=0.1,
                max_tokens=800,
            )
            return self._parse_calibration(raw, rule_result)
        except Exception as e:
            logger.warning(f"[SmallModelCalibrator] 校准失败，fallback: {e}")
            return self._fallback_calibration(rule_result)

    def _parse_calibration(self, raw: str, rule_result: LLMRuleResult) -> CalibrationResult:
        text = raw.strip()
        for marker in ["```json", "```"]:
            if marker in text:
                text = text.replace(marker, "").strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    return self._fallback_calibration(rule_result)
            else:
                return self._fallback_calibration(rule_result)

        intent_str = data.get("intent", "CHAT")
        try:
            intent = LLMIntentType(intent_str)
        except ValueError:
            intent = LLMIntentType.CHAT

        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
        return CalibrationResult(
            intent=intent,
            confidence=confidence,
            reason=data.get("reason", ""),
            slots=data.get("slots", {}),
            slot_sources=data.get("slot_sources", {}),
            missing_slots=data.get("missing_slots", []),
            needs_clarification=bool(data.get("needs_clarification", False)),
            clarification_question=data.get("clarification_question", ""),
            clarification_options=data.get("clarification_options", []),
            rule_agreement=bool(data.get("rule_agreement", True)),
            context_driven=bool(data.get("context_driven", False)),
            needs_llm_fallback=bool(data.get("needs_llm_fallback", False)),
        )

    def _fallback_calibration(self, rule_result: LLMRuleResult) -> CalibrationResult:
        if rule_result.intent:
            return CalibrationResult(
                intent=rule_result.intent,
                confidence=0.75 if rule_result.strength == RuleStrength.STRONG else 0.60,
                reason=f"校准失败，fallback 到规则结果: {rule_result.rule_name}",
                slots={},
                slot_sources={},
                missing_slots=[],
                needs_clarification=False,
                clarification_question="",
                clarification_options=[],
                rule_agreement=True,
                context_driven=False,
                needs_llm_fallback=rule_result.strength != RuleStrength.STRONG,
            )
        return CalibrationResult(
            intent=LLMIntentType.CHAT, confidence=0.0,
            reason="规则未命中且校准失败",
            slots={}, slot_sources={}, missing_slots=[],
            needs_clarification=False, clarification_question="", clarification_options=[],
            rule_agreement=False, context_driven=False, needs_llm_fallback=True,
        )

    # ═══════════════════════════════════════════════════════
    # 多意图改造：新增 calibrate_multi
    # ═══════════════════════════════════════════════════════

    async def calibrate_multi(
        self,
        rule_matches: list,  # List[LLMRuleResult]
        rewrite_result: QueryRewriteResult,
        session: SessionMemory,
    ) -> list:  # List[IntentCandidate]
        """对规则引擎输出的每个候选意图，分别执行槽位抽取与置信度校准"""
        candidates = []
        for rule_result in rule_matches:
            if rule_result.intent is None:
                continue
            # STRONG 规则跳过 Ollama 校准，直接生成候选（节省 25-40s）
            if rule_result.strength == RuleStrength.STRONG:
                candidate = self._strong_rule_candidate(rule_result)
                logger.info(
                    f"[SmallModelCalibrator] STRONG规则跳过校准 | "
                    f"intent={candidate.intent_type.value} | rule={rule_result.rule_name}"
                )
            else:
                candidate = await self._calibrate_single_intent(
                    rule_result=rule_result,
                    rewrite_result=rewrite_result,
                    session=session,
                )
            candidates.append(candidate)
        # 按置信度降序
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates

    async def _calibrate_single_intent(
        self,
        rule_result: LLMRuleResult,
        rewrite_result: QueryRewriteResult,
        session: SessionMemory,
    ) -> IntentCandidate:
        """针对单一意图的校准流程（复用现有 prompt 和解析逻辑）"""
        if self.llm is None:
            try:
                self.llm = LLMClient.from_config("planner")
            except Exception as e:
                logger.warning(f"[SmallModelCalibrator] planner LLM 不可用，fallback: {e}")
                return self._fallback_single_intent(rule_result)

        working_history, compressed_history, long_term_profile = self._build_memory_context(session)
        system_prompt = f"{INTENT_CALIBRATION_SYSTEM}\n\n{INTENT_CALIBRATION_EXAMPLES}"
        rule_intent_str = rule_result.intent.value if rule_result.intent else "null"
        system_prompt = system_prompt.format(
            rule_intent=rule_intent_str,
            rule_strength=rule_result.strength.value,
            rule_name=rule_result.rule_name,
            rewritten_query=rewrite_result.rewritten_query,
            follow_up_type=rewrite_result.follow_up_type,
            working_history=working_history,
            compressed_history=compressed_history,
            long_term_profile=long_term_profile,
        )

        # ═══════════════════════════════════════════════════════
        # 多意图场景：追加意图隔离提示
        # ═══════════════════════════════════════════════════════
        multi_intent_hint = f"""
【多意图隔离提示】
当前用户query可能包含多个意图。本次校准仅针对【{rule_intent_str}】意图。
请忽略query中与其他意图相关的部分，只为【{rule_intent_str}】意图抽取槽位。
特别注意：
1. search_keywords 应仅包含与【{rule_intent_str}】相关的关键词，不要混入其他意图的词
2. attributes 只应包含属于【{rule_intent_str}】的属性，不要包含其他意图的属性
3. 如果query中某句话明显属于其他意图（如"再帮我...""另外..."），请忽略该部分
本意图的触发关键词：{rule_result.trigger or "无"}
"""
        system_prompt += multi_intent_hint

        try:
            raw = await self.llm.generate(
                prompt=f"请基于以上信息对该意图（{rule_intent_str}）做槽位联合抽取与校准，严格输出 JSON：",
                system=system_prompt,
                temperature=0.1,
                max_tokens=800,
            )
            return self._parse_single_intent(raw, rule_result, rewrite_result)
        except Exception as e:
            logger.warning(f"[SmallModelCalibrator] 单意图校准失败: {e}")
            return self._fallback_single_intent(rule_result)

    def _parse_single_intent(self, raw: str, rule_result: LLMRuleResult, rewrite_result: QueryRewriteResult) -> IntentCandidate:
        """解析单意图校准结果为 IntentCandidate"""
        text = raw.strip()
        for marker in ["```json", "```"]:
            if marker in text:
                text = text.replace(marker, "").strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    return self._fallback_single_intent(rule_result)
            else:
                return self._fallback_single_intent(rule_result)

        intent_str = data.get("intent", "CHAT")
        try:
            intent = LLMIntentType(intent_str)
        except ValueError:
            intent = LLMIntentType.CHAT

        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
        rule_agreement = bool(data.get("rule_agreement", True))
        
        # 后处理1：若模型声称同意规则但意图不一致，强制修正为规则意图
        if rule_agreement and rule_result.intent and intent != rule_result.intent:
            logger.warning(
                f"[SmallModelCalibrator] 校准意图修正 | rule={rule_result.intent.value} | "
                f"calibrated={intent.value} | 强制回退到规则意图"
            )
            intent = rule_result.intent
            confidence = min(confidence, 0.82)  # 修正后降低置信度，提示需关注
        
        # 后处理2：防止模型过度保守 fallback 到 CHAT
        # 规则明确识别为非 CHAT 意图，但校准模型保守地改成了 CHAT → 信任规则
        if (
            rule_result.intent
            and rule_result.intent != LLMIntentType.CHAT
            and intent == LLMIntentType.CHAT
            and confidence >= 0.80
        ):
            logger.warning(
                f"[SmallModelCalibrator] 防止过度保守fallback | rule={rule_result.intent.value} | "
                f"calibrated=chat | 强制回退到规则意图"
            )
            intent = rule_result.intent
            confidence = min(confidence, 0.78)
            rule_agreement = True
        
        # 映射执行属性
        cost_map = {
            LLMIntentType.ASSESS: "high", LLMIntentType.EXPLORE: "high",
            LLMIntentType.PREPARE: "medium", LLMIntentType.VERIFY: "medium",
            LLMIntentType.MANAGE: "low", LLMIntentType.CHAT: "low",
        }
        dep_map = {
            LLMIntentType.PREPARE: [LLMIntentType.ASSESS],
            LLMIntentType.EXPLORE: [LLMIntentType.ASSESS],
        }
        parallel_map = {
            LLMIntentType.VERIFY: True, LLMIntentType.CHAT: True, LLMIntentType.MANAGE: False,
        }

        return IntentCandidate(
            intent_type=intent,
            confidence=confidence,
            reason=data.get("reason", ""),
            slots=data.get("slots", {}),
            slot_sources=data.get("slot_sources", {}),
            missing_slots=data.get("missing_slots", []),
            needs_clarification=bool(data.get("needs_clarification", False)),
            clarification_question=data.get("clarification_question", ""),
            clarification_options=data.get("clarification_options", []),
            source="rule_strong" if rule_result.strength == RuleStrength.STRONG else "rule_weak",
            rule_result=rule_result,
            rule_agreement=bool(data.get("rule_agreement", True)),
            execution_cost=cost_map.get(intent, "medium"),
            dependencies=dep_map.get(intent, []),
            can_parallel=parallel_map.get(intent, True),
        )

    def _fallback_single_intent(self, rule_result: LLMRuleResult) -> IntentCandidate:
        """校准失败时的单意图 fallback"""
        intent = rule_result.intent or LLMIntentType.CHAT
        return IntentCandidate(
            intent_type=intent,
            confidence=0.75 if rule_result.strength == RuleStrength.STRONG else 0.60,
            reason=f"校准失败，fallback 到规则结果: {rule_result.rule_name}",
            slots=rule_result.metadata or {},
            slot_sources={k: "rule_extraction" for k in (rule_result.metadata or {}).keys()},
            missing_slots=[],
            source="rule_fallback",
            rule_result=rule_result,
            rule_agreement=True,
        )

    def _strong_rule_candidate(self, rule_result: LLMRuleResult) -> IntentCandidate:
        """规则 STRONG 时跳过校准，直接基于规则结果生成候选"""
        intent = rule_result.intent or LLMIntentType.CHAT
        cost_map = {
            LLMIntentType.ASSESS: "high", LLMIntentType.EXPLORE: "high",
            LLMIntentType.PREPARE: "medium", LLMIntentType.VERIFY: "medium",
            LLMIntentType.MANAGE: "low", LLMIntentType.CHAT: "low",
        }
        dep_map = {
            LLMIntentType.PREPARE: [LLMIntentType.ASSESS],
            LLMIntentType.EXPLORE: [LLMIntentType.ASSESS],
        }
        slots = rule_result.metadata or {}
        
        # 根据意图类型和槽位完整度推断 missing_slots
        missing_slots = []
        if intent in (LLMIntentType.ASSESS, LLMIntentType.VERIFY, LLMIntentType.PREPARE):
            if not slots.get("company") and not slots.get("position"):
                missing_slots.extend(["company", "position"])
            elif not slots.get("company"):
                missing_slots.append("company")
            elif not slots.get("position"):
                missing_slots.append("position")
        # VERIFY：attributes 缺失时，如果是综合查询（已有 company/position）则不强制澄清
        if intent == LLMIntentType.VERIFY and not slots.get("attributes"):
            has_entity = bool(slots.get("company") or slots.get("position"))
            if not has_entity:
                missing_slots.append("attributes")
        if intent == LLMIntentType.EXPLORE and not slots.get("search_keywords"):
            missing_slots.append("search_keywords")
        
        return IntentCandidate(
            intent_type=intent,
            confidence=0.88,
            reason=f"规则STRONG跳过校准: {rule_result.rule_name} | trigger={rule_result.trigger}",
            slots=slots,
            slot_sources={k: "rule_extraction" for k in slots.keys()},
            missing_slots=missing_slots,
            needs_clarification=bool(missing_slots),
            clarification_question="",
            clarification_options=[],
            source="rule_strong_fastpath",
            rule_result=rule_result,
            rule_agreement=True,
            execution_cost=cost_map.get(intent, "medium"),
            dependencies=dep_map.get(intent, []),
        )


# ═══════════════════════════════════════════════════════
# 10. 大模型兜底分类器
# ═══════════════════════════════════════════════════════

LLM_FALLBACK_SYSTEM = """你是一位深度意图与槽位联合理解专家。当前输入经过规则引擎和小模型校准后仍存在歧义或槽位缺失，需要你基于完整上下文做最终裁决，并补全/修正槽位。

【六类意图定义】
- EXPLORE：探索选岗、排序推荐、对比多家
- ASSESS：单JD匹配度分析、差距评估
- VERIFY：查询具体属性（薪资/要求/学历/技术栈）
- PREPARE：面试题生成、模拟面试、备考建议
- MANAGE：文件上传/删除/更新、资料管理
- CHAT：问候、闲聊、通用职业咨询

【槽位 Schema（同小模型）】
EXPLORE: resume_available, filters, sort_by, top_k, search_keywords
ASSESS: company, position, jd_source, attributes, resume_available, search_keywords
VERIFY: company, position, attributes, qa_type, search_keywords
PREPARE: company, position, has_history_match, focus_area, count, difficulty
MANAGE: operation, file_data, text_data, target_id
CHAT: general_type, topic_hint

【槽位补全优先级】
1. 工作记忆 → 2. 压缩记忆 → 3. 长期记忆 → 4. 附件元数据 → 5. 系统状态
无法补全则标记为null。

【输入信息】
1. 【规则引擎输出】
   - 意图：{rule_intent}
   - 强度：{rule_strength}
   - 触发规则：{rule_name}

2. 【小模型校准输出】
   - 意图：{calib_intent}
   - 置信度：{calib_confidence}
   - 判断理由：{calib_reason}
   - 槽位：{calib_slots}
   - 缺失槽位：{calib_missing_slots}
   - 是否与规则一致：{rule_agreement}

3. 【完整上下文】
   - 改写查询：{rewritten_query}
   - 追问类型：{follow_up_type}
   - 工作记忆：{working_history}
   - 压缩记忆：{compressed_history}
   - 长期记忆：{long_term_profile}

【你的任务】
1. 综合规则、小模型、三层记忆的所有信息，做出最终意图判断
2. 尽可能从三层记忆中补全缺失槽位
3. 若仍有必填槽位缺失且无法补全，生成澄清问题和选项
4. 输出最终置信度（0.0-1.0），必须基于证据强度诚实评估，不得虚高

【输出格式】
严格JSON，不要markdown代码块：

{{
  "intent": "EXPLORE/ASSESS/VERIFY/PREPARE/MANAGE/CHAT",
  "confidence": 0.0-1.0,
  "reason": "详细推理过程，说明如何处理规则与小模型的冲突，以及如何补全槽位",
  "arbitration": "采信规则/采信小模型/独立判断",
  "slots": {{
    "company": "...",
    "position": "...",
    ...
  }},
  "slot_sources": {{
    "company": "working_memory/current_query/long_term/attachment/system_state/default",
    ...
  }},
  "missing_slots": ["字段名"],
  "needs_clarification": true/false,
  "clarification_question": "若缺失必填槽位，生成问题",
  "clarification_options": ["选项1", "选项2", "选项3"]
}}

【澄清触发条件】（满足任一即触发）
- 最终 confidence < 0.7
- 必填槽位缺失且三层记忆无法补全
- 规则与小模型严重冲突且无法仲裁
- 输入涉及多个可能意图且无法确定主次
"""

LLM_FALLBACK_EXAMPLES = """
【示例1：规则与小模型冲突，大模型仲裁 + 槽位补全】
规则：ASSESS (STRONG)
小模型：VERIFY (0.75)，slots={company:"字节跳动"，position:"算法工程师"，attributes:["薪资"]，qa_type:"factual"}
上下文：用户上轮问"字节跳动算法岗匹配吗"，本轮问"要求什么学历"
推理：规则基于"这家+要求"命中ASSESS，小模型基于"属性词"判为VERIFY。实际上用户是在匹配度框架下询问硬性要求，属于ASSESS的展开型追问。company和position从工作记忆补全，attributes修正为["学历要求"]以匹配用户query。
输出：
{"intent":"ASSESS","confidence":0.82,"reason":"用户在上轮匹配分析框架下追问硬性要求，实质是评估匹配的维度扩展，而非独立的属性查询。小模型的属性词判断过于表面，忽略了话题连贯性。槽位从工作记忆补全company/position，attributes根据当前query修正为['学历要求']","arbitration":"采信规则","slots":{"company":"字节跳动","position":"算法工程师","attributes":["学历要求"],"jd_source":"kb","resume_available":true,"search_keywords":"字节跳动 算法工程师 学历要求"},"slot_sources":{"company":"working_memory","position":"working_memory","attributes":"current_query","jd_source":"default","resume_available":"system_state","search_keywords":"current_query"},"missing_slots":[],"needs_clarification":false,"clarification_question":null,"clarification_options":[]}

【示例2：小模型低置信度，大模型补全槽位】
规则：CHAT (WEAK)
小模型：CHAT (0.65)，slots={general_type:"other"}，missing_slots=[]
上下文：用户问"帮我规划一下"，上轮是ASSESS字节跳动算法岗
推理："规划"一词歧义大，但结合上轮ASSESS上下文，用户更可能问"投递规划"（属于EXPLORE）而非"职业规划"（CHAT）。然而信息不足，无法确定filters等槽位。
输出：
{"intent":"EXPLORE","confidence":0.68,"reason":"'规划'一词歧义大，但结合上轮ASSESS上下文，用户更可能问投递规划。然而当前query未提供任何筛选条件，filters等关键槽位无法确定，信息不足","arbitration":"独立判断","slots":{"resume_available":true,"filters":{"skills":[],"location":null,"experience_years":null},"sort_by":"match_score","top_k":5,"search_keywords":"规划"},"slot_sources":{"resume_available":"system_state","filters":"default","sort_by":"default","top_k":"default","search_keywords":"current_query"},"missing_slots":["filters.skills","search_keywords"],"needs_clarification":true,"clarification_question":"您想基于什么条件进行规划？比如技能、地点、经验年限等","clarification_options":["基于我的技能栈Python和Go推荐","不限条件，全局推荐","先帮我看看字节跳动的匹配度"]}
"""


class LLMFallbackClassifier:
    """大模型兜底分类器：使用 chat 层大模型做最终裁决"""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client

    def _build_memory_context(self, session: SessionMemory) -> tuple[str, str, str]:
        working_history = ""
        if session.working_memory.turns:
            working_history = session.working_memory.get_recent_context(3)
        if not working_history.strip():
            working_history = "（空，首轮对话）"

        compressed_history = ""
        if session.compressed_memories:
            recent_cm = session.compressed_memories[-10:]
            lines = []
            for cm in recent_cm:
                lines.append(f"轮次{cm.start_turn}-{cm.end_turn}：{cm.summary[:200]}")
                for fact in cm.key_facts[:3]:
                    lines.append(f"  · {fact}")
            compressed_history = "\n".join(lines)
        if not compressed_history.strip():
            compressed_history = "（空）"

        long_term_profile = ""
        if session.long_term:
            lt = session.long_term
            lines = []
            if lt.entities:
                for k, v in lt.entities.items():
                    if v:
                        if isinstance(v, list):
                            lines.append(f"  {k}：{', '.join(v)}")
                        else:
                            lines.append(f"  {k}：{v}")
            if lt.preferences:
                for k, v in lt.preferences.items():
                    if v:
                        lines.append(f"  {k}偏好：{v}")
            long_term_profile = "\n".join(lines) if lines else "（空）"
        else:
            long_term_profile = "（空）"

        return working_history, compressed_history, long_term_profile

    async def classify(
        self,
        rule_result: LLMRuleResult,
        cal_result: CalibrationResult,
        rewrite_result: QueryRewriteResult,
        session: SessionMemory,
    ) -> FallbackResult:
        """大模型兜底裁决"""
        if self.llm is None:
            try:
                self.llm = LLMClient.from_config("chat")
            except Exception as e:
                logger.warning(f"[LLMFallbackClassifier] chat LLM 不可用，fallback: {e}")
                return self._fallback(rule_result, cal_result)

        working_history, compressed_history, long_term_profile = self._build_memory_context(session)

        system_prompt = f"{LLM_FALLBACK_SYSTEM}\n\n{LLM_FALLBACK_EXAMPLES}"
        system_prompt = system_prompt.format(
            rule_intent=rule_result.intent.value if rule_result.intent else "null",
            rule_strength=rule_result.strength.value,
            rule_name=rule_result.rule_name,
            calib_intent=cal_result.intent.value,
            calib_confidence=cal_result.confidence,
            calib_reason=cal_result.reason,
            calib_slots=json.dumps(cal_result.slots, ensure_ascii=False) if cal_result.slots else "无",
            calib_missing_slots=json.dumps(cal_result.missing_slots, ensure_ascii=False) if cal_result.missing_slots else "无",
            rule_agreement="是" if cal_result.rule_agreement else "否",
            rewritten_query=rewrite_result.rewritten_query,
            follow_up_type=rewrite_result.follow_up_type,
            working_history=working_history,
            compressed_history=compressed_history,
            long_term_profile=long_term_profile,
        )

        try:
            raw = await self.llm.generate(
                prompt="请基于以上信息做最终意图裁决，严格输出 JSON：",
                system=system_prompt,
                temperature=0.2,
                max_tokens=1200,
            )
            return self._parse_fallback(raw, rule_result, cal_result)
        except Exception as e:
            logger.warning(f"[LLMFallbackClassifier] 兜底失败: {e}")
            return self._fallback(rule_result, cal_result)

    def _parse_fallback(self, raw: str, rule_result: LLMRuleResult, cal_result: CalibrationResult) -> FallbackResult:
        text = raw.strip()
        for marker in ["```json", "```"]:
            if marker in text:
                text = text.replace(marker, "").strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    return self._fallback(rule_result, cal_result)
            else:
                return self._fallback(rule_result, cal_result)

        intent_str = data.get("intent", "CHAT")
        try:
            intent = LLMIntentType(intent_str)
        except ValueError:
            intent = LLMIntentType.CHAT

        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
        needs_clarification = bool(data.get("needs_clarification", False))
        # 如果 confidence < 0.7 强制触发澄清
        if confidence < 0.7:
            needs_clarification = True

        return FallbackResult(
            intent=intent,
            confidence=confidence,
            reason=data.get("reason", ""),
            slots=data.get("slots", {}),
            slot_sources=data.get("slot_sources", {}),
            missing_slots=data.get("missing_slots", []),
            needs_clarification=needs_clarification,
            clarification_question=data.get("clarification_question", ""),
            clarification_options=data.get("clarification_options", []),
            arbitration=data.get("arbitration", "独立判断"),
            candidate_options=data.get("candidate_options", []),
        )

    def _fallback(self, rule_result: LLMRuleResult, cal_result: CalibrationResult) -> FallbackResult:
        """兜底失败时，采信校准结果（或规则结果）"""
        return FallbackResult(
            intent=cal_result.intent,
            confidence=cal_result.confidence,
            reason=f"大模型兜底失败，采信小模型校准结果: {cal_result.reason}",
            slots=cal_result.slots,
            slot_sources=cal_result.slot_sources,
            missing_slots=cal_result.missing_slots,
            needs_clarification=cal_result.confidence < 0.7,
            clarification_question=cal_result.clarification_question,
            clarification_options=cal_result.clarification_options,
            arbitration="采信小模型",
            candidate_options=[],
        )

    # ═══════════════════════════════════════════════════════
    # 多意图改造：新增 arbitrate 多意图仲裁
    # ═══════════════════════════════════════════════════════

    async def _generate_clarification_question(
        self, rewrite_result: QueryRewriteResult, session: SessionMemory
    ) -> str:
        """方案B：当意图完全模糊时，调用chat模型生成友好的澄清问题"""
        if self.llm is None:
            try:
                self.llm = LLMClient.from_config("chat")
            except Exception as e:
                logger.warning(f"[LLMFallbackClassifier] chat LLM 不可用，使用默认澄清: {e}")
                return "抱歉，我没有完全理解您的意思，能再详细说明一下吗？"

        working_history = ""
        if session.working_memory.turns:
            working_history = session.working_memory.get_recent_context(2)

        prompt = f"""用户输入："{rewrite_result.rewritten_query}"

对话历史：
{working_history or "（首轮对话）"}

用户的意图非常模糊，系统无法判断用户想要做什么。
请生成一句**友好、自然、简短**的澄清问题，引导用户明确表达需求。

要求：
1. 不要暴露内部技术术语（如"意图"、"槽位"、"校准"等）
2. 语气友好，像真人顾问
3. 可以给出 2-3 个常见选项供用户选择
4. 控制在 50 字以内

直接输出澄清问题，不要加任何前缀："""

        try:
            raw = await self.llm.generate(
                prompt=prompt,
                system="你是一位专业的求职顾问，善于引导用户表达真实需求。",
                temperature=0.7,
                max_tokens=200,
            )
            q = raw.strip().strip('"').strip("'")
            if q and len(q) > 5:
                return q
        except Exception as e:
            logger.warning(f"[LLMFallbackClassifier] 澄清问题生成失败: {e}")
        return "抱歉，我没有完全理解您的意思，能再详细说明一下吗？"

    async def arbitrate(
        self,
        calibrated_candidates: list,  # List[IntentCandidate]
        rewrite_result: QueryRewriteResult,
        session: SessionMemory,
    ) -> MultiIntentResult:
        """多意图统一仲裁：过滤、冲突消解、拓扑构建、全局槽位合并"""
        # 步骤1: 过滤低置信度
        filtered = [c for c in calibrated_candidates if c.confidence >= 0.5]
        if not filtered:
            # 方案B：调用 LLM 生成友好澄清问题
            clarification_q = await self._generate_clarification_question(rewrite_result, session)
            return MultiIntentResult(
                candidates=[],
                primary_intent=LLMIntentType.CHAT,
                needs_clarification=True,
                clarification_reason=clarification_q,
            )

        # 步骤2: 冲突消解（硬规则）
        resolved = self._resolve_conflicts(filtered)

        # 步骤3: 若候选 > 1 且存在复杂依赖/冲突，调用大模型做最终仲裁
        if len(resolved) > 1 and self._needs_llm_arbitration(resolved):
            resolved = await self._llm_arbitrate(resolved, rewrite_result, session)

        # 步骤4: 构建执行拓扑
        topology = self._build_execution_topology(resolved)

        # 步骤5: 全局槽位池合并
        global_slots = self._merge_global_slots(resolved, session=session)

        # 步骤6: 澄清判断
        needs_clarification, reason = self._check_clarification_need(resolved, global_slots)

        return MultiIntentResult(
            candidates=resolved,
            primary_intent=resolved[0].intent_type if resolved else LLMIntentType.CHAT,
            needs_clarification=needs_clarification,
            clarification_reason=reason,
            global_slots=global_slots,
            execution_topology=topology,
        )

    def _resolve_conflicts(self, candidates: list) -> list:
        """意图冲突消解规则（硬规则，轻量快速）"""
        intent_types = {c.intent_type for c in candidates}

        # 规则1: CHAT + 其他意图 → CHAT降级或移除
        if LLMIntentType.CHAT in intent_types and len(intent_types) > 1:
            chat_c = next((c for c in candidates if c.intent_type == LLMIntentType.CHAT), None)
            if chat_c:
                others_max = max(c.confidence for c in candidates if c.intent_type != LLMIntentType.CHAT)
                if chat_c.confidence < others_max - 0.15:
                    candidates = [c for c in candidates if c.intent_type != LLMIntentType.CHAT]

        # 规则2: MANAGE + 其他意图 → MANAGE优先执行（文件操作先完成）
        if LLMIntentType.MANAGE in intent_types and len(intent_types) > 1:
            manage_c = next((c for c in candidates if c.intent_type == LLMIntentType.MANAGE), None)
            if manage_c:
                manage_c.dependencies = []
                manage_c.can_parallel = False

        # 规则3: ASSESS + PREPARE → 确保 PREPARE 依赖 ASSESS
        if LLMIntentType.ASSESS in intent_types and LLMIntentType.PREPARE in intent_types:
            prepare_c = next((c for c in candidates if c.intent_type == LLMIntentType.PREPARE), None)
            if prepare_c and LLMIntentType.ASSESS not in prepare_c.dependencies:
                prepare_c.dependencies.append(LLMIntentType.ASSESS)
                prepare_c.can_parallel = False

        # 按置信度重新排序
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates

    def _build_execution_topology(self, candidates: list) -> list:
        """基于依赖关系构建分层拓扑，同层可并行"""
        topology = []
        executed = set()
        candidate_map = {c.intent_type: c for c in candidates}

        while len(executed) < len(candidates):
            ready = [
                c.intent_type for c in candidates
                if c.intent_type not in executed
                and all(d in executed for d in c.dependencies)
            ]
            if not ready:
                # 存在循环依赖，强制按置信度执行第一个未执行的
                remaining = [c.intent_type for c in candidates if c.intent_type not in executed]
                ready = remaining[:1]
            topology.append(ready)
            executed.update(ready)

        return topology

    def _merge_global_slots(self, candidates: list, session: SessionMemory = None) -> dict:
        """合并所有候选意图的槽位，去重并保留"""
        global_slots = {}
        # 先注入 session 中的全局槽位（如 resume_available）
        if session and hasattr(session, "global_slots") and session.global_slots:
            for k, v in session.global_slots.items():
                if v is not None:
                    global_slots[k] = v
        for c in candidates:
            for slot_key, slot_val in c.slots.items():
                if slot_val is None:
                    continue
                # 若槽位已存在且新值置信度更高，则覆盖
                # 简化：首次出现保留，后续出现不覆盖（实际可比较 slot_confidence）
                if slot_key not in global_slots:
                    global_slots[slot_key] = slot_val
        return global_slots

    def _check_clarification_need(self, candidates: list, global_slots: dict) -> tuple:
        """多意图场景下的澄清判断"""
        if all(c.confidence < 0.7 for c in candidates):
            return True, "多意图置信度均偏低，需要用户确认具体需求"
        # 只有 ASSESS 严格需要简历；EXPLORE/PREPARE 可以在无简历时工作
        assess_only = [c for c in candidates if c.intent_type == LLMIntentType.ASSESS]
        if assess_only and not global_slots.get("resume_available"):
            return True, "分析匹配度需要简历信息，请先上传简历"
        # VERIFY 缺少 company 和 position 时需要澄清
        verify_candidates = [c for c in candidates if c.intent_type == LLMIntentType.VERIFY]
        for vc in verify_candidates:
            has_company = bool(vc.slots.get("company") or global_slots.get("company"))
            has_position = bool(vc.slots.get("position") or global_slots.get("position"))
            if not has_company and not has_position:
                return True, "您想了解哪个公司的什么信息？请提供公司名称。"
        if len(candidates) > 3:
            return True, "检测到多个意图，请确认优先级"
        return False, None

    def _needs_llm_arbitration(self, candidates: list) -> bool:
        """判断是否需要大模型介入仲裁"""
        if len(candidates) <= 1:
            return False
        has_deps = any(len(c.dependencies) > 0 for c in candidates)
        has_cost_conflict = len([c for c in candidates if c.execution_cost == "high"]) >= 2
        # 存在意图冲突（如 CHAT + ASSESS 同时存在但 CHAT 未被移除）
        has_chat_conflict = LLMIntentType.CHAT in {c.intent_type for c in candidates}
        return has_deps or has_cost_conflict or has_chat_conflict

    async def _llm_arbitrate(self, candidates: list, rewrite_result: QueryRewriteResult, session: SessionMemory) -> list:
        """大模型最终仲裁（仅复杂场景触发）"""
        working_history, compressed_history, long_term_profile = self._build_memory_context(session)
        prompt = f"""
用户Query: {rewrite_result.rewritten_query}
候选意图列表（已校准）:
{json.dumps([{"intent": c.intent_type.value, "confidence": c.confidence, "slots": c.slots, "cost": c.execution_cost, "dependencies": [d.value for d in c.dependencies]} for c in candidates], ensure_ascii=False, indent=2)}

请做最终仲裁：
1. 判断哪些意图是用户真实意图，哪些可能是误触发
2. 确定执行优先级顺序（考虑依赖关系和成本）
3. 若某意图槽位严重不足且无法补全，建议移除或触发澄清

输出严格JSON: {{"selected_intents": ["assess", "prepare"], "execution_order": [["assess"], ["prepare"]], "reasoning": "..."}}
"""
        try:
            raw = await self.llm.generate(
                prompt=prompt,
                system="你是一位多意图仲裁专家。请基于候选意图的置信度、槽位完整度和执行成本，筛选出用户真正想要的意图并确定执行顺序。输出严格JSON。",
                temperature=0.2,
                max_tokens=800,
            )
            text = raw.strip()
            for marker in ["```json", "```"]:
                if marker in text:
                    text = text.replace(marker, "").strip()
            data = json.loads(text)
            selected_names = data.get("selected_intents", [])
            selected = [c for c in candidates if c.intent_type.value in selected_names]
            order_map = {name: idx for idx, group in enumerate(data.get("execution_order", [])) for name in group}
            selected.sort(key=lambda c: order_map.get(c.intent_type.value, 999))
            return selected if selected else candidates
        except Exception as e:
            logger.warning(f"[LLMFallbackClassifier] LLM仲裁失败，fallback: {e}")
            return candidates


# ═══════════════════════════════════════════════════════
# 11. LLM 意图路由（三级流水线入口）
# ═══════════════════════════════════════════════════════

class LLMIntentRouter:
    """LLM 路线意图路由统一入口"""

    def __init__(self, planner_llm: Optional[LLMClient] = None, chat_llm: Optional[LLMClient] = None):
        self.rule_registry = _create_rule_registry()
        self.calibrator = SmallModelCalibrator(llm_client=planner_llm)
        self.fallback = LLMFallbackClassifier(llm_client=chat_llm)

    async def route(
        self,
        rewrite_result: QueryRewriteResult,
        session: SessionMemory,
        attachments: list,
        raw_message: Optional[str] = None,
    ) -> LLMIntentResult:
        # ① 规则引擎（使用原始消息避免 QueryRewriter 副作用）
        msg_for_rules = raw_message or rewrite_result.rewritten_query
        rule_result = self.rule_registry.classify(msg_for_rules, attachments)
        logger.info(
            f"[LLMIntentRouter] 规则结果 | intent={rule_result.intent.value if rule_result.intent else 'None'} | "
            f"strength={rule_result.strength} | rule={rule_result.rule_name}"
        )

        # ② 小模型校准（所有情况一律进入，包括 STRONG）
        cal_result = await self.calibrator.calibrate(rule_result, rewrite_result, session)
        logger.info(
            f"[LLMIntentRouter] 校准结果 | intent={cal_result.intent.value} | "
            f"confidence={cal_result.confidence:.2f} | rule_agreement={cal_result.rule_agreement} | "
            f"needs_fallback={cal_result.needs_llm_fallback}"
        )

        # ③ 快速通道：规则 STRONG + 小模型同意 + 置信度高 → 直接采信
        if (
            rule_result.strength == RuleStrength.STRONG
            and cal_result.rule_agreement
            and cal_result.confidence >= 0.85
            and not cal_result.needs_llm_fallback
        ):
            return LLMIntentResult(
                intent=cal_result.intent,
                confidence=cal_result.confidence,
                layer="calibration",
                rule_result=rule_result,
                calibration_result=cal_result,
                reason=f"规则 STRONG + 小模型确认: {cal_result.reason}",
                slots=cal_result.slots,
                slot_sources=cal_result.slot_sources,
                missing_slots=cal_result.missing_slots,
                needs_clarification=cal_result.needs_clarification,
                clarification_question=cal_result.clarification_question,
                candidate_options=cal_result.clarification_options,
            )

        # ④ 常规通道：小模型置信度足够 → 采信
        if cal_result.confidence >= 0.8 and not cal_result.needs_llm_fallback:
            return LLMIntentResult(
                intent=cal_result.intent,
                confidence=cal_result.confidence,
                layer="calibration",
                rule_result=rule_result,
                calibration_result=cal_result,
                reason=cal_result.reason,
                slots=cal_result.slots,
                slot_sources=cal_result.slot_sources,
                missing_slots=cal_result.missing_slots,
                needs_clarification=cal_result.needs_clarification,
                clarification_question=cal_result.clarification_question,
                candidate_options=cal_result.clarification_options,
            )

        # ⑤ 大模型兜底（规则与小模型冲突 / 置信度不足）
        logger.info(f"[LLMIntentRouter] 触发大模型兜底 | cal_confidence={cal_result.confidence:.2f}")
        fb_result = await self.fallback.classify(rule_result, cal_result, rewrite_result, session)
        logger.info(
            f"[LLMIntentRouter] 兜底结果 | intent={fb_result.intent.value} | "
            f"confidence={fb_result.confidence:.2f} | arbitration={fb_result.arbitration} | "
            f"needs_clarification={fb_result.needs_clarification}"
        )

        # ⑥ 兜底后判定：是否需要澄清
        if fb_result.needs_clarification or fb_result.confidence < 0.7:
            return LLMIntentResult(
                intent=fb_result.intent,
                confidence=fb_result.confidence,
                layer="clarification",
                rule_result=rule_result,
                calibration_result=cal_result,
                reason=f"{fb_result.reason} | arbitration={fb_result.arbitration}",
                slots=fb_result.slots,
                slot_sources=fb_result.slot_sources,
                missing_slots=fb_result.missing_slots,
                needs_clarification=True,
                clarification_question=fb_result.clarification_question,
                candidate_options=fb_result.clarification_options,
            )

        # 兜底裁决成功
        return LLMIntentResult(
            intent=fb_result.intent,
            confidence=fb_result.confidence,
            layer="llm_fallback",
            rule_result=rule_result,
            calibration_result=cal_result,
            reason=f"{fb_result.reason} | arbitration={fb_result.arbitration}",
            slots=fb_result.slots,
            slot_sources=fb_result.slot_sources,
            missing_slots=fb_result.missing_slots,
            needs_clarification=fb_result.needs_clarification,
            clarification_question=fb_result.clarification_question,
            candidate_options=fb_result.clarification_options,
        )

    # ═══════════════════════════════════════════════════════
    # 多意图改造：新增 route_multi
    # ═══════════════════════════════════════════════════════

    async def route_multi(
        self,
        rewrite_result: QueryRewriteResult,
        session: SessionMemory,
        attachments: list,
        raw_message: Optional[str] = None,
    ) -> MultiIntentResult:
        """多意图识别流水线入口：RuleRegistry → SmallModelCalibrator → LLMFallbackClassifier"""
        # ① 规则引擎（优先使用 QueryRewrite 结果，更好地处理多轮对话）
        msg_for_rules = rewrite_result.rewritten_query or raw_message
        rule_matches = self.rule_registry.classify_all(
            msg_for_rules, 
            attachments,
            fallback_message=rewrite_result.rewritten_query if rewrite_result.rewritten_query != msg_for_rules else None
        )
        logger.info(
            f"[LLMIntentRouter-Multi] 规则结果 | matches={len(rule_matches)} | "
            f"intents={[r.intent.value if r.intent else 'None' for r in rule_matches]} | "
            f"strengths={[r.strength for r in rule_matches]}"
        )

        # ①b clarify 场景特殊处理：规则全部 MISS 时，强制创建虚拟 CHAT 候选送入校准器
        # 让校准器利用工作记忆（上轮澄清上下文）推断真实意图
        has_non_miss = any(r.intent is not None for r in rule_matches)
        if not has_non_miss:
            has_pending_clarification = (
                session is not None
                and hasattr(session, "pending_clarification")
                and session.pending_clarification is not None
            )
            if rewrite_result.follow_up_type == "clarify" or has_pending_clarification:
                virtual_rule = LLMRuleResult(
                    intent=LLMIntentType.CHAT,
                    strength=RuleStrength.WEAK,
                    rule_name="clarify_virtual",
                    metadata={},
                    trigger="",
                )
                rule_matches.append(virtual_rule)
                logger.info(
                    f"[LLMIntentRouter-Multi] clarify场景强制创建虚拟CHAT候选，"
                    f"follow_up_type={rewrite_result.follow_up_type} | has_pending={has_pending_clarification}"
                )

        # ② 小模型校准（逐意图校准槽位）
        calibrated = await self.calibrator.calibrate_multi(
            rule_matches=rule_matches,
            rewrite_result=rewrite_result,
            session=session,
        )
        logger.info(
            f"[LLMIntentRouter-Multi] 校准结果 | candidates={len(calibrated)} | "
            f"intents={[c.intent_type.value for c in calibrated]} | "
            f"confidences={[f'{c.confidence:.2f}' for c in calibrated]} | "
            f"rule_agreements={[c.rule_agreement for c in calibrated]}"
        )

        # ②c clarify 场景后处理：校准器返回 CHAT 但工作记忆显示上轮澄清时，
        # 根据上轮语义和本轮补充的实体硬编码修正意图
        if session and hasattr(session, "pending_clarification") and session.pending_clarification:
            pc = session.pending_clarification
            # missing_slots 非空，或 resolved_slots 为空（上轮意图模糊未解析出实体）
            is_clarify_scenario = bool(pc.missing_slots) or not bool(pc.resolved_slots)
            if pc.pending_intent == "chat" and is_clarify_scenario:
                for c in calibrated:
                    if c.intent_type == LLMIntentType.CHAT:
                        # 检查本轮是否补充了缺失的 company/position
                        # 先从 rewrite_result 和候选 slots 中检查
                        has_company = bool(
                            c.slots.get("company") 
                            or rewrite_result.resolved_references.get("company")
                            or rewrite_result.resolved_references.get("__correct_company__")
                        )
                        has_position = bool(
                            c.slots.get("position")
                            or rewrite_result.resolved_references.get("position")
                            or rewrite_result.resolved_references.get("__correct_position__")
                        )
                        # 若仍未命中，从 rewritten_query 中按长度降序提取实体
                        if not has_company or not has_position:
                            kb_companies, kb_positions = _load_kb_entities()
                            msg = rewrite_result.rewritten_query or ""
                            if not has_company:
                                company = next((c for c in sorted(kb_companies, key=len, reverse=True) if c in msg), None)
                                if company:
                                    has_company = True
                                    c.slots["company"] = company
                            if not has_position:
                                position = next((p for p in sorted(kb_positions, key=len, reverse=True) if p in msg), None)
                                if position:
                                    has_position = True
                                    c.slots["position"] = position
                        if has_company or has_position:
                            # 获取上轮用户输入，推断真实意图
                            last_user_msg = ""
                            if session.working_memory.turns:
                                last_user_msg = session.working_memory.turns[-1].user_message
                            # 上轮语义关键词分类
                            assess_keywords = ["匹配", "适合", "差距", "契合", "够格", "搭不搭"]
                            verify_keywords = ["薪资", "工资", "要求", "学历", "经验", "加班", "福利", "地点", "怎么样", "分析", "介绍", "了解", "看看"]
                            if any(kw in last_user_msg for kw in assess_keywords):
                                c.intent_type = LLMIntentType.ASSESS
                                c.confidence = 0.82
                                c.reason = f"clarify场景修正：上轮'{last_user_msg}'触发澄清，本轮补充实体，推断为ASSESS"
                                c.slots["company"] = c.slots.get("company") or rewrite_result.resolved_references.get("company") or rewrite_result.resolved_references.get("__correct_company__")
                                c.slots["position"] = c.slots.get("position") or rewrite_result.resolved_references.get("position") or rewrite_result.resolved_references.get("__correct_position__")
                                c.slots["attributes"] = c.slots.get("attributes", ["匹配度"])
                                c.slots["jd_source"] = "kb"
                                c.slots["search_keywords"] = rewrite_result.rewritten_query
                                logger.info(f"[LLMIntentRouter-Multi] clarify场景修正 CHAT→ASSESS | slots={c.slots}")
                            elif any(kw in last_user_msg for kw in verify_keywords):
                                c.intent_type = LLMIntentType.VERIFY
                                c.confidence = 0.82
                                c.reason = f"clarify场景修正：上轮'{last_user_msg}'触发澄清，本轮补充实体，推断为VERIFY（综合查询）"
                                c.slots["company"] = c.slots.get("company") or rewrite_result.resolved_references.get("company") or rewrite_result.resolved_references.get("__correct_company__")
                                c.slots["position"] = c.slots.get("position") or rewrite_result.resolved_references.get("position") or rewrite_result.resolved_references.get("__correct_position__")
                                # 综合查询：attributes 默认填充为综合情况，不强制要求具体属性词
                                c.slots["attributes"] = c.slots.get("attributes", ["综合情况"])
                                c.slots["qa_type"] = "factual"
                                c.slots["search_keywords"] = rewrite_result.rewritten_query
                                logger.info(f"[LLMIntentRouter-Multi] clarify场景修正 CHAT→VERIFY | slots={c.slots}")

        # ②b 快速通道：所有候选都来自规则且置信度高 → 直接采信，跳过仲裁
        if calibrated and len(calibrated) <= 2:
            all_fast = all(
                c.rule_result and c.confidence >= 0.85 and c.rule_agreement
                for c in calibrated
            )
            # 检查意图冲突：CHAT + 其他意图需要仲裁
            intent_types = {c.intent_type for c in calibrated}
            has_chat_conflict = LLMIntentType.CHAT in intent_types and len(intent_types) > 1
            
            if all_fast and not has_chat_conflict:
                fast_candidates = []
                for c in calibrated:
                    fc = IntentCandidate(
                        intent_type=c.intent_type,
                        confidence=c.confidence,
                        reason=f"快速通道: {c.reason}",
                        slots=c.slots,
                        slot_sources=c.slot_sources,
                        missing_slots=c.missing_slots,
                        needs_clarification=c.needs_clarification,
                        clarification_question=c.clarification_question,
                        clarification_options=c.clarification_options,
                        source="fastpath",
                        rule_result=c.rule_result,
                        rule_agreement=True,
                        execution_cost=c.execution_cost,
                        dependencies=c.dependencies,
                    )
                    fast_candidates.append(fc)
                
                topology = self.fallback._build_execution_topology(fast_candidates)
                global_slots = self.fallback._merge_global_slots(fast_candidates, session=session)
                needs_clarification, reason = self.fallback._check_clarification_need(fast_candidates, global_slots)
                
                logger.info(
                    f"[LLMIntentRouter-Multi] 快速通道 | candidates={len(fast_candidates)} | "
                    f"intents={[c.intent_type.value for c in fast_candidates]} | "
                    f"conflicts=none"
                )
                return MultiIntentResult(
                    candidates=fast_candidates,
                    primary_intent=fast_candidates[0].intent_type,
                    needs_clarification=needs_clarification,
                    clarification_reason=reason,
                    global_slots=global_slots,
                    execution_topology=topology,
                )

        # ③ 大模型兜底仲裁（冲突消解 + 拓扑构建）
        multi_result = await self.fallback.arbitrate(
            calibrated_candidates=calibrated,
            rewrite_result=rewrite_result,
            session=session,
        )
        logger.info(
            f"[LLMIntentRouter-Multi] 仲裁结果 | primary={multi_result.primary_intent.value if multi_result.primary_intent else 'None'} | "
            f"final_candidates={len(multi_result.candidates)} | "
            f"needs_clarification={multi_result.needs_clarification} | "
            f"topology={multi_result.execution_topology}"
        )

        return multi_result
