"""
LLM 路线专用意图识别模块 —— 规则引擎 + 大模型兜底

采用 JTBD 六类意图体系，不与规则路线的四类意图共用任何组件。

流水线：
    Query 改写 → RuleRegistry（16 条规则，分层遍历）
    → STRONG / WEAK → 规则直接采信（生成 IntentCandidate）
    → 指代消解注入（QueryRewriter resolved_references + evidence_cache）
    → 冲突消解（互斥意图去重）
    → 全部 MISS / confidence < 0.7 / 多候选冲突 → LLMFallbackClassifier（大模型兜底）
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
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)


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
    # 多意图支持：当大模型输出 intents 数组时，额外意图存储在这里
    additional_intents: list = field(default_factory=list)



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
            if not msg:
                return
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

from app.core.kb_entities import _load_kb_entities, KB_COMPANIES, KB_POSITIONS


def _extract_basic_slots(msg: str, intent: LLMIntentType, kb_companies: list, kb_positions: list) -> dict:
    """规则层基础实体提取：根据意图类型提取 company/position/attributes 等"""
    slots = {}
    # 公司和岗位（ASSESS/VERIFY/PREPARE/EXPLORE 都可能需要）
    # 按长度降序匹配，优先命中更长的实体（如 "字节跳动" 优先于 "字节"）
    if intent in (LLMIntentType.ASSESS, LLMIntentType.VERIFY, LLMIntentType.PREPARE, LLMIntentType.EXPLORE):
        # 提取所有匹配的公司（支持多公司对比场景，如"百度和美团"）
        matched_companies = [c for c in sorted(kb_companies, key=len, reverse=True) if c in msg]
        # 去重：长匹配优先，短匹配如果被长匹配包含则跳过
        deduped_companies = []
        for c in matched_companies:
            if not any(c != other and c in other for other in matched_companies):
                deduped_companies.append(c)
        if deduped_companies:
            slots["company"] = deduped_companies[0]
            if len(deduped_companies) > 1:
                slots["companies"] = deduped_companies

        # 提取所有匹配的岗位（支持多岗位对比场景）
        matched_positions = [p for p in sorted(kb_positions, key=len, reverse=True) if p in msg]
        deduped_positions = []
        for p in matched_positions:
            if not any(p != other and p in other for other in matched_positions):
                deduped_positions.append(p)
        if deduped_positions:
            slots["position"] = deduped_positions[0]
            if len(deduped_positions) > 1:
                slots["positions"] = deduped_positions
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
    ASSESS_KWS = ["这家", "这个", "该", "匹配", "适合", "差距", "差多少", "契合", "够格", "搭不搭", "我能去"]
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
    if not msg:
        return None
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

    # ═══════════════════════════════════════════════════════
    # 强 explore 句式：直接命中，不受 ASSESS_KWS 排除影响
    # 这些句式语义明确，无歧义就是探索/筛选意图
    # ═══════════════════════════════════════════════════════
    STRONG_EXPLORE_PATTERNS = [
        "筛几个能投", "筛选几个能投", "筛几个岗", "筛几个岗位",
        "推几个能投", "挑几个能投", "帮我筛几个", "帮我筛选几个",
        "帮我推几个", "帮我挑几个", "帮我看看有什么", "帮我找几个",
        "投哪些", "投递哪些", "能投哪些", "能投递哪些",
    ]
    for pattern in STRONG_EXPLORE_PATTERNS:
        if pattern in msg:
            slots = _extract_basic_slots(msg, LLMIntentType.EXPLORE, kb_companies, kb_positions)
            return LLMRuleResult(
                intent=LLMIntentType.EXPLORE, strength=RuleStrength.STRONG,
                rule_name="L2-强探索句式", trigger=pattern, metadata=slots,
            )

    RANGE_KWS = ["所有", "全局", "哪家", "排序", "适合我", "投哪些", "投递哪些", "帮我选", "对比几家", "筛几个", "筛选", "推几个", "挑几个", "看看有什么", "有什么", "帮我找", "帮我看看", "有用的", "合适的", "有价值的", "分析一遍", "对哪个岗", "哪个岗", "相关", "相关岗位"]
    has_range = any(kw in msg for kw in RANGE_KWS)
    ATTR_WORDS = ["薪资", "工资", "要求", "技能", "学历", "福利"]
    has_attr = any(w in msg for w in ATTR_WORDS)
    if has_attr and not has_range:
        return None
    if has_range:
        # 排除：同时存在 ASSESS 关键词 + 具体实体 → 让 L3-引用式评估处理
        ASSESS_KWS = ["匹配度", "匹配", "适合吗", "适合", "差距", "差多少", "契合", "够格", "能去", "能投", "转行"]
        has_assess_kw = any(kw in msg for kw in ASSESS_KWS)
        has_entity = any(e in msg for e in kb_companies + kb_positions)
        logger.debug(f"[L2-全局探索-DEBUG] has_assess_kw={has_assess_kw} has_entity={has_entity} msg='{msg[:30]}...'")
        if has_assess_kw and has_entity:
            logger.debug("[L2-全局探索-DEBUG] 触发排除条件，返回 None")
            return None
        hit = next(kw for kw in RANGE_KWS if kw in msg)
        slots = _extract_basic_slots(msg, LLMIntentType.EXPLORE, kb_companies, kb_positions)
        # 有明确实体时降级为WEAK，让校准器基于语义做最终判断（避免"帮我看看字节跳动后端开发"被强制判为explore）
        strength = RuleStrength.WEAK if has_entity else RuleStrength.STRONG
        return LLMRuleResult(
            intent=LLMIntentType.EXPLORE, strength=strength,
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
        slots = _extract_basic_slots(msg, LLMIntentType.PREPARE, kb_companies, kb_positions)
        return LLMRuleResult(
            intent=LLMIntentType.PREPARE, strength=RuleStrength.STRONG,
            rule_name="L2-面试准备", trigger=hit, metadata=slots,
        )
    return None


def rule_l2_attr_verify(msg, attachments, ctx, kb_companies, kb_positions):
    """属性核实：基于短语模式匹配，区分 verify/assess/explore 语境"""
    # 【触发短语】必须是"属性查询"的完整表达，单字不触发
    VERIFY_PHRASES = [
        # 薪资类
        "薪资多少", "工资多少", "月薪", "年薪", "待遇如何", "待遇怎么样",
        # 要求类 — 必须是"询问要求"的表达
        "要求什么", "有什么要求", "需要什么", "门槛是什么", "条件是什么",
        "要求是什么", "具体要求", "有要求吗", "有要求",
        "学历要求", "经验要求", "技能要求", "硬性要求", "软性要求",
        "要求多高", "要求高吗", "要求严吗", "要求严格吗",
        # 福利/工作条件类
        "福利怎么样", "福利如何", "有福利", "五险一金",
        "加班吗", "加班多吗", "加班严重吗", "加班情况",
        "出差", "出差吗", "出差多吗",
        "工作地点", "工作地址", "办公地点", "远程", "在家办公",
        "工作年限", "团队规模", "发展前景", "晋升空间",
        # 技术栈类
        "技术栈", "用什么技术", "用什么语言", "技术架构",
        # 工作内容类
        "工作内容", "岗位职责", "具体做什么", "干什么",
        # 综合类
        "什么情况", "具体介绍", "详细说说", "详细信息",
        "区别是什么", "有什么不同", "差异是什么",
        # 存在性询问
        "有吗", "有没有", "存在吗",
        "几年经验", "多少年", "工作几年",
    ]
    
    # 【排除短语】这些语境下即使含触发词也不是 verify
    VERIFY_EXCLUDES = [
        # 评估语境（assess）— 仅当包含明确的匹配度询问词时排除
        "够格吗", "能去吗", "能投吗", "适合吗", "匹配吗", "差距大吗",
        "够得上", "够不上", "行不行", "能不能过", "能过吗",
        # 筛选语境（explore）
        "符合要求", "满足要求", "达到要求", "不要求", "没要求",
    ]
    
    # 先检查排除条件
    if any(p in msg for p in VERIFY_EXCLUDES):
        return None
    
    # 再检查触发条件（短语匹配）
    has_verify_phrase = any(p in msg for p in VERIFY_PHRASES)
    if not has_verify_phrase:
        return None
    
    hit = next(p for p in VERIFY_PHRASES if p in msg)
    
    # 多意图场景下，同时存在匹配度词时强度降为 WEAK
    STRONG_MATCH_KWS = ["匹配度", "差距", "差多少", "契合", "够格"]
    strength = RuleStrength.WEAK if any(kw in msg for kw in STRONG_MATCH_KWS) else RuleStrength.STRONG
    slots = _extract_basic_slots(msg, LLMIntentType.VERIFY, kb_companies, kb_positions)
    return LLMRuleResult(
        intent=LLMIntentType.VERIFY, strength=strength,
        rule_name="L2-属性核实", trigger=hit, metadata=slots,
    )


def rule_l2_introduce_verify(msg, attachments, ctx, kb_companies, kb_positions):
    """岗位综合分析/介绍：分析/介绍一下/了解一下 + 实体 → VERIFY（综合查询）"""
    if attachments and len(attachments) > 0:
        return None
    INTRODUCE_KWS = ["分析一下", "介绍一下", "分析", "介绍", "了解一下", "说说", "讲讲"]
    has_introduce = any(kw in msg for kw in INTRODUCE_KWS)
    has_entity = any(e in msg for e in kb_companies + kb_positions)
    # 排除已被 ASSESS 规则覆盖的匹配度场景
    ASSESS_KWS = [
        "匹配度", "差距", "差多少", "契合", "够格", "适合吗", "搭不搭",
        "能去", "能过吗", "能不能去", "能不能过", "适合不适合",
        "够格吗", "够得上", "够不上", "行不行", "行不行啊",
    ]
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
    GENERAL_KWS = ["职业规划", "行业趋势", "如何转行", "简历优化"]
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
    has_entity = has_company or has_position

    # ═══════════════════════════════════════════════════════
    # 强匹配度词：用户明确在问匹配度
    # ═══════════════════════════════════════════════════════
    MATCH_KWS = ["匹配度", "匹配", "适合吗", "适合", "差距", "差多少", "契合", "够格", "能去", "能投", "转行"]
    has_match = any(kw in msg for kw in MATCH_KWS)
    if has_entity and has_match:
        hit = next(kw for kw in MATCH_KWS if kw in msg)
        slots = _extract_basic_slots(msg, LLMIntentType.ASSESS, kb_companies, kb_positions)
        return LLMRuleResult(
            intent=LLMIntentType.ASSESS, strength=RuleStrength.WEAK,
            rule_name="L3-引用式评估", trigger=hit,
            metadata=slots,
        )

    # ═══════════════════════════════════════════════════════
    # 弱评估询问：用户想评估某个具体岗位，但没有用强匹配度词
    # 典型句式："帮我看看/分析 阿里那个后端开发"、"重点看看字节那个PM"
    # 条件：有实体 + 有"帮我" + 有看看/分析/了解一下 + 无明确属性查询
    # ═══════════════════════════════════════════════════════
    if has_entity and "帮我" in msg:
        WEAK_ASSESS_KWS = ["看看", "分析", "了解一下", "了解"]
        has_weak_assess = any(kw in msg for kw in WEAK_ASSESS_KWS)
        # 明确属性查询词：如果用户在问具体属性，则走 verify 而非 assess
        ATTR_QUERIES = ["薪资", "工资", "要求什么", "需要什么", "门槛", "条件", "技能", "学历", "加班", "福利"]
        has_attr_query = any(a in msg for a in ATTR_QUERIES)
        if has_weak_assess and not has_attr_query:
            hit = next(kw for kw in WEAK_ASSESS_KWS if kw in msg)
            slots = _extract_basic_slots(msg, LLMIntentType.ASSESS, kb_companies, kb_positions)
            return LLMRuleResult(
                intent=LLMIntentType.ASSESS, strength=RuleStrength.WEAK,
                rule_name="L3-弱评估询问", trigger=hit,
                metadata=slots,
            )

    return None


def rule_l3_referenced_verify(msg, attachments, ctx, kb_companies, kb_positions):
    """引用式核实：疑问词 + 属性短语 + 实体 → VERIFY"""
    
    # 【排除短语】assess/explore 语境，不触发 verify
    VERIFY_EXCLUDES = [
        "我有", "我的", "够格吗", "能去吗", "能投吗", "适合吗", "匹配吗",
        "差距大吗", "够得上", "够不上", "行不行", "能不能过", "能过吗",
        "符合要求", "满足要求", "达到要求",
    ]
    if any(p in msg for p in VERIFY_EXCLUDES):
        return None
    
    # 【疑问词】
    QUESTION_WORDS = ["什么", "多少", "怎么", "吗", "呢", "哪些", "如何", "介绍一下", "怎样", "啥"]
    has_question = any(w in msg for w in QUESTION_WORDS)
    
    # 【属性短语】比单字更精确，区分 verify/assess/explore 语境
    VERIFY_ATTR_PHRASES = [
        "技能", "技术栈", "用什么技术",
        "薪资", "工资", "月薪", "年薪", "待遇",
        "要求什么", "有什么要求", "需要什么", "门槛", "条件是什么",
        "学历", "经验要求", "工作年限",
        "福利", "五险一金",
        "加班", "工作地点", "办公地点",
        "团队规模", "发展前景", "晋升空间",
        "工作内容", "岗位职责", "具体做什么",
        "区别", "有什么不同", "差异",
    ]
    has_attr_phrase = any(p in msg for p in VERIFY_ATTR_PHRASES)
    
    has_entity = any(e in msg for e in kb_companies + kb_positions)
    
    # 条件A：疑问词 + 属性短语 + 实体
    if has_question and has_attr_phrase and has_entity:
        slots = _extract_basic_slots(msg, LLMIntentType.VERIFY, kb_companies, kb_positions)
        slots["mode"] = "question"
        return LLMRuleResult(
            intent=LLMIntentType.VERIFY, strength=RuleStrength.WEAK,
            rule_name="L3-引用式核实", trigger="question+attr+entity",
            metadata=slots,
        )
    
    # 条件B：以常见属性词结尾 + 实体 + 短query（如"字节那个岗薪资"）
    ENDING_ATTRS = ["薪资", "工资", "福利", "加班", "学历", "要求", "经验", "门槛", "条件", "待遇", "前景"]
    ends_with_attr = any(msg.strip().endswith(w) for w in ENDING_ATTRS)
    if ends_with_attr and has_entity and len(msg) < 60:
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
    import re
    has_company = any(c in msg for c in kb_companies)
    # 补充：英文公司名（如Meta, Google）或含常见后缀的公司名
    if not has_company:
        company_like = any(suffix in msg for suffix in ["集团", "科技", "网络", "公司", "有限"])
        has_english_company = bool(re.search(r'\b[A-Z][a-zA-Z]+\b', msg))
        has_company = company_like or has_english_company
    EXIST_KWS = ["有没有", "有", "存在"]
    has_exist = any(kw in msg for kw in EXIST_KWS)
    # 必须包含 '有'/'有没有'/'存在' 且包含疑问词，才认定为存在性询问
    QUESTION_MARKERS = ["吗", "么", "没", "不"]
    has_question = any(q in msg for q in QUESTION_MARKERS)
    # 补充：'有...岗吗' 也是存在性询问（如"百度有Java后端岗吗"）
    has_job_existence = bool(re.search(r'有\w+岗吗', msg))
    if has_company and (has_exist or has_job_existence) and has_question:
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
# 大模型兜底 Prompt
# ═══════════════════════════════════════════════════════

LLM_FALLBACK_SYSTEM = """你是一位深度意图与槽位联合理解专家。当前输入经过规则引擎匹配后未命中或存在歧义，需要你基于完整上下文做最终裁决，并补全/修正槽位。

【六类意图定义】
- EXPLORE：探索选岗、排序推荐、对比多家
- ASSESS：单JD匹配度分析、差距评估
- VERIFY：查询具体属性（薪资/要求/学历/技术栈）
- PREPARE：面试题生成、模拟面试、备考建议
- MANAGE：文件上传/删除/更新、资料管理
- CHAT：问候、闲聊、通用职业咨询

【EXPLORE vs VERIFY 边界判定】
- EXPLORE = 用户想从"多个岗位"中做筛选、排序、推荐、对比、选择
  → 关键语义：存在范围词（哪些、哪个、所有、全部、几家、几个、有什么）或选择动作（筛、投、推、挑、看看有什么）
  → 典型句式："能投哪些"、"哪个岗最有用"、"帮我看看有什么Java岗"
- VERIFY = 用户想了解"某个/某些具体岗位"的属性信息
  → 关键语义：询问具体属性（薪资、要求、技能、学历、加班、福利、怎么样）
  → 典型句式："需要什么技能"、"薪资多少"、"这个岗加班吗"

【EXPLORE vs ASSESS 边界判定】
- EXPLORE = 用户想让系统**帮他找/筛/推**合适的岗位（动作是"找"）
  → 典型句式："帮我筛几个能投的"、"推几个技术岗"、"帮我看看有什么Java岗"
  → 关键词：筛、推、挑、找、看看有什么、投哪些
  → 即使句子后半段提到具体公司（如"再重点看看字节那个PM"），前半段的"筛/推/挑"仍表明整体是 explore 意图，后半段是 explore 结果中的重点关注对象
- ASSESS = 用户已经有一个**具体目标岗位**，想评估自己能不能去（动作是"评"）
  → 典型句式："字节PM我够格吗"、"我的背景匹配吗"、"我能转行去吗"
  → 关键词：匹配、够格、适合、差距、能去、转行
- 反例（易混淆）：
  × "帮我筛几个能投的，再重点看看字节那个PM" → EXPLORE（不是 ASSESS，"筛"是核心动作）
  × "帮我推几个技术岗，重点看看阿里后端" → EXPLORE+ASSESS（"推"=explore，"看看"=弱assess）

【ASSESS vs VERIFY 边界判定】
这是最常见的误判来源，关键看用户的**核心诉求**：
- ASSESS = 用户在评估**自己**与岗位的匹配度
  → 关键语义：用户主动提及自己的背景、技能、经验，并询问"够不够格""能不能去""适不适合""有没有帮助"
  → 典型句式："我有3年经验能投吗""我的设计背景对产品岗有帮助吗""我够格吗"
  → 核心目的：得到"去/不去"或"匹配度高/低"的结论
- VERIFY = 用户在查询**岗位本身**的属性信息，不关心自己的匹配度
  → 关键语义：询问岗位的客观属性（薪资、要求、学历、加班），不包含自我评估
  → 典型句式："这个岗薪资多少""要求什么学历""需要加班吗"
  → 核心目的：了解岗位细节，与自己无关
- 判定口诀：
  1. 用户说"我的X经验对Y岗有帮助吗/够格吗/能去吗" → ASSESS（在评估自己的匹配度）
  2. 用户说"Y岗需要什么经验/要求什么学历" → VERIFY（在查询岗位要求）
  3. 若 query 同时包含自我背景+匹配度词+属性词（如"我有3年经验但JD要求精通分布式，我够格吗"）→ ASSESS（核心诉求是评估匹配度，"要求"只是背景描述）
  4. "没有工作经验能投产品岗吗" → ASSESS（用户在评估自己够不够格，不是查岗位要求）
- 重要原则：当 query 包含自我背景（"我有X年经验""我的背景""我没有X"）+ 疑问语气（"吗""么"）时，默认优先 ASSESS 而非 VERIFY

【规则全MISS时的判定原则】
规则引擎未命中不代表用户在做闲聊（CHAT）。以下情况即使规则MISS也应判为业务意图：
- 包含公司名（字节、阿里、腾讯、百度、美团等）或岗位名（产品经理、后端开发、设计师等）
- 包含自我背景描述（"我有X年经验""我的背景是"）
- 包含评估/询问语气词（"够格吗""能去吗""匹配吗""有用吗"）
- 默认优先级：VERIFY > ASSESS > EXPLORE > CHAT（当存在实体和疑问词时，绝不返回CHAT）

【槽位 Schema】
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
   - 槽位：{calib_slots}
   - 缺失槽位：{calib_missing_slots}

2. 【完整上下文】
   - 改写查询：{rewritten_query}
   - 追问类型：{follow_up_type}
   - 工作记忆：{working_history}
   - 压缩记忆：{compressed_history}
   - 长期记忆：{long_term_profile}

3. 【系统状态】
   - resume_available: {resume_available}
   - resume_text摘要: {resume_summary}
   【重要】resume_available 反映系统是否已有用户简历，这是客观事实，**必须如实填入 slots**。如果 resume_available=true，则 ASSESS/EXPLORE 等意图的 slots 中 resume_available 必须设为 true，绝不能自行猜测为 false。

【你的任务】
1. 基于规则引擎输出和三层记忆，做出最终意图判断
2. **多意图识别**：用户query可能同时包含多个意图（如"帮我分析一下字节跳动的匹配度，再给我一些面试建议"），请识别出所有真实意图
3. 尽可能从三层记忆中补全缺失槽位
4. 若仍有必填槽位缺失且无法补全，生成澄清问题和选项
5. 输出最终置信度（0.0-1.0），必须基于证据强度诚实评估，不得虚高

【重要提醒：规则引擎可能出错，不要过于依赖】
- 规则引擎基于关键词匹配，可能过度匹配或漏匹配
- **你必须基于 query 本身的语义和三层记忆做独立判断**，不要把规则的结果当作权威
- 当规则判定为非 CHAT 但你认为 query 中确实没有业务关键词时，可以判 CHAT
- **【关键】当规则全 MISS（strength=MISS）时，不代表用户在闲聊**。规则MISS仅说明关键词不在规则列表中，不代表意图不是业务查询。你必须审查 rewritten_query 和上下文记忆：如果其中包含公司名、岗位名、属性询问词（薪资/要求/学历/出差/加班等），就必须判定为对应的业务意图，绝不能因为规则MISS就默认返回CHAT。

【默认原则：规则MISS时的优先选择】
在求职助手场景下，用户的query绝大多数是业务查询。当规则全MISS且你无法确定具体意图时，**默认优先选择VERIFY**，而非CHAT。只有当query明显是问候（"你好"）、感谢（"谢谢"）、闲聊（"今天天气怎么样"）或通用职业咨询（"行业趋势如何"）时，才能选择CHAT。

【CHAT判定的严格条件】（必须全部满足，缺一不可）
1. query中没有任何公司名、岗位名、技能名等业务实体
2. query中没有任何属性询问词（薪资/要求/学历/技能/加班/福利/出差等）
3. query中没有任何探索/筛选/推荐动词（帮我看看/推荐/筛选/对比/分析一下等）
4. query明显是问候、感谢、闲聊或纯通用咨询
5. 即使结合上下文记忆，也无法推断出任何业务意图
**以上条件只要有一条不满足，就不能判CHAT。**

【多意图判断规则】
- 只有当query中确实存在多个独立诉求时，才输出多个意图
- 不要过度拆分：同一意图框架下的追问（如匹配度分析后问薪资）应归为一个意图
- 多意图时，请为每个意图分别输出 confidence、slots、missing_slots
- 如果无法确定主次意图或意图之间存在冲突，标记 needs_clarification=true

【输出格式】
严格JSON，不要markdown代码块。

**多意图模式**（推荐，当存在多个意图时使用）：
{{
  "intents": [
    {{
      "intent": "ASSESS",
      "confidence": 0.88,
      "slots": {{"company":"字节跳动","position":"算法工程师",...}},
      "slot_sources": {{"company":"working_memory",...}},
      "missing_slots": []
    }},
    {{
      "intent": "PREPARE",
      "confidence": 0.75,
      "slots": {{"company":"字节跳动","position":"算法工程师","focus_area":"技术面试",...}},
      "slot_sources": {{"company":"working_memory",...}},
      "missing_slots": []
    }}
  ],
  "reason": "详细推理过程",
  "arbitration": "采信规则/独立判断",
  "needs_clarification": false,
  "clarification_question": null,
  "clarification_options": []
}}

**单一意图模式**（向后兼容，只有一个意图时使用）：
{{
  "intent": "EXPLORE/ASSESS/VERIFY/PREPARE/MANAGE/CHAT",
  "confidence": 0.0-1.0,
  "reason": "详细推理过程",
  "arbitration": "采信规则/独立判断",
  "slots": {{...}},
  "slot_sources": {{...}},
  "missing_slots": ["字段名"],
  "needs_clarification": true/false,
  "clarification_question": "若缺失必填槽位，生成问题",
  "clarification_options": ["选项1", "选项2", "选项3"]
}}

【澄清触发条件】（满足任一即触发）
- 最终 confidence < 0.7
- 必填槽位缺失且三层记忆无法补全
- 规则与语义严重冲突且无法仲裁
- 输入涉及多个可能意图且无法确定主次
"""

LLM_FALLBACK_EXAMPLES = """
【示例1：规则误判，大模型纠正为业务意图】
规则：VERIFY (WEAK)，触发规则="属性核实"
上下文：用户问"分析一下字节跳动的AI产品经理这个岗位"
推理：规则基于"岗位"+"分析"误判为 VERIFY，但实际上 query 中有明确的业务关键词"分析""岗位""字节跳动""AI产品经理"，这是一个典型的岗位评估请求，应判定为 ASSESS。company 和 position 可从当前 query 直接提取。
输出：
{"intent":"ASSESS","confidence":0.88,"reason":"规则将'分析一下岗位'误判为VERIFY属性查询。我审查后发现query中包含明确的业务关键词'分析''岗位''字节跳动''AI产品经理'，结合用户表达的是对特定岗位的评估需求，应判定为ASSESS。company和position从当前query提取","arbitration":"独立判断","slots":{"company":"字节跳动","position":"AI产品经理","attributes":["匹配度"],"jd_source":"kb","resume_available":true,"search_keywords":"字节跳动 AI产品经理 分析"},"slot_sources":{"company":"current_query","position":"current_query","attributes":"default","jd_source":"default","resume_available":"system_state","search_keywords":"current_query"},"missing_slots":[],"needs_clarification":false,"clarification_question":null,"clarification_options":[]}

【示例2：规则全MISS，大模型独立判断】
规则：CHAT (MISS)
上下文：用户问"Meta 和 Google 在国内有 AI 产品岗吗"
推理：规则未命中（无明确触发词），但 query 中包含明确的探索意图：范围词"有...吗"、公司名"Meta""Google"、岗位名"AI产品岗"。用户在询问这些公司是否有特定岗位，属于 EXPLORE（存在性查询）。
输出：
{"intent":"EXPLORE","confidence":0.75,"reason":"规则未命中，但query中包含明确的探索语义：范围词'有...吗'、公司名'Meta''Google'、岗位名'AI产品岗'。用户在询问这些公司是否有特定岗位，属于EXPLORE存在性查询","arbitration":"独立判断","slots":{"search_keywords":"Meta Google AI产品岗","filters":{"company":["Meta","Google"],"position":"AI产品岗"},"resume_available":false},"slot_sources":{"search_keywords":"current_query","filters":"current_query","resume_available":"system_state"},"missing_slots":[],"needs_clarification":false,"clarification_question":null,"clarification_options":[]}

【示例3：歧义query，大模型补全槽位并触发澄清】
规则：CHAT (WEAK)
上下文：用户问"帮我规划一下"，上轮是ASSESS字节跳动算法岗
推理："规划"一词歧义大，但结合上轮ASSESS上下文，用户更可能问"投递规划"（属于EXPLORE）而非"职业规划"（CHAT）。然而信息不足，无法确定filters等槽位。
输出：
{"intent":"EXPLORE","confidence":0.68,"reason":"'规划'一词歧义大，但结合上轮ASSESS上下文，用户更可能问投递规划。然而当前query未提供任何筛选条件，filters等关键槽位无法确定，信息不足","arbitration":"独立判断","slots":{"resume_available":true,"filters":{"skills":[],"location":null,"experience_years":null},"sort_by":"match_score","top_k":5,"search_keywords":"规划"},"slot_sources":{"resume_available":"system_state","filters":"default","sort_by":"default","top_k":"default","search_keywords":"current_query"},"missing_slots":["filters.skills","search_keywords"],"needs_clarification":true,"clarification_question":"您想基于什么条件进行规划？比如技能、地点、经验年限等","clarification_options":["基于我的技能栈Python和Go推荐","不限条件，全局推荐","先帮我看看字节跳动的匹配度"]}

【示例4：规则全MISS + 改写后语义极其明确 → 必须判VERIFY而非CHAT】
规则：CHAT (MISS)
上下文：用户原query是"上面那个岗具体要求是什么"，QueryRewrite已将其改写为"阿里巴巴后端开发岗位具体要求是什么"，并补全了company=阿里巴巴, position=后端开发。
推理：规则未命中（"要求是什么"不在规则关键词列表中），但改写后的query语义极其明确：公司名"阿里巴巴" + 岗位名"后端开发" + 属性询问"具体要求是什么"。slots中company和position已通过指代消解补全。根据重要提醒，规则MISS不代表非业务查询，绝不能判CHAT。
输出：
{"intent":"VERIFY","confidence":0.85,"reason":"规则未命中，但改写后的query语义极其明确：'阿里巴巴后端开发岗位'+'具体要求是什么'。slots中company和position已通过指代消解补全。含明确业务实体和属性询问，绝不能判CHAT","arbitration":"独立判断","slots":{"company":"阿里巴巴","position":"后端开发","attributes":["要求"],"qa_type":"requirements","search_keywords":"阿里巴巴 后端开发 岗位要求","resume_available":true},"slot_sources":{"company":"query_rewrite","position":"query_rewrite","attributes":"default","qa_type":"default","search_keywords":"current_query","resume_available":"system_state"},"missing_slots":[],"needs_clarification":false,"clarification_question":null,"clarification_options":[]}
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
        all_candidates: list = None,
    ) -> FallbackResult:
        """大模型兜底裁决（统一入口）
        
        Args:
            all_candidates: 当传入多候选列表时，prompt 会包含所有候选信息，支持多意图裁决
        """
        if self.llm is None:
            try:
                self.llm = LLMClient.from_config("core")
            except Exception as e:
                logger.warning(f"[LLMFallbackClassifier] core LLM 不可用，fallback: {e}")
                return self._fallback(rule_result, cal_result)

        working_history, compressed_history, long_term_profile = self._build_memory_context(session)

        system_prompt = f"{LLM_FALLBACK_SYSTEM}\n\n{LLM_FALLBACK_EXAMPLES}"
        # 构造简历状态摘要
        resume_available = False
        resume_summary = "无"
        if session and hasattr(session, "global_slots") and session.global_slots:
            resume_available = bool(session.global_slots.get("resume_available", False))
            resume_text = session.global_slots.get("resume_text", "")
            if resume_text and resume_text != "尚未上传简历":
                resume_summary = resume_text[:200].replace("\n", " ") + "..." if len(resume_text) > 200 else resume_text.replace("\n", " ")
        
        # 多候选模式：追加候选列表描述到 prompt
        candidate_section = ""
        if all_candidates:
            candidate_lines = []
            for idx, c in enumerate(all_candidates):
                rr = c.rule_result
                rule_info = f"规则:{rr.intent.value if rr and rr.intent else 'None'}({rr.strength.value if rr else 'N/A'})" if rr else "规则:无"
                line = (
                    f"  候选{idx+1}: intent={c.intent_type.value}, confidence={c.confidence:.2f}, "
                    f"{rule_info}, rule_agreement={c.rule_agreement}, slots={json.dumps(c.slots, ensure_ascii=False)}"
                )
                candidate_lines.append(line)
            candidate_section = "\n\n【多意图候选列表】\n" + "\n".join(candidate_lines)
            candidate_section += "\n\n请综合以上所有候选信息，判断用户的真实意图（可能有一个或多个），并输出多意图格式的JSON。"
        
        # 使用 str.replace 逐个替换，避免 .format() 误替换 prompt 中 JSON 示例的花括号
        replacements = {
            "{rule_intent}": rule_result.intent.value if rule_result.intent else "null",
            "{rule_strength}": rule_result.strength.value,
            "{rule_name}": rule_result.rule_name,
            "{calib_intent}": cal_result.intent.value,
            "{calib_confidence}": str(cal_result.confidence),
            "{calib_reason}": cal_result.reason,
            "{calib_slots}": json.dumps(cal_result.slots, ensure_ascii=False) if cal_result.slots else "无",
            "{calib_missing_slots}": json.dumps(cal_result.missing_slots, ensure_ascii=False) if cal_result.missing_slots else "无",
            "{rule_agreement}": "是" if cal_result.rule_agreement else "否",
            "{rewritten_query}": rewrite_result.rewritten_query,
            "{follow_up_type}": rewrite_result.follow_up_type,
            "{working_history}": working_history,
            "{compressed_history}": compressed_history,
            "{long_term_profile}": long_term_profile,
            "{resume_available}": "true" if resume_available else "false",
            "{resume_summary}": resume_summary,
        }
        for old, new in replacements.items():
            system_prompt = system_prompt.replace(old, new)
        system_prompt += candidate_section

        try:
            raw = await self.llm.generate(
                prompt="请基于以上信息做最终意图裁决，严格输出 JSON：",
                system=system_prompt,
                temperature=0.2,
                max_tokens=1800 if all_candidates else 1200,
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

        # 支持多意图格式（intents 数组）
        if "intents" in data and isinstance(data["intents"], list) and data["intents"]:
            first = data["intents"][0]
            intent_str = first.get("intent", "CHAT")
            try:
                intent = LLMIntentType(intent_str)
            except ValueError:
                intent = LLMIntentType.CHAT
            confidence = max(0.0, min(1.0, float(first.get("confidence", 0.0))))
            slots = first.get("slots", {})
            slot_sources = first.get("slot_sources", {})
            missing_slots = first.get("missing_slots", [])
            needs_clarification = bool(data.get("needs_clarification", False))
            if confidence < 0.7:
                needs_clarification = True
            # 提取额外意图
            additional_intents = []
            for item in data["intents"][1:]:
                add_intent_str = item.get("intent", "")
                try:
                    add_intent = LLMIntentType(add_intent_str)
                except ValueError:
                    continue
                additional_intents.append({
                    "intent": add_intent,
                    "confidence": max(0.0, min(1.0, float(item.get("confidence", 0.0)))),
                    "slots": item.get("slots", {}),
                    "slot_sources": item.get("slot_sources", {}),
                    "missing_slots": item.get("missing_slots", []),
                })
            return FallbackResult(
                intent=intent,
                confidence=confidence,
                reason=data.get("reason", ""),
                slots=slots,
                slot_sources=slot_sources,
                missing_slots=missing_slots,
                needs_clarification=needs_clarification,
                clarification_question=data.get("clarification_question", ""),
                clarification_options=data.get("clarification_options", []),
                arbitration=data.get("arbitration", "独立判断"),
                candidate_options=data.get("candidate_options", []),
                additional_intents=additional_intents,
            )
        
        # 单一意图格式（向后兼容）
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
    # 统一兜底入口：classify_multi（支持多意图）
    # ═══════════════════════════════════════════════════════

    async def classify_multi(
        self,
        calibrated_candidates: list,
        rewrite_result: QueryRewriteResult,
        session: SessionMemory,
    ) -> MultiIntentResult:
        """统一大模型兜底：支持多意图裁决
        
        将多个候选意图的信息传给 classify()，让大模型做统一裁决，
        然后转换为 MultiIntentResult。
        """
        if not calibrated_candidates:
            return MultiIntentResult(
                candidates=[],
                primary_intent=LLMIntentType.CHAT,
                needs_clarification=True,
                clarification_reason="没有候选意图，需要用户明确表达需求",
            )
        
        # 选择 confidence 最高的候选作为代表
        representative = max(calibrated_candidates, key=lambda c: c.confidence)
        
        rule_result = representative.rule_result or LLMRuleResult(
            intent=representative.intent_type,
            strength=RuleStrength.WEAK,
            rule_name="multi_fallback",
            metadata={},
            trigger="",
        )
        
        cal_result = CalibrationResult(
            intent=representative.intent_type,
            confidence=representative.confidence,
            reason=f"多意图代表候选 | total_candidates={len(calibrated_candidates)}",
            slots=representative.slots,
            slot_sources=representative.slot_sources,
            missing_slots=representative.missing_slots,
            needs_clarification=representative.needs_clarification,
            clarification_question=representative.clarification_question,
            clarification_options=representative.clarification_options,
            rule_agreement=representative.rule_agreement,
            needs_llm_fallback=True,
        )
        
        # 调用统一兜底 classify，传入所有候选
        fb_result = await self.classify(
            rule_result=rule_result,
            cal_result=cal_result,
            rewrite_result=rewrite_result,
            session=session,
            all_candidates=calibrated_candidates,
        )
        
        # 将 FallbackResult 转换为 MultiIntentResult
        candidates = []
        
        # 主意图
        main_candidate = IntentCandidate(
            intent_type=fb_result.intent,
            confidence=fb_result.confidence,
            reason=fb_result.reason,
            slots=fb_result.slots,
            slot_sources=fb_result.slot_sources,
            missing_slots=fb_result.missing_slots,
            needs_clarification=fb_result.needs_clarification,
            clarification_question=fb_result.clarification_question,
            clarification_options=fb_result.clarification_options,
            source="llm_fallback",
            rule_result=rule_result,
            rule_agreement=True,
            execution_cost={"explore": "high", "assess": "high", "prepare": "medium", "verify": "medium", "manage": "low", "chat": "low"}.get(fb_result.intent.value, "medium"),
            dependencies=[],
        )
        candidates.append(main_candidate)
        
        # 额外意图
        for add in fb_result.additional_intents:
            add_candidate = IntentCandidate(
                intent_type=add["intent"],
                confidence=add["confidence"],
                reason=f"多意图附加: {fb_result.reason}",
                slots=add.get("slots", {}),
                slot_sources=add.get("slot_sources", {}),
                missing_slots=add.get("missing_slots", []),
                needs_clarification=False,
                source="llm_fallback_multi",
                rule_result=None,
                rule_agreement=True,
                execution_cost={"explore": "high", "assess": "high", "prepare": "medium", "verify": "medium", "manage": "low", "chat": "low"}.get(add["intent"].value, "medium"),
                dependencies=[],
            )
            # ASSESS + PREPARE 依赖关系
            if add["intent"] == LLMIntentType.PREPARE and fb_result.intent == LLMIntentType.ASSESS:
                add_candidate.dependencies.append(LLMIntentType.ASSESS)
            candidates.append(add_candidate)
        
        # 冲突消解 + 拓扑构建
        candidates = self._resolve_conflicts(candidates)
        topology = self._build_execution_topology(candidates)
        global_slots = self._merge_global_slots(candidates, session=session)
        
        return MultiIntentResult(
            candidates=candidates,
            primary_intent=candidates[0].intent_type if candidates else LLMIntentType.CHAT,
            needs_clarification=fb_result.needs_clarification,
            clarification_reason=fb_result.clarification_question if fb_result.needs_clarification else None,
            global_slots=global_slots,
            execution_topology=topology,
        )

    # ═══════════════════════════════════════════════════════
    # 多意图改造：arbitrate（硬规则部分保留，LLM兜底走 classify_multi）
    # ═══════════════════════════════════════════════════════

    async def arbitrate(
        self,
        calibrated_candidates: list,  # List[IntentCandidate]
        rewrite_result: QueryRewriteResult,
        session: SessionMemory,
    ) -> MultiIntentResult:
        """多意图统一仲裁：过滤、拓扑构建、全局槽位合并（无冲突消解，无STRONG优先）"""
        # 步骤1: 过滤低置信度
        filtered = [c for c in calibrated_candidates if c.confidence >= 0.5]
        if not filtered:
            chat_candidates = [c for c in calibrated_candidates if c.intent_type == LLMIntentType.CHAT]
            if chat_candidates:
                logger.info(f"[LLMFallbackClassifier] 分层兜底 | 校准器判为CHAT（conf<0.5），返回CHAT而非澄清")
                return MultiIntentResult(
                    candidates=chat_candidates,
                    primary_intent=LLMIntentType.CHAT,
                    needs_clarification=False,
                    clarification_reason=None,
                )
            logger.info(f"[LLMFallbackClassifier] 所有候选被过滤，调用 classify_multi 统一兜底")
            return await self.classify_multi(calibrated_candidates, rewrite_result, session)

        # 步骤2: 冲突消解（轻量硬规则，零LLM成本）
        resolved = self._resolve_conflicts(filtered)
        if len(resolved) < len(filtered):
            removed = [c.intent_type.value for c in filtered if c not in resolved]
            logger.info(f"[LLMFallbackClassifier] 冲突消解移除了: {removed}")

        # 步骤3: 构建执行拓扑
        topology = self._build_execution_topology(resolved)

        # 步骤4: 全局槽位池合并
        global_slots = self._merge_global_slots(resolved, session=session)

        # 步骤5: 澄清判断
        needs_clarification, reason = self._check_clarification_need(resolved, global_slots, session)

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

        # 规则1: MANAGE + 其他意图 → MANAGE优先执行（文件操作先完成）
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

        # 规则4: EXPLORE + VERIFY 冲突消解
        # 当 VERIFY 没有明确属性词（如薪资/要求/学历等），通常是 EXPLORE 的误匹配
        # 典型场景："帮我看看什么岗位适合我"同时命中 explore+verify，但用户意图只是 explore
        if LLMIntentType.EXPLORE in intent_types and LLMIntentType.VERIFY in intent_types:
            verify_c = next((c for c in candidates if c.intent_type == LLMIntentType.VERIFY), None)
            if verify_c:
                attrs = verify_c.slots.get("attributes", [])
                # 如果没有具体属性词，或属性只是"综合情况"/"要求"等泛化词，移除 verify
                vague_attrs = {"综合情况", "技能", "条件", "经验"}
                if not attrs or set(attrs).issubset(vague_attrs):
                    logger.info(f"[_resolve_conflicts] EXPLORE+VERIFY冲突，VERIFY属性模糊({attrs})，移除VERIFY")
                    candidates = [c for c in candidates if c.intent_type != LLMIntentType.VERIFY]
                    intent_types.discard(LLMIntentType.VERIFY)

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
        """合并所有候选意图的槽位，去重并保留（candidates 的槽位优先覆盖 session 旧值）"""
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
                # 【关键】search_keywords/rewritten_query/query 必须覆盖旧值，
                # 否则多轮对话中 _check_clarification_need 会拿到上轮 query 导致上下文引用检测失效
                if slot_key in ("search_keywords", "rewritten_query", "query"):
                    global_slots[slot_key] = slot_val
                elif slot_key not in global_slots:
                    global_slots[slot_key] = slot_val
        return global_slots

    def _check_clarification_need(self, candidates: list, global_slots: dict, session: SessionMemory = None) -> tuple:
        """多意图场景下的澄清判断（支持从 evidence_cache 补全上下文引用槽位）"""
        if all(c.confidence < 0.7 for c in candidates):
            return True, "多意图置信度均偏低，需要用户确认具体需求"
        # 只有 ASSESS 严格需要简历；EXPLORE/PREPARE 可以在无简历时工作
        assess_only = [c for c in candidates if c.intent_type == LLMIntentType.ASSESS]
        if assess_only and not global_slots.get("resume_available"):
            return True, "分析匹配度需要简历信息，请先上传简历"
        # VERIFY 缺少 company 和 position 时需要澄清
        verify_candidates = [c for c in candidates if c.intent_type == LLMIntentType.VERIFY]
        # 多意图并行时（如 EXPLORE+VERIFY），VERIFY 槽位可由主意图结果后验填充，不触发全局澄清
        has_primary_intent = any(
            c.intent_type not in (LLMIntentType.VERIFY, LLMIntentType.CHAT)
            for c in candidates
        )
        for vc in verify_candidates:
            has_company = bool(vc.slots.get("company") or global_slots.get("company"))
            has_position = bool(vc.slots.get("position") or global_slots.get("position"))
            # 上下文引用检测：从 slots 或 global_slots 中获取 search_keywords
            search_kw = vc.slots.get("search_keywords", "") or global_slots.get("search_keywords", "")
            has_context_ref = any(ref in search_kw for ref in ["上面", "那个", "这个", "刚才", "之前"])
            pass
            # 【关键】检测到上下文引用且缺槽位时，尝试从 evidence_cache 补全
            if has_context_ref and (not has_company or not has_position):
                if session and hasattr(session, "evidence_cache") and session.evidence_cache:
                    for chunk in session.evidence_cache[:3]:
                        meta = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
                        if not has_company and meta.get("company"):
                            vc.slots["company"] = meta["company"]
                            has_company = True
                        if not has_position and meta.get("position"):
                            vc.slots["position"] = meta["position"]
                            has_position = True
                        if has_company and has_position:
                            break
                    if has_company or has_position:
                        logger.info(
                            f"[LLMFallbackClassifier] 上下文引用槽位补全 | "
                            f"company={vc.slots.get('company')} position={vc.slots.get('position')}"
                        )
            # 伴随意图场景：VERIFY 缺槽位不拖累全局
            if has_primary_intent and not has_company and not has_position and not has_context_ref:
                continue
            if not has_company and not has_position and not has_context_ref:
                return True, "您想了解哪个公司的什么信息？请提供公司名称。"

        # ASSESS / PREPARE 缺少 company 和 position 时也需要澄清
        for intent_type in (LLMIntentType.ASSESS, LLMIntentType.PREPARE):
            typed_candidates = [c for c in candidates if c.intent_type == intent_type]
            for tc in typed_candidates:
                has_company = bool(tc.slots.get("company") or global_slots.get("company"))
                has_position = bool(tc.slots.get("position") or global_slots.get("position"))
                search_kw = tc.slots.get("search_keywords", "") or global_slots.get("search_keywords", "")
                has_context_ref = any(ref in search_kw for ref in ["上面", "那个", "这个", "刚才", "之前"])
                # 上下文引用尝试补全
                if has_context_ref and (not has_company or not has_position):
                    if session and hasattr(session, "evidence_cache") and session.evidence_cache:
                        for chunk in session.evidence_cache[:3]:
                            meta = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
                            if not has_company and meta.get("company"):
                                tc.slots["company"] = meta["company"]
                                has_company = True
                            if not has_position and meta.get("position"):
                                tc.slots["position"] = meta["position"]
                                has_position = True
                            if has_company and has_position:
                                break
                if not has_company and not has_position and not has_context_ref:
                    intent_name = "分析匹配度" if intent_type == LLMIntentType.ASSESS else "准备面试"
                    return True, f"您想{intent_name}的是哪个公司、哪个岗位？请提供具体信息。"

        if len(candidates) > 3:
            return True, "检测到多个意图，请确认优先级"
        return False, None

    def _needs_llm_arbitration(self, candidates: list) -> bool:
        """判断是否需要大模型介入仲裁"""
        if len(candidates) <= 1:
            return False
        has_deps = any(len(c.dependencies) > 0 for c in candidates)
        has_cost_conflict = len([c for c in candidates if c.execution_cost == "high"]) >= 2
        return has_deps or has_cost_conflict

# ═══════════════════════════════════════════════════════
# 11. LLM 意图路由（三级流水线入口）
# ═══════════════════════════════════════════════════════

class LLMIntentRouter:
    """LLM 路线意图路由统一入口"""

    def __init__(self, planner_llm: Optional[LLMClient] = None, chat_llm: Optional[LLMClient] = None):
        self.rule_registry = _create_rule_registry()
        self.fallback = LLMFallbackClassifier(llm_client=chat_llm)

    async def route(
        self,
        rewrite_result: QueryRewriteResult,
        session: SessionMemory,
        attachments: list,
        raw_message: Optional[str] = None,
    ) -> LLMIntentResult:
        """单意图路由：规则引擎 → [直接采信 | 大模型兜底]"""
        # ① 规则引擎
        msg_for_rules = raw_message or rewrite_result.rewritten_query
        rule_result = self.rule_registry.classify(msg_for_rules, attachments)
        logger.info(
            f"[LLMIntentRouter] 规则结果 | intent={rule_result.intent.value if rule_result.intent else 'None'} | "
            f"strength={rule_result.strength} | rule={rule_result.rule_name}"
        )

        # ② 基于规则结果直接生成 CalibrationResult（跳过小模型）
        intent = rule_result.intent or LLMIntentType.CHAT
        confidence = 0.88 if rule_result.strength == RuleStrength.STRONG else (0.75 if rule_result.strength == RuleStrength.WEAK else 0.55)
        cal_result = CalibrationResult(
            intent=intent,
            confidence=confidence,
            reason=f"规则直接采信({rule_result.strength}): {rule_result.rule_name}",
            slots=rule_result.metadata or {},
            slot_sources={k: "rule_extraction" for k in (rule_result.metadata or {}).keys()},
            rule_agreement=True,
        )

        # ③ 规则 STRONG/WEAK 直接采信
        if rule_result.strength in (RuleStrength.STRONG, RuleStrength.WEAK) and confidence >= 0.7:
            return LLMIntentResult(
                intent=intent,
                confidence=confidence,
                layer="rule_direct",
                rule_result=rule_result,
                calibration_result=cal_result,
                reason=cal_result.reason,
                slots=cal_result.slots,
                slot_sources=cal_result.slot_sources,
                missing_slots=[],
                needs_clarification=False,
            )

        # ④ 规则 MISS → 大模型兜底
        logger.info(f"[LLMIntentRouter] 规则MISS，触发大模型兜底")
        fb_result = await self.fallback.classify(rule_result, cal_result, rewrite_result, session)
        logger.info(
            f"[LLMIntentRouter] 兜底结果 | intent={fb_result.intent.value} | "
            f"confidence={fb_result.confidence:.2f} | arbitration={fb_result.arbitration}"
        )

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
        """多意图识别流水线入口：RuleRegistry → LLMFallbackClassifier（大模型兜底）"""
        # ① 规则引擎（优先使用 QueryRewrite 结果，更好地处理多轮对话）
        msg_for_rules = rewrite_result.rewritten_query or raw_message
        # 同时用原始消息做二次匹配，捕获被 QueryRewriter 丢失的多意图
        fallback_message = rewrite_result.original_message if rewrite_result.original_message != msg_for_rules else None
        rule_matches = self.rule_registry.classify_all(
            msg_for_rules, 
            attachments,
            fallback_message=fallback_message
        )
        logger.info(
            f"[LLMIntentRouter-Multi] 规则结果 | matches={len(rule_matches)} | "
            f"intents={[r.intent.value if r.intent else 'None' for r in rule_matches]} | "
            f"strengths={[r.strength for r in rule_matches]}"
        )

        # ①b 规则全部 MISS 时，强制创建虚拟 CHAT 候选送入校准器
        # 关键修正：strength 设为 MISS（而非 WEAK），让校准器知道这是"完全无规则参考"，
        # 必须做纯语义判断，不能把 CHAT 当作有依据的参考来确认
        has_non_miss = any(r.intent is not None for r in rule_matches)
        if not has_non_miss:
            virtual_rule = LLMRuleResult(
                intent=LLMIntentType.CHAT,
                strength=RuleStrength.MISS,  # ← 修正：从 WEAK 改为 MISS
                rule_name="miss_virtual_chat",
                metadata={},
                trigger="",
            )
            rule_matches.append(virtual_rule)
            logger.info(
                f"[LLMIntentRouter-Multi] 规则全部MISS，强制创建虚拟CHAT候选（strength=MISS）送入校准器"
            )

        # ② 规则直接采信（跳过小模型校准）
        calibrated = []
        for rule_result in rule_matches:
            candidate = self._direct_rule_candidate(rule_result)
            self._inject_resolved_references(candidate, rewrite_result, session)
            calibrated.append(candidate)
        calibrated.sort(key=lambda c: c.confidence, reverse=True)
        logger.info(
            f"[LLMIntentRouter-Multi] 规则直接采信 | candidates={len(calibrated)} | "
            f"intents={[c.intent_type.value for c in calibrated]} | "
            f"confidences={[f'{c.confidence:.2f}' for c in calibrated]}"
        )

        # ②c clarify 场景后处理：校准器返回 CHAT 但工作记忆显示上轮澄清时，
        # 根据上轮语义和本轮补充的实体硬编码修正意图
        if session and hasattr(session, "pending_clarification") and session.pending_clarification:
            pc = session.pending_clarification
            # missing_slots 非空，或 resolved_slots 为空（上轮意图模糊未解析出实体）
            is_clarify_scenario = bool(pc.missing_slots) or not bool(pc.resolved_slots)
            # 旧架构意图名 → 新架构映射（chat.py 中 pending_intent 存储的是旧名如 "match_assess"）
            _OLD_INTENT_MAP = {
                "match_assess": "assess",
                "attribute_verify": "verify",
                "interview_prepare": "prepare",
                "global_explore": "explore",
                "file_manage": "manage",
                "general_chat": "chat",
            }
            mapped_pending = _OLD_INTENT_MAP.get(pc.pending_intent, pc.pending_intent)
            if mapped_pending in ("chat", "assess", "verify") and is_clarify_scenario:
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
                            # 上轮语义关键词分类（keywords 兜底，mapped_pending 优先）
                            assess_keywords = ["匹配", "适合", "差距", "契合", "够格", "搭不搭"]
                            verify_keywords = ["薪资", "工资", "要求", "学历", "经验", "加班", "福利", "地点", "怎么样", "分析", "介绍", "了解", "看看"]
                            if mapped_pending == "assess" or any(kw in last_user_msg for kw in assess_keywords):
                                c.intent_type = LLMIntentType.ASSESS
                                c.confidence = 0.82
                                c.reason = f"clarify场景修正：上轮'{last_user_msg}'触发ASSESS澄清，本轮补充实体，推断为ASSESS"
                                c.slots["company"] = c.slots.get("company") or rewrite_result.resolved_references.get("company") or rewrite_result.resolved_references.get("__correct_company__")
                                c.slots["position"] = c.slots.get("position") or rewrite_result.resolved_references.get("position") or rewrite_result.resolved_references.get("__correct_position__")
                                c.slots["attributes"] = c.slots.get("attributes", ["匹配度"])
                                c.slots["jd_source"] = "kb"
                                c.slots["search_keywords"] = rewrite_result.rewritten_query
                                logger.info(f"[LLMIntentRouter-Multi] clarify场景修正 CHAT→ASSESS | slots={c.slots}")
                            elif mapped_pending == "verify" or any(kw in last_user_msg for kw in verify_keywords):
                                c.intent_type = LLMIntentType.VERIFY
                                c.confidence = 0.82
                                c.reason = f"clarify场景修正：上轮'{last_user_msg}'触发VERIFY澄清，本轮补充实体，推断为VERIFY（综合查询）"
                                c.slots["company"] = c.slots.get("company") or rewrite_result.resolved_references.get("company") or rewrite_result.resolved_references.get("__correct_company__")
                                c.slots["position"] = c.slots.get("position") or rewrite_result.resolved_references.get("position") or rewrite_result.resolved_references.get("__correct_position__")
                                # 综合查询：attributes 默认填充为综合情况，不强制要求具体属性词
                                c.slots["attributes"] = c.slots.get("attributes", ["综合情况"])
                                c.slots["qa_type"] = "factual"
                                c.slots["search_keywords"] = rewrite_result.rewritten_query
                                logger.info(f"[LLMIntentRouter-Multi] clarify场景修正 CHAT→VERIFY | slots={c.slots}")

        # ②b 检查是否触发大模型兜底 或 直接返回 clarification
        # 触发条件：① 小模型推翻规则判 CHAT；② 任意候选 confidence < 0.7；③ 低置信度 clarification
        # 高置信度 clarification 直接返回，不走兜底
        needs_fallback = False
        fallback_reasons = []
        
        for c in calibrated:
            # 检测 clarification 请求
            is_clarification = c.needs_clarification
            if not is_clarification and c.intent_type == LLMIntentType.CHAT:
                reason_lower = (c.reason or "").lower()
                if any(kw in reason_lower for kw in ["clarification", "澄清", "再说一遍", "重复", "没懂", "什么意思", "详细说说", "解释一下"]):
                    is_clarification = True
            
            # 高置信度 clarification：只处理 CHAT 类型的主动澄清请求
            # 非 CHAT 候选的 needs_clarification 表示缺槽位，应由 _check_clarification_need 统一判断
            if is_clarification and c.confidence >= 0.7 and c.intent_type == LLMIntentType.CHAT:
                logger.info(
                    f"[LLMIntentRouter-Multi] 高置信度 CHAT-clarification 请求，直接返回 | "
                    f"reason={c.reason[:60]}"
                )
                return MultiIntentResult(
                    candidates=[c],
                    primary_intent=LLMIntentType.CHAT,
                    needs_clarification=True,
                    clarification_reason="用户请求澄清/解释",
                )
            
            # 低置信度 clarification：标记为兜底原因
            if is_clarification and c.confidence < 0.7:
                needs_fallback = True
                fallback_reasons.append(f"clarification_low_conf({c.confidence:.2f})")
                continue
            
            rr = c.rule_result
            # 条件1: 规则非CHAT，但小模型判CHAT
            if rr and rr.intent and rr.intent != LLMIntentType.CHAT and c.intent_type == LLMIntentType.CHAT:
                needs_fallback = True
                fallback_reasons.append(f"{rr.intent.value}_overridden_to_chat")
            # 条件2: confidence 低于阈值
            if c.confidence < 0.7:
                needs_fallback = True
                fallback_reasons.append(f"{c.intent_type.value}_low_conf({c.confidence:.2f})")
        
        if needs_fallback:
            logger.info(
                f"[LLMIntentRouter-Multi] 触发大模型兜底: {','.join(fallback_reasons)} | "
                f"candidates={[c.intent_type.value for c in calibrated]}"
            )
            return await self.fallback.classify_multi(calibrated, rewrite_result, session)
        
        # ③ 不触发兜底：直接调用简化版仲裁（无冲突消解，无STRONG优先）
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

    # ═══════════════════════════════════════════════════════
    # 规则直接转候选（原 SmallModelCalibrator 迁移）
    # ═══════════════════════════════════════════════════════

    def _direct_rule_candidate(self, rule_result: LLMRuleResult) -> IntentCandidate:
        """直接基于规则结果生成候选（跳过小模型）"""
        intent = rule_result.intent or LLMIntentType.CHAT
        cost_map = {
            LLMIntentType.ASSESS: "high", LLMIntentType.EXPLORE: "high",
            LLMIntentType.PREPARE: "medium", LLMIntentType.VERIFY: "medium",
            LLMIntentType.MANAGE: "low", LLMIntentType.CHAT: "low",
        }
        dep_map = {
            LLMIntentType.PREPARE: [LLMIntentType.ASSESS],
        }
        slots = rule_result.metadata or {}
        
        # 根据规则强度设置 confidence
        if rule_result.strength == RuleStrength.STRONG:
            confidence = 0.88
        elif rule_result.strength == RuleStrength.WEAK:
            confidence = 0.75
        else:  # MISS
            confidence = 0.55
        
        # 推断 missing_slots（仅 ASSESS/VERIFY 需要 company+position）
        missing_slots = []
        if intent in (LLMIntentType.ASSESS, LLMIntentType.VERIFY):
            if not slots.get("company") and not slots.get("position"):
                missing_slots.extend(["company", "position"])
            elif not slots.get("company"):
                missing_slots.append("company")
            elif not slots.get("position"):
                missing_slots.append("position")
        
        if intent == LLMIntentType.VERIFY and not slots.get("attributes"):
            has_entity = bool(slots.get("company") or slots.get("position"))
            if not has_entity:
                missing_slots.append("attributes")
        
        if intent == LLMIntentType.EXPLORE and not slots.get("search_keywords"):
            missing_slots.append("search_keywords")
        
        return IntentCandidate(
            intent_type=intent,
            confidence=confidence,
            reason=f"规则直接采信({rule_result.strength}): {rule_result.rule_name} | trigger={rule_result.trigger}",
            slots=slots,
            slot_sources={k: "rule_extraction" for k in slots.keys()},
            missing_slots=missing_slots,
            needs_clarification=bool(missing_slots),
            source="rule_direct",
            rule_result=rule_result,
            rule_agreement=True,
            execution_cost=cost_map.get(intent, "medium"),
            dependencies=dep_map.get(intent, []),
            can_parallel=True,
        )

    def _inject_resolved_references(self, candidate: IntentCandidate, rewrite_result: QueryRewriteResult, session: SessionMemory):
        """将 QueryRewriter 的指代消解结果和 evidence_cache 注入 candidate slots"""
        from app.core.query_rewrite import QueryRewriter
        
        refs = rewrite_result.resolved_references or {}
        
        # 1. 从 rewrite_result.resolved_references 注入 company/position
        # 1a. 标准键直接注入
        for key in ["company", "position", "__correct_company__", "__correct_position__", "__switch_company__", "__switch_position__"]:
            if refs.get(key) and key not in ["company", "position"]:
                target_key = key.replace("__correct_", "").replace("__switch_", "").replace("__", "")
                if target_key in ("company", "position") and not candidate.slots.get(target_key):
                    candidate.slots[target_key] = refs[key]
            elif refs.get(key) and not candidate.slots.get(key):
                candidate.slots[key] = refs[key]
        
        # 1b. resolved_references 中的值可能是 company+position 组合（如"百度 AI产品实习生"）
        # 或者纯 position（如"AI产品实习生"），需要解析后注入标准槽位
        if not candidate.slots.get("company") or not candidate.slots.get("position"):
            for ref_word, ref_value in refs.items():
                if not ref_value or not isinstance(ref_value, str):
                    continue
                # 尝试从值中提取 company（知识库中的公司名列表）
                if not candidate.slots.get("company"):
                    kb_companies, _ = _load_kb_entities()
                    for c in sorted(kb_companies, key=len, reverse=True):
                        if c in ref_value:
                            candidate.slots["company"] = c
                            break
                # 尝试从值中提取 position（知识库中的岗位名列表）
                if not candidate.slots.get("position"):
                    _, kb_positions = _load_kb_entities()
                    for p in sorted(kb_positions, key=len, reverse=True):
                        if p in ref_value:
                            candidate.slots["position"] = p
                            break
                # 如果值中没有匹配到标准实体，但整体看起来像 position（含"实习生""工程师"等后缀）
                if not candidate.slots.get("position"):
                    position_suffixes = ["实习生", "工程师", "产品经理", "运营", "设计师", "分析师", "开发", "测试"]
                    for suffix in position_suffixes:
                        if suffix in ref_value:
                            candidate.slots["position"] = ref_value.strip()
                            break
                if candidate.slots.get("company") and candidate.slots.get("position"):
                    break
        
        # 2. 从 evidence_cache 兜底补全（解决 LLM 改写失败、规则也无法解析的场景）
        if session and hasattr(session, "evidence_cache") and session.evidence_cache:
            needs_company = not candidate.slots.get("company")
            needs_position = not candidate.slots.get("position")
            if needs_company or needs_position:
                for chunk in session.evidence_cache[:3]:
                    meta = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
                    if needs_company and meta.get("company"):
                        candidate.slots["company"] = meta["company"]
                        needs_company = False
                    if needs_position and meta.get("position"):
                        candidate.slots["position"] = meta["position"]
                        needs_position = False
                    if not needs_company and not needs_position:
                        break
                if candidate.slots.get("company") or candidate.slots.get("position"):
                    logger.info(
                        f"[LLMIntentRouter] 指代消解槽位注入 | "
                        f"intent={candidate.intent_type.value} | "
                        f"company={candidate.slots.get('company')} | "
                        f"position={candidate.slots.get('position')} | "
                        f"source=evidence_cache"
                    )
        
        # 3. 重新计算 missing_slots（注入后可能不再缺失）
        if candidate.intent_type in (LLMIntentType.ASSESS, LLMIntentType.VERIFY, LLMIntentType.PREPARE):
            missing = []
            if not candidate.slots.get("company") and not candidate.slots.get("position"):
                missing.extend(["company", "position"])
            elif not candidate.slots.get("company"):
                missing.append("company")
            elif not candidate.slots.get("position"):
                missing.append("position")
            if candidate.intent_type == LLMIntentType.VERIFY and not candidate.slots.get("attributes"):
                if not candidate.slots.get("company") and not candidate.slots.get("position"):
                    missing.append("attributes")
            candidate.missing_slots = missing
            candidate.needs_clarification = bool(missing)
