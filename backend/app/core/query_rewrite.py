"""
Query改写模块 v2

职责：
1. 指代消解（resolved_references）
2. 口语降噪（口语化→结构化query）
3. 追问补全（根据follow_up_type继承/修正上轮槽位）
4. 提取search_keywords

不做：
- 意图模糊判断
- 实体缺失判断
- 澄清决策

输入：用户原始消息 + 三层记忆 + history_slots
输出：QueryRewriteResult
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.core.llm_client import LLMClient, TIMEOUT_LIGHT
from app.core.memory import SessionMemory

logger = logging.getLogger(__name__)


@dataclass
class QueryRewriteResult:
    """Query改写结果"""
    rewritten_query: str = ""              # 改写后的结构化query
    search_keywords: str = ""              # 检索关键词（空格分隔）
    resolved_references: Dict[str, str] = field(default_factory=dict)  # 指代映射
    is_follow_up: bool = False             # 是否追问
    follow_up_type: str = "none"           # none / expand / switch / clarify
    original_message: str = ""             # 保留原始消息


class QueryRewriter:
    """
    Query改写器。
    基于规则做follow_up检测和简单的指代消解。
    复杂的语义改写由L2/L3在意图识别阶段处理。
    """

    # 切换词
    _SWITCH_WORDS = ["不说", "不看", "不聊", "换", "那看看", "另外", "再看看", "对比", "比较一下"]
    # 澄清/修正词
    _CLARIFY_WORDS = ["不对", "我说的是", "纠正", "不是", "错了", "应该是", "其实"]
    # 指代词
    _REFERENCES = ["那个", "这个", "它", "他", "那边", "刚才", "之前", "上一轮"]
    # 通用职位词
    _GENERIC_POSITIONS = ["岗位", "职位", "工作", "机会"]

    def __init__(self):
        pass

    async def rewrite(
        self,
        raw_query: str,
        session: SessionMemory,
    ) -> QueryRewriteResult:
        """
        主入口：改写用户query。
        优先使用 LLM 做 follow-up 检测和指代消解，失败则回退到规则。
        """
        result = QueryRewriteResult(original_message=raw_query)

        # 1. 获取上轮槽位（用于follow_up判断和指代消解）
        history_slots = self._get_history_slots(session)

        # 2. 获取最近3轮对话上下文
        history_context = self._get_history_context(session)

        # 3. 优先尝试 LLM-based 改写
        llm_result = await self._llm_rewrite(raw_query, history_context, history_slots)
        if llm_result:
            result.is_follow_up = llm_result["is_follow_up"]
            result.follow_up_type = llm_result["follow_up_type"]
            result.resolved_references = llm_result["resolved_references"]
            result.rewritten_query = llm_result["rewritten_query"]
        else:
            # 回退到规则-based
            result.is_follow_up, result.follow_up_type = self._detect_follow_up(
                raw_query, history_slots
            )
            result.resolved_references = self._resolve_references(
                raw_query, history_slots, result.follow_up_type
            )
            result.rewritten_query = self._build_rewritten_query(
                raw_query, result.resolved_references, history_slots, result.follow_up_type
            )

        # 4. 提取search_keywords
        result.search_keywords = self._extract_search_keywords(result.rewritten_query)

        logger.info(
            f"[QueryRewrite] '{raw_query[:30]}...' -> '{result.rewritten_query[:40]}...' "
            f"| follow_up={result.is_follow_up}/{result.follow_up_type} "
            f"| refs={result.resolved_references}"
        )
        return result

    def _get_history_context(self, session: SessionMemory) -> str:
        """获取最近3轮对话上下文文本"""
        if not session or not hasattr(session, "working_memory"):
            return ""
        return session.working_memory.get_recent_context(n=3, exclude_last=True)

    def _get_history_slots(self, session: SessionMemory) -> Dict[str, str]:
        """获取上轮已确认的槽位状态"""
        if not session or not hasattr(session, "working_memory"):
            return {}

        # 从working_memory获取最近一轮的global_slots
        turns = session.working_memory.turns if hasattr(session.working_memory, "turns") else []
        if not turns:
            return {}

        # 取最近一轮的槽位（实际存储位置取决于memory实现）
        last_turn = turns[-1]
        # 如果turn对象有global_slots属性
        if hasattr(last_turn, "global_slots"):
            return {k: v for k, v in last_turn.global_slots.items() if v is not None}

        # 兜底：从session的global_slots获取（如果存在）
        if hasattr(session, "global_slots"):
            return {k: v for k, v in session.global_slots.items() if v is not None}

        return {}

    def _detect_follow_up(
        self, raw_query: str, history_slots: Dict[str, str]
    ) -> Tuple[bool, str]:
        """
        基于规则判断follow_up类型。
        返回：(is_follow_up, follow_up_type)

        注意：当没有对话历史（history_slots 为空）时，无法判断是否为追问，
        安全默认返回 (False, "none")，表示这是一个全新的查询，直接返回原始 query。
        这是预期行为，不是 bug。
        """
        if not history_slots:
            return False, "none"

        # 规则1：含切换词 → switch
        for w in self._SWITCH_WORDS:
            if w in raw_query:
                return True, "switch"

        # 规则2：含澄清/修正词 → clarify
        for w in self._CLARIFY_WORDS:
            if w in raw_query:
                return True, "clarify"

        # 规则3：query极短（<12字）且有历史槽位 → expand
        if len(raw_query) < 12:
            return True, "expand"

        # 规则4：含指代词且有历史槽位 → expand
        for w in self._REFERENCES:
            if w in raw_query:
                return True, "expand"

        # 规则5：query中没有任何实体但有历史槽位 → 可能是expand
        has_entity = self._has_any_entity(raw_query)
        if not has_entity and history_slots:
            return True, "expand"

        return False, "none"

    def _has_any_entity(self, query: str) -> bool:
        """简单判断query中是否包含明显的实体信息"""
        # 公司名关键词（简化版，实际可接入更全的词典）
        company_keywords = ["字节", "百度", "阿里", "腾讯", "美团", "京东", "小米", "拼多多", "网易"]
        for c in company_keywords:
            if c in query:
                return True
        # 岗位关键词
        position_keywords = ["工程师", "产品经理", "算法", "后端", "前端", "测试", "运营", "设计", "开发"]
        for p in position_keywords:
            if p in query:
                return True
        # 属性关键词
        attr_keywords = ["薪资", "工资", "加班", "福利", "地点", "要求", "匹配", "面试"]
        for a in attr_keywords:
            if a in query:
                return True
        return False

    def _resolve_references(
        self, raw_query: str, history_slots: Dict[str, str], follow_up_type: str
    ) -> Dict[str, str]:
        """
        指代消解 + 槽位补全。
        返回：指代映射表 {指代词: 解析后的实体值}
        """
        resolved = {}

        if follow_up_type == "none" or not history_slots:
            return resolved

        if follow_up_type == "expand":
            # 指代消解：将"那个/这个"映射到历史槽位中的实体
            for ref_word in self._REFERENCES:
                if ref_word in raw_query:
                    # 判断指代的是哪个槽位
                    # 简单策略：如果有company就指代company+position的组合
                    if "company" in history_slots:
                        target = history_slots["company"]
                        if "position" in history_slots:
                            target += " " + history_slots["position"]
                        resolved[ref_word] = target
                    elif "position" in history_slots:
                        resolved[ref_word] = history_slots["position"]
                    break

            # 通用职位词指代："这个岗位"→历史position
            for generic in self._GENERIC_POSITIONS:
                if generic in raw_query and "position" in history_slots:
                    resolved[generic] = history_slots["position"]

        elif follow_up_type == "switch":
            # 提取新的公司/岗位名，标记为替换
            new_company = self._extract_company(raw_query)
            if new_company:
                resolved["__switch_company__"] = new_company
            new_position = self._extract_position(raw_query)
            if new_position:
                resolved["__switch_position__"] = new_position

        elif follow_up_type == "clarify":
            # 提取修正后的实体
            new_company = self._extract_company(raw_query)
            if new_company:
                resolved["__correct_company__"] = new_company
            new_position = self._extract_position(raw_query)
            if new_position:
                resolved["__correct_position__"] = new_position

        return resolved

    def _build_rewritten_query(
        self,
        raw_query: str,
        resolved_refs: Dict[str, str],
        history_slots: Dict[str, str],
        follow_up_type: str,
    ) -> str:
        """构建改写后的结构化query"""
        rewritten = raw_query

        # 替换指代词
        for ref_word, target in resolved_refs.items():
            if not ref_word.startswith("__"):  # 跳过内部标记
                rewritten = rewritten.replace(ref_word, target)

        # 根据follow_up_type融合历史槽位
        if follow_up_type == "expand":
            # expand模式下，把历史槽位补全到query中（如果当前没提到）
            slots_to_inherit = ["company", "position"]
            for slot in slots_to_inherit:
                if slot in history_slots and slot not in rewritten:
                    # 如果query中没有这个实体，继承
                    pass  # 不修改rewritten，由意图识别层使用history_slots补全

        elif follow_up_type == "switch":
            # switch模式下，已提取的新实体已在resolved_refs中
            pass

        elif follow_up_type == "clarify":
            # clarify模式下，修正后的实体已提取
            pass

        # 口语降噪
        rewritten = self._denoise(rewritten)

        return rewritten.strip()

    def _denoise(self, query: str) -> str:
        """口语降噪：去除冗余词、规范化表达"""
        # 去除常见口语前缀
        prefixes = ["我想问一下", "请问一下", "那个", "就是", "嗯", "啊"]
        for p in prefixes:
            if query.startswith(p):
                query = query[len(p):].strip("，,。. ")

        # 规范化常见表达
        replacements = {
            "咋样": "怎么样",
            "啥": "什么",
            "咋": "怎么",
            "投": "投递",
            "面": "面试",
        }
        for old, new in replacements.items():
            query = query.replace(old, new)

        return query

    def _extract_search_keywords(self, rewritten_query: str) -> str:
        """从改写后的query中提取检索关键词"""
        # 简单实现：去掉常见疑问词和助词，保留核心名词
        stop_words = ["请问", "我想知道", "能告诉我", "怎么样", "吗", "呢", "吧", "啊", "？", "?"]
        keywords = rewritten_query
        for sw in stop_words:
            keywords = keywords.replace(sw, " ")

        # 压缩多余空格
        keywords = " ".join(keywords.split())
        return keywords.strip()

    def _extract_company(self, query: str) -> Optional[str]:
        """简单提取公司名（规则版）"""
        company_keywords = ["字节跳动", "字节", "百度", "阿里", "腾讯", "美团", "京东", "小米", "拼多多", "网易", "华为", "快手", "滴滴", "B站", "哔哩哔哩"]
        for c in company_keywords:
            if c in query:
                return c
        return None

    async def _llm_rewrite(
        self,
        raw_query: str,
        history_context: str,
        history_slots: Dict[str, str],
    ) -> Optional[Dict]:
        """
        使用 LLM 进行 follow-up 检测、指代消解和口语降噪。
        返回结构化结果，失败返回 None（由上层回退到规则）。
        """
        # 即使首轮对话也应走 LLM 改写（口语降噪），不跳过

        system_prompt = """你是对话理解助手。根据用户当前问题和最近对话历史，判断：
1. follow_up_type：当前问题与历史的关系
   - "expand"：追问/展开（如"具体怎么做""薪资呢""那个岗位加班吗"）
   - "switch"：切换话题/实体（如"那看看百度呢""换成后端"）
   - "clarify"：澄清/修正上轮系统的回答（如"不对，我说的是算法岗""纠正一下"）
   - "none"：全新问题，与历史无关。注意：如果对话历史为空，则必须是"none"

2. resolved_references：指代消解映射
   - 如果用户用"那个""这个""它""刚才"等指代历史中的实体，给出映射 {指代词: 具体实体}
   - 没有指代则返回空对象 {}

3. rewritten_query：改写后的完整query（将指代替换为具体实体，去除口语冗余）
   重要原则：
   - 不要改变用户的原始意图
   - 不要补全用户没有提到的信息
   - 不要把它变成疑问句（如"哪个岗位"），保持原意
   - 首轮对话中用户说"分析一下这个岗"，改写后应为"分析一下这个岗位"，不要变成"请具体分析一下哪个岗位"

必须输出合法JSON，不要markdown代码块：
{"follow_up_type": "...", "resolved_references": {...}, "rewritten_query": "..."}"""

        user_prompt = f"""最近对话历史：
{history_context if history_context else "（无）"}

历史已确认槽位：{json.dumps(history_slots, ensure_ascii=False)}

当前用户问题：{raw_query}

请输出JSON。"""

        try:
            llm = LLMClient.from_config("rewrite")
            raw = await llm.generate(
                prompt=user_prompt,
                system=system_prompt,
                temperature=0.1,
                max_tokens=512,
                json_mode=True,
                timeout=60.0,
            )
            parsed = LLMClient.safe_parse_json(raw)
            if not parsed:
                logger.warning(f"[QueryRewrite] LLM 返回无效 JSON，回退到规则: {raw[:200]}")
                return None

            follow_up_type = parsed.get("follow_up_type", "none")
            if follow_up_type not in ("expand", "switch", "clarify", "none"):
                follow_up_type = "none"
            
            # 兜底：首轮对话（无历史上下文）强制 follow_up_type="none"
            if not history_context and follow_up_type != "none":
                logger.info(f"[QueryRewrite] 首轮对话被LLM误判为 {follow_up_type}，强制修正为 none")
                follow_up_type = "none"

            resolved_references = parsed.get("resolved_references", {})
            if not isinstance(resolved_references, dict):
                resolved_references = {}

            rewritten_query = parsed.get("rewritten_query", raw_query)
            if not isinstance(rewritten_query, str):
                rewritten_query = raw_query

            is_follow_up = follow_up_type != "none"

            return {
                "is_follow_up": is_follow_up,
                "follow_up_type": follow_up_type,
                "resolved_references": resolved_references,
                "rewritten_query": rewritten_query,
            }
        except Exception as e:
            logger.warning(f"[QueryRewrite] LLM 调用失败，回退到规则: {e}")
            return None

    def _extract_position(self, query: str) -> Optional[str]:
        """简单提取岗位名（规则版）"""
        position_keywords = ["算法工程师", "后端开发", "前端开发", "产品经理", "测试工程师", "运营", "设计师", "数据分析师", "AI工程师"]
        for p in position_keywords:
            if p in query:
                return p
        return None
