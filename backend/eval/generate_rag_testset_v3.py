"""
生成RAG评测集（Golden Dataset）v3

精确标注每条测试用例的期望检索结果，支持：
- 精确率/召回率（按chunk级别）
- Hit Rate / MRR / NDCG（按JD级别，可扩展按chunk级别）
"""

import json
from typing import List, Dict, Any


def load_jds(path: str = "data/jds.json") -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_test_cases(path: str = "eval/test_dataset.jsonl") -> List[Dict]:
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[WARN] Skip malformed line {i}")
    return cases


def build_jd_maps(jds: List[Dict]) -> Dict:
    by_id = {}
    by_company = {}
    by_position = {}
    by_company_position = {}
    for jd in jds:
        jd_id = jd["id"]
        company = jd["company"]
        position = jd["position"]
        by_id[jd_id] = jd
        by_company.setdefault(company, []).append(jd)
        by_position.setdefault(position, []).append(jd)
        by_company_position[f"{company}_{position}"] = jd
    return {"by_id": by_id, "by_company": by_company, "by_position": by_position, "by_company_position": by_company_position}


def infer_rewritten_query(case: Dict) -> tuple:
    msg = case["message"]
    ctx = case.get("eval_context", {})
    slots = ctx.get("gold_slots", {})
    follow_up = ctx.get("follow_up_type", "none")
    session_group = case.get("session_group")

    rewritten = msg
    replacements = {"咋样": "怎么样", "啥": "什么", "咋": "怎么", "投": "投递", "面": "面试", "够格": "匹配", "开多少": "薪资是多少"}
    for old, new in replacements.items():
        rewritten = rewritten.replace(old, new)

    if follow_up == "expand" and session_group:
        if "上面那个" in msg or "那个岗" in msg or "这个岗" in msg:
            company = slots.get("company", "")
            position = slots.get("position", "")
            if company and position:
                rewritten = rewritten.replace("上面那个岗", f"{company}{position}").replace("那个岗", f"{company}{position}").replace("这个岗", f"{company}{position}")
    
    if follow_up == "clarify":
        if len(rewritten) < 15 and (slots.get("company") or slots.get("position")):
            company = slots.get("company", "")
            position = slots.get("position", "")
            rewritten = f"{company}{position}"

    stop_words = ["请问", "我想知道", "能告诉我", "怎么样", "吗", "呢", "吧", "啊", "？", "?", "帮我", "看看", "有什么", "的", "是", "什么"]
    search_keywords = rewritten
    for sw in stop_words:
        search_keywords = search_keywords.replace(sw, " ")
    search_keywords = " ".join(search_keywords.split())

    return rewritten.strip(), search_keywords.strip()


# ========== 手动标注映射表 ==========
# 基于对测试集和JD库的深度理解，为每个case定义期望检索结果

CASE_ANNOTATIONS = {
    # --- 陈雨桐（AI产品简历）---
    "eval_chen_01": {
        "rewritten_query": "帮我看看有什么适合我的AI产品实习岗",
        "search_keywords": "适合我的AI产品实习岗",
        "golden_jd_ids": [5, 7, 9, 10, 11, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        "golden_chunk_ids": "auto",
        "critical_sections": ["basic_info", "hard_requirements", "soft_requirements"],
        "retrieval_type": "explore",
        "notes": "在校生找AI产品实习/校招，期望粗筛返回实习/校招JD。JD 1-4为正式社招岗，不应优先返回。",
        "relevance_scores": {5: 3, 7: 3, 9: 3, 10: 3, 11: 3, 14: 3, 15: 3, 17: 3, 18: 3, 19: 3, 20: 3, 21: 3, 22: 3, 23: 3, 24: 3, 25: 3, 26: 3, 27: 3, 28: 3, 29: 3, 30: 3, 1: 0, 2: 0, 3: 0, 4: 0, 6: 0, 8: 0, 12: 0, 13: 0, 16: 0},
    },
    "eval_chen_02": {
        "rewritten_query": "字节跳动的AI产品经理我匹配吗",
        "search_keywords": "字节跳动 AI产品经理 匹配",
        "golden_jd_ids": [5],
        "golden_chunk_ids": "auto",
        "critical_sections": ["basic_info", "responsibilities", "hard_requirements", "soft_requirements", "keywords"],
        "retrieval_type": "single_jd",
        "notes": "明确查询字节跳动AI产品经理（JD ID=5），期望精确命中该JD全部chunks",
        "relevance_scores": {5: 3},
    },
    "eval_chen_03": {
        "rewritten_query": "百度的AI产品实习生要求什么学历",
        "search_keywords": "百度 AI产品实习生 学历要求",
        "golden_jd_ids": [10, 11],
        "golden_chunk_ids": "auto",
        "critical_sections": ["hard_requirements"],
        "retrieval_type": "verify",
        "notes": "百度AI产品实习生岗位（JD 10=AI产品实习生，JD 11=AI能力产品实习生）。JD 2=大模型应用PM是正式岗非实习，JD 12/28是AI应用产品经理。学历信息主要在hard_requirements chunk中。",
        "relevance_scores": {10: 3, 11: 3, 2: 1, 12: 1, 28: 1},
    },
    "eval_chen_04": {
        "rewritten_query": "面试字节AI产品经理要准备什么",
        "search_keywords": "面试 字节跳动 AI产品经理 准备",
        "golden_jd_ids": [5],
        "golden_chunk_ids": "auto",
        "critical_sections": ["basic_info", "responsibilities", "hard_requirements", "soft_requirements", "keywords"],
        "retrieval_type": "single_jd",
        "notes": "面试准备需基于JD全部信息生成针对性问题",
        "relevance_scores": {5: 3},
    },
    "eval_chen_05": {
        "rewritten_query": "阿里巴巴的AI Agent产品经理匹配吗",
        "search_keywords": "阿里巴巴 AI Agent产品经理 匹配",
        "golden_jd_ids": [23],
        "golden_chunk_ids": "auto",
        "critical_sections": ["basic_info", "responsibilities", "hard_requirements", "soft_requirements", "keywords"],
        "retrieval_type": "single_jd",
        "notes": "明确查询阿里巴巴AI Agent产品经理（JD ID=23）",
        "relevance_scores": {23: 3},
    },
    "eval_chen_06": {
        "rewritten_query": "阿里巴巴AI Agent产品经理薪资是多少",
        "search_keywords": "阿里巴巴 AI Agent产品经理 薪资",
        "golden_jd_ids": [23],
        "golden_chunk_ids": "auto",
        "critical_sections": ["basic_info"],
        "retrieval_type": "verify_expand",
        "notes": "多轮展开（session_group=chen_m1），继承上一轮槽位：company=阿里巴巴, position=AI Agent产品经理。应复用kb_retrieve证据或重新检索JD 23的basic_info chunk。",
        "relevance_scores": {23: 3},
    },
    "eval_chen_07": {
        "rewritten_query": "帮我筛选几个能投递的岗位，再重点看看字节跳动AI产品经理",
        "search_keywords": "筛选 投递 岗位 字节跳动 AI产品经理",
        "golden_jd_ids": [5],
        "golden_chunk_ids": "auto",
        "critical_sections": ["basic_info", "responsibilities", "hard_requirements", "soft_requirements", "keywords"],
        "retrieval_type": "explore+single_jd",
        "notes": "多意图：explore返回多个候选+assess针对字节AI产品经理（JD 5）深度分析。explore部分期望命中AI产品相关JD，assess部分必须命中JD 5全部chunks。",
        "relevance_scores": {5: 3, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2, 16: 2, 17: 2, 18: 2, 19: 2, 20: 2, 21: 2, 22: 2, 23: 2, 24: 2, 25: 2, 26: 2, 27: 2, 28: 2, 29: 2, 30: 2, 1: 0, 2: 0, 3: 0, 4: 0},
    },
    "eval_chen_08": {
        "rewritten_query": "小米AI培训方向产品实习生我匹配吗",
        "search_keywords": "小米 AI培训方向产品实习生 匹配",
        "golden_jd_ids": [17],
        "golden_chunk_ids": "auto",
        "critical_sections": ["basic_info", "responsibilities", "hard_requirements", "soft_requirements", "keywords"],
        "retrieval_type": "single_jd",
        "notes": "明确查询小米AI培训方向产品实习生（JD ID=17）",
        "relevance_scores": {17: 3},
    },
    "eval_chen_09": {
        "rewritten_query": "我的RAG和Embedding经验对哪个岗位最有用",
        "search_keywords": "RAG Embedding 经验 岗位 最有用",
        "golden_jd_ids": [2, 4, 5, 6, 15, 16, 23, 24, 25, 26, 29],
        "golden_chunk_ids": "auto",
        "critical_sections": ["keywords", "hard_requirements"],
        "retrieval_type": "skill_explore",
        "notes": "技能导向探索，RAG/Embedding相关JD应排名靠前。JD 2（百度大模型应用PM，关键词含RAG）、JD 4（美团搜索推荐AI PM，关键词含RAG）、JD 5（字节AI产品经理，关键词含RAG）、JD 6（蚂蚁集团AI产品经理，关键词含Agent/Planning）、JD 15/16（蚂蚁集团AI产品工程师）、JD 23/24/25（阿里巴巴/淘天集团AI Agent/大模型产品经理）、JD 26（联想产品经理-AI方向实习，关键词含RAG）、JD 29（vivo AI产品经理-实习，关键词含RAG）。",
        "relevance_scores": {2: 3, 4: 3, 5: 3, 26: 3, 29: 3, 6: 2, 15: 2, 16: 2, 23: 2, 24: 2, 25: 2},
    },
    "eval_chen_10": {
        "rewritten_query": "蚂蚁集团AI产品经理需要什么技能",
        "search_keywords": "蚂蚁集团 AI产品经理 技能",
        "golden_jd_ids": [6],
        "golden_chunk_ids": "auto",
        "critical_sections": ["hard_requirements", "soft_requirements", "keywords"],
        "retrieval_type": "verify",
        "notes": "蚂蚁集团AI产品经理（JD ID=6），测试对hard_requirements的精准提取",
        "relevance_scores": {6: 3},
    },
    "eval_chen_11": {
        "rewritten_query": "先帮我挑选几个能投递的岗位，再告诉我要准备什么面试题",
        "search_keywords": "挑选 投递 岗位 面试题 准备",
        "golden_jd_ids": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        "golden_chunk_ids": "auto",
        "critical_sections": ["basic_info", "keywords"],
        "retrieval_type": "explore",
        "notes": "多意图explore+prepare，prepare依赖explore的top结果。explore期望返回AI产品相关JD。",
        "relevance_scores": {5: 3, 6: 3, 7: 3, 8: 3, 9: 3, 10: 3, 11: 3, 12: 3, 13: 3, 14: 3, 15: 3, 16: 3, 17: 3, 18: 3, 19: 3, 20: 3, 21: 3, 22: 3, 23: 3, 24: 3, 25: 3, 26: 3, 27: 3, 28: 3, 29: 3, 30: 3, 1: 0, 2: 0, 3: 0, 4: 0},
    },
    "eval_chen_12": {
        "rewritten_query": "快手搜索推荐AI PM我能去吗",
        "search_keywords": "快手 搜索推荐AI PM 匹配",
        "golden_jd_ids": [],
        "golden_chunk_ids": [],
        "critical_sections": [],
        "retrieval_type": "single_jd_not_found",
        "notes": "知识库中无快手搜索推荐AI PM（美团ID=4有搜索推荐AI PM），测试系统对不存在的JD的处理。期望检索为空或返回近似JD（如美团ID=4）。",
        "relevance_scores": {},
    },
    "eval_chen_14": {
        "rewritten_query": "蚂蚁集团AI产品经理",
        "search_keywords": "蚂蚁集团 AI产品经理",
        "golden_jd_ids": [6],
        "golden_chunk_ids": "auto",
        "critical_sections": ["basic_info", "responsibilities", "hard_requirements", "soft_requirements", "keywords"],
        "retrieval_type": "single_jd",
        "notes": "澄清后用户补充实体，期望命中蚂蚁集团AI产品经理（JD 6）",
        "relevance_scores": {6: 3},
    },
    "eval_chen_15": {
        "rewritten_query": "没有工作经验能投递产品岗吗",
        "search_keywords": "没有工作经验 产品岗 投递",
        "golden_jd_ids": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        "golden_chunk_ids": "auto",
        "critical_sections": ["hard_requirements", "soft_requirements"],
        "retrieval_type": "verify",
        "notes": "检索知识库中产品岗经验要求。所有产品相关JD的hard_requirements中关于经验的要求都是关键chunk。",
        "relevance_scores": {5: 3, 6: 3, 7: 3, 8: 3, 9: 3, 10: 3, 11: 3, 12: 3, 13: 3, 14: 3, 15: 3, 16: 3, 17: 3, 18: 3, 19: 3, 20: 3, 21: 3, 22: 3, 23: 3, 24: 3, 25: 3, 26: 3, 27: 3, 28: 3, 29: 3, 30: 3, 1: 0, 2: 0, 3: 0, 4: 0},
    },
    "eval_chen_16": {
        "rewritten_query": "把所有AI产品岗位都给我分析一遍",
        "search_keywords": "所有 AI产品岗位 分析",
        "golden_jd_ids": [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        "golden_chunk_ids": "auto",
        "critical_sections": ["basic_info", "keywords"],
        "retrieval_type": "explore",
        "notes": "大范围探索，期望粗筛层返回所有AI产品相关JD。非AI产品岗（JD 1,3）不应返回。",
        "relevance_scores": {2: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3, 9: 3, 10: 3, 11: 3, 12: 3, 13: 3, 14: 3, 15: 3, 16: 3, 17: 3, 18: 3, 19: 3, 20: 3, 21: 3, 22: 3, 23: 3, 24: 3, 25: 3, 26: 3, 27: 3, 28: 3, 29: 3, 30: 3, 1: 0, 3: 0},
    },
    "eval_chen_17": {
        "rewritten_query": "我比较看重Prompt Engineering的应用场景，有合适的岗位吗",
        "search_keywords": "Prompt Engineering 应用场景 合适岗位",
        "golden_jd_ids": [2, 4, 5, 6, 15, 16, 19, 21, 23, 24, 25, 26, 29],
        "golden_chunk_ids": "auto",
        "critical_sections": ["keywords", "hard_requirements"],
        "retrieval_type": "skill_explore",
        "notes": "偏好导向探索，Prompt Engineering相关JD应排名靠前。JD 2/4/5/6/15/16/19/21/23/24/25/26/29的关键词或要求中包含Prompt Engineering。",
        "relevance_scores": {2: 3, 4: 3, 5: 3, 19: 3, 21: 3, 23: 3, 24: 3, 25: 3, 26: 3, 29: 3, 6: 2, 15: 2, 16: 2},
    },
    "eval_chen_18": {
        "rewritten_query": "百度大模型应用PM和美团搜索推荐AI PM哪个更适合我",
        "search_keywords": "百度 大模型应用PM 美团 搜索推荐AI PM 适合",
        "golden_jd_ids": [2, 4],
        "golden_chunk_ids": "auto",
        "critical_sections": ["basic_info", "responsibilities", "hard_requirements", "soft_requirements", "keywords"],
        "retrieval_type": "multi_jd_compare",
        "notes": "两个JD分别匹配后对比。必须同时命中百度大模型应用PM（JD 2）和美团搜索推荐AI PM（JD 4）的全部chunks。",
        "relevance_scores": {2: 3, 4: 3},
    },
    
    # --- 李工程师（Java后端简历）---
    "eval_li_01": {
        "rewritten_query": "帮我看看有什么Java后端岗",
        "search_keywords": "Java后端 岗位",
        "golden_jd_ids": [1, 3],
        "golden_chunk_ids": "auto",
        "critical_sections": ["basic_info", "keywords"],
        "retrieval_type": "explore",
        "notes": "期望命中阿里巴巴-后端开发(ID=1)、某小公司-Java后端(ID=3)。JD 2/4+为AI产品岗，不应优先返回。",
        "relevance_scores": {1: 3, 3: 3, 2: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0, 29: 0, 30: 0},
    },
    "eval_li_02": {
        "rewritten_query": "阿里巴巴后端开发我匹配吗",
        "search_keywords": "阿里巴巴 后端开发 匹配",
        "golden_jd_ids": [1],
        "golden_chunk_ids": "auto",
        "critical_sections": ["basic_info", "responsibilities", "hard_requirements", "soft_requirements", "keywords"],
        "retrieval_type": "single_jd",
        "notes": "明确查询阿里巴巴后端开发（JD ID=1），期望精确命中全部chunks",
        "relevance_scores": {1: 3},
    },
    "eval_li_03": {
        "rewritten_query": "阿里巴巴后端岗薪资是多少",
        "search_keywords": "阿里巴巴 后端岗 薪资",
        "golden_jd_ids": [1],
        "golden_chunk_ids": "auto",
        "critical_sections": ["basic_info"],
        "retrieval_type": "verify",
        "notes": "阿里后端JD薪资范围35k-65k，信息在basic_info chunk中",
        "relevance_scores": {1: 3},
    },
    "eval_li_05": {
        "rewritten_query": "我有3年Spring Boot和高并发经验，能投递哪些岗位",
        "search_keywords": "3年 Spring Boot 高并发经验 投递 岗位",
        "golden_jd_ids": [1, 3],
        "golden_chunk_ids": "auto",
        "critical_sections": ["basic_info", "keywords"],
        "retrieval_type": "skill_explore",
        "notes": "期望Spring Boot/高并发技能提升相关JD排序。JD 1（阿里后端，关键词含Spring Boot）和JD 3（某小公司Java后端，关键词含Spring Boot）最相关。",
        "relevance_scores": {1: 3, 3: 3},
    },
    "eval_li_06": {
        "rewritten_query": "阿里巴巴后端要求几年经验",
        "search_keywords": "阿里巴巴 后端 经验要求",
        "golden_jd_ids": [1],
        "golden_chunk_ids": "auto",
        "critical_sections": ["hard_requirements"],
        "retrieval_type": "verify",
        "notes": "阿里巴巴ID=1要求3年以上Java，信息在hard_requirements chunk中",
        "relevance_scores": {1: 3},
    },
    "eval_li_07": {
        "rewritten_query": "帮我筛选几个岗位，重点看看阿里巴巴那个后端开发",
        "search_keywords": "筛选 岗位 阿里巴巴 后端开发",
        "golden_jd_ids": [1],
        "golden_chunk_ids": "auto",
        "critical_sections": ["basic_info", "responsibilities", "hard_requirements", "soft_requirements", "keywords"],
        "retrieval_type": "explore+single_jd",
        "notes": "explore做全局返回Java后端相关JD，assess针对阿里后端开发（JD 1）深度分析",
        "relevance_scores": {1: 3, 3: 2},
    },
    "eval_li_08": {
        "rewritten_query": "字节AI产品经理我能转行去吗",
        "search_keywords": "字节跳动 AI产品经理 转行 匹配",
        "golden_jd_ids": [5],
        "golden_chunk_ids": "auto",
        "critical_sections": ["basic_info", "responsibilities", "hard_requirements", "soft_requirements", "keywords"],
        "retrieval_type": "single_jd",
        "notes": "Java后端转AI产品，期望匹配分析指出技能差距。必须命中JD 5全部chunks",
        "relevance_scores": {5: 3},
    },
    "eval_li_09": {
        "rewritten_query": "我的MySQL分库分表经验对哪个岗位最有价值",
        "search_keywords": "MySQL 分库分表 经验 岗位 价值",
        "golden_jd_ids": [1],
        "golden_chunk_ids": "auto",
        "critical_sections": ["hard_requirements", "keywords"],
        "retrieval_type": "skill_explore",
        "notes": "JD中对SQL有要求的多，但提到分库分表的只有JD 1（阿里后端，hard_requirements中未明确说分库分表但提到了MySQL）。此条可能触发外部搜索或global_rank。",
        "relevance_scores": {1: 3, 3: 2},
    },
    "eval_li_11": {
        "rewritten_query": "阿里巴巴后端岗具体要求是什么",
        "search_keywords": "阿里巴巴 后端岗 具体要求",
        "golden_jd_ids": [1],
        "golden_chunk_ids": "auto",
        "critical_sections": ["hard_requirements", "soft_requirements", "responsibilities"],
        "retrieval_type": "verify_expand",
        "notes": "多轮展开（session_group=li_m1），引用上一轮阿里后端岗（JD 1），应复用kb_retrieve证据或重新检索JD 1的全部要求相关chunks。",
        "relevance_scores": {1: 3},
    },
    "eval_li_12": {
        "rewritten_query": "帮我推荐几个技术岗，顺便说下薪资和要求",
        "search_keywords": "推荐 技术岗 薪资 要求",
        "golden_jd_ids": [1, 3],
        "golden_chunk_ids": "auto",
        "critical_sections": ["basic_info", "hard_requirements"],
        "retrieval_type": "explore+verify",
        "notes": "技术岗探索+属性查询。explore期望返回Java后端相关JD（1,3），verify针对top岗位询问薪资和要求。",
        "relevance_scores": {1: 3, 3: 3},
    },
    "eval_li_15": {
        "rewritten_query": "阿里巴巴后端开发",
        "search_keywords": "阿里巴巴 后端开发",
        "golden_jd_ids": [1],
        "golden_chunk_ids": "auto",
        "critical_sections": ["basic_info", "responsibilities", "hard_requirements", "soft_requirements", "keywords"],
        "retrieval_type": "single_jd",
        "notes": "澄清后用户补充实体，期望命中阿里巴巴后端开发（JD 1）",
        "relevance_scores": {1: 3},
    },
    "eval_li_16": {
        "rewritten_query": "百度有Java后端岗吗",
        "search_keywords": "百度 Java后端岗",
        "golden_jd_ids": [],
        "golden_chunk_ids": [],
        "critical_sections": [],
        "retrieval_type": "single_jd_not_found",
        "notes": "百度4条JD全为AI产品，无Java岗。期望kb_retrieve返回空后正确告知用户。",
        "relevance_scores": {},
    },
    
    # --- 王设计师（UI设计简历）---
    "eval_wang_01": {
        "rewritten_query": "帮我看看有什么设计岗",
        "search_keywords": "设计岗",
        "golden_jd_ids": [],
        "golden_chunk_ids": [],
        "critical_sections": [],
        "retrieval_type": "explore",
        "notes": "30条JD中无纯UI/设计岗，测试系统搜索后如何回答空结果。期望检索为空或返回零个chunk。",
        "relevance_scores": {},
    },
    "eval_wang_02": {
        "rewritten_query": "有UI设计相关的岗位吗",
        "search_keywords": "UI设计 岗位",
        "golden_jd_ids": [],
        "golden_chunk_ids": [],
        "critical_sections": [],
        "retrieval_type": "explore",
        "notes": "30条JD中无纯UI设计JD，测试空结果处理。期望检索为空。",
        "relevance_scores": {},
    },
    "eval_wang_03": {
        "rewritten_query": "我的Figma组件化经验能投递哪个岗位",
        "search_keywords": "Figma 组件化 经验 投递 岗位",
        "golden_jd_ids": [7, 18],
        "golden_chunk_ids": "auto",
        "critical_sections": ["basic_info", "hard_requirements", "keywords"],
        "retrieval_type": "skill_explore",
        "notes": "JD 7（ACG干帆智能体平台产品实习生）hard_requirements明确写明熟练使用Axure、Figma等产品工具，与Figma查询高度相关。JD 18（小米AI产品实习生）关键词含Figma，也最相关。两者均应纳入golden set。",
        "relevance_scores": {7: 3, 18: 3, 8: 1, 17: 1, 19: 1, 26: 1, 27: 1, 29: 1},
    },
    "eval_wang_04": {
        "rewritten_query": "产品岗对设计能力有要求吗",
        "search_keywords": "产品岗 设计能力 要求",
        "golden_jd_ids": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        "golden_chunk_ids": "auto",
        "critical_sections": ["soft_requirements", "hard_requirements"],
        "retrieval_type": "verify",
        "notes": "不指定公司，测试对所有产品JD的soft_requirements聚合。需要检索所有产品相关JD中是否提到设计能力/Figma/UI等。",
        "relevance_scores": {5: 3, 6: 3, 7: 3, 8: 3, 9: 3, 10: 3, 11: 3, 12: 3, 13: 3, 14: 3, 15: 3, 16: 3, 17: 3, 18: 3, 19: 3, 20: 3, 21: 3, 22: 3, 23: 3, 24: 3, 25: 3, 26: 3, 27: 3, 28: 3, 29: 3, 30: 3, 1: 0, 2: 0, 3: 0, 4: 0},
    },
    "eval_wang_05": {
        "rewritten_query": "我从品牌设计转UI，能投递AI产品岗吗",
        "search_keywords": "品牌设计 转 UI 投递 AI产品岗",
        "golden_jd_ids": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        "golden_chunk_ids": "auto",
        "critical_sections": ["basic_info", "hard_requirements", "soft_requirements"],
        "retrieval_type": "explore",
        "notes": "设计背景投产品岗，期望分析指出设计经验是加分项但产品思维有差距。需检索AI产品相关JD的全部chunks。",
        "relevance_scores": {5: 3, 6: 3, 7: 3, 8: 3, 9: 3, 10: 3, 11: 3, 12: 3, 13: 3, 14: 3, 15: 3, 16: 3, 17: 3, 18: 3, 19: 3, 20: 3, 21: 3, 22: 3, 23: 3, 24: 3, 25: 3, 26: 3, 27: 3, 28: 3, 29: 3, 30: 3, 1: 0, 2: 0, 3: 0, 4: 0},
    },
    "eval_wang_06": {
        "rewritten_query": "帮我看看有没有需要互联网设计经验的岗位",
        "search_keywords": "互联网设计经验 岗位",
        "golden_jd_ids": [7, 18],
        "golden_chunk_ids": "auto",
        "critical_sections": ["basic_info", "hard_requirements", "soft_requirements", "keywords"],
        "retrieval_type": "explore",
        "notes": "30条JD中无纯设计岗，但JD 7（ACG）hard_requirements明确要求Axure/Figma，JD 18（小米）也要求Figma/Sketch，均为互联网设计经验的高匹配岗，不应预设空结果。",
        "relevance_scores": {7: 3, 18: 3, 8: 1, 17: 1, 19: 1, 26: 1, 27: 1, 29: 1},
    },
    "eval_wang_07": {
        "rewritten_query": "那个岗位需要Figma吗",
        "search_keywords": "那个岗位 Figma 要求",
        "golden_jd_ids": [],
        "golden_chunk_ids": [],
        "critical_sections": [],
        "retrieval_type": "verify_expand",
        "notes": "多轮展开（session_group=wang_m1），引用上一轮岗位。但上一轮eval_wang_06没有明确的company/position槽位，也没有指定具体JD。因此'上面那个岗'无法解析为确定岗位。query_rewrite应只做简单指代消解（去掉'上面'），不应假设公司。此case依赖evidence_cache复用，单独测试RAG时无法预先确定golden chunks。",
        "relevance_scores": {},
    },
    "eval_wang_09": {
        "rewritten_query": "我比较看重组件化和设计规范，有合适的岗位吗",
        "search_keywords": "组件化 设计规范 合适岗位",
        "golden_jd_ids": [7, 18],
        "golden_chunk_ids": "auto",
        "critical_sections": ["hard_requirements", "keywords"],
        "retrieval_type": "skill_explore",
        "notes": "30条JD中无设计规范/组件化专用岗。JD 7（ACG）和JD 18（小米）均提到Figma，但文本中无'组件化'或'设计规范'字样，仅靠Figma间接关联，relevance=3稍牵强；30条JD中确实无更优选项，两者均应纳入但relevance降档为2。",
        "relevance_scores": {7: 2, 18: 2, 8: 1, 17: 1, 19: 1, 26: 1, 27: 1, 29: 1},
    },
    "eval_wang_10": {
        "rewritten_query": "美团搜索推荐AI PM我匹配吗",
        "search_keywords": "美团 搜索推荐AI PM 匹配",
        "golden_jd_ids": [4],
        "golden_chunk_ids": "auto",
        "critical_sections": ["basic_info", "responsibilities", "hard_requirements", "soft_requirements", "keywords"],
        "retrieval_type": "single_jd",
        "notes": "UI设计师投搜索推荐AI PM，期望分析跨领域可行性。必须命中美团搜索推荐AI PM（JD 4）全部chunks",
        "relevance_scores": {4: 3},
    },
    "eval_wang_11": {
        "rewritten_query": "帮我筛选几个岗位，再看看哪个需要设计经验",
        "search_keywords": "筛选 岗位 设计经验",
        "golden_jd_ids": [7, 18],
        "golden_chunk_ids": "auto",
        "critical_sections": ["basic_info", "hard_requirements", "soft_requirements", "keywords"],
        "retrieval_type": "explore+verify",
        "notes": "30条JD中无纯设计岗，verify针对explore推荐的top岗位询问设计要求。JD 7（ACG）hard_requirements明确提Figma/Axure，JD 18（小米）提Figma/Sketch，均应纳入golden set。",
        "relevance_scores": {7: 3, 18: 3, 8: 1, 17: 1, 19: 1, 26: 1, 27: 1, 29: 1},
    },
    "eval_wang_12": {
        "rewritten_query": "我的4年互联网设计经验对产品岗有帮助吗",
        "search_keywords": "4年 互联网设计经验 产品岗 帮助",
        "golden_jd_ids": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        "golden_chunk_ids": "auto",
        "critical_sections": ["soft_requirements", "hard_requirements"],
        "retrieval_type": "verify",
        "notes": "用户经验对产品岗是否有帮助，需检索知识库中产品岗JD要求，确认是否包含设计经验。所有产品相关JD的soft_requirements和hard_requirements都是关键chunks。",
        "relevance_scores": {5: 3, 6: 3, 7: 3, 8: 3, 9: 3, 10: 3, 11: 3, 12: 3, 13: 3, 14: 3, 15: 3, 16: 3, 17: 3, 18: 3, 19: 3, 20: 3, 21: 3, 22: 3, 23: 3, 24: 3, 25: 3, 26: 3, 27: 3, 28: 3, 29: 3, 30: 3, 1: 0, 2: 0, 3: 0, 4: 0},
    },
    
    # --- 通用/补充 ---
    "eval_gen_03": {
        "rewritten_query": "帮我看看有什么岗位，然后第一个推荐的具体要求是什么，再帮我准备几道面试题",
        "search_keywords": "岗位 推荐 具体要求 面试题 准备",
        "golden_jd_ids": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        "golden_chunk_ids": "auto",
        "critical_sections": ["basic_info", "keywords"],
        "retrieval_type": "explore",
        "notes": "复杂多意图explore+verify+prepare。explore期望返回AI产品相关JD（用户陈雨桐是AI产品背景）。",
        "relevance_scores": {5: 3, 6: 3, 7: 3, 8: 3, 9: 3, 10: 3, 11: 3, 12: 3, 13: 3, 14: 3, 15: 3, 16: 3, 17: 3, 18: 3, 19: 3, 20: 3, 21: 3, 22: 3, 23: 3, 24: 3, 25: 3, 26: 3, 27: 3, 28: 3, 29: 3, 30: 3, 1: 0, 2: 0, 3: 0, 4: 0},
    },
    "eval_sup_03": {
        "rewritten_query": "帮我看看所有互联网大厂的AI产品岗",
        "search_keywords": "互联网大厂 AI产品岗",
        "golden_jd_ids": [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        "golden_chunk_ids": "auto",
        "critical_sections": ["basic_info", "keywords"],
        "retrieval_type": "explore",
        "notes": "大厂AI产品岗探索，期望返回所有AI产品相关JD。'大厂'包括阿里、百度、美团、字节、蚂蚁、小米、vivo、联想、滴滴、淘天等。",
        "relevance_scores": {2: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3, 9: 3, 10: 3, 11: 3, 12: 3, 13: 3, 14: 3, 15: 3, 16: 3, 17: 3, 18: 3, 19: 3, 20: 3, 21: 3, 22: 3, 23: 3, 24: 3, 25: 3, 26: 3, 27: 3, 28: 3, 29: 3, 30: 3, 1: 0, 3: 0},
    },
    "eval_sup_04": {
        "rewritten_query": "帮我看看有什么Java后端岗",
        "search_keywords": "Java后端岗",
        "golden_jd_ids": [1, 3],
        "golden_chunk_ids": "auto",
        "critical_sections": ["basic_info", "keywords"],
        "retrieval_type": "explore",
        "notes": "期望命中阿里巴巴-后端开发(ID=1)、某小公司-Java后端(ID=3)。",
        "relevance_scores": {1: 3, 3: 3, 2: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0, 29: 0, 30: 0},
    },
}


def resolve_chunk_ids(ann: Dict, jd_maps: Dict) -> List[str]:
    """根据golden_jd_ids和critical_sections解析具体的chunk_ids"""
    if ann.get("golden_chunk_ids") != "auto":
        return ann.get("golden_chunk_ids", [])
    
    result = []
    for jd_id in ann.get("golden_jd_ids", []):
        jd = jd_maps["by_id"].get(jd_id)
        if not jd:
            continue
        all_chunks = jd.get("meta", {}).get("chunk_ids", [])
        sections = ann.get("critical_sections", [])
        
        if not sections or not all_chunks:
            result.extend(all_chunks)
            continue
        
        # 按section映射chunk index
        sec = jd.get("sections", {})
        hard_count = len(sec.get("hard_requirements", []))
        soft_count = len(sec.get("soft_requirements", []))
        
        for s in sections:
            if s == "basic_info" and len(all_chunks) > 0:
                result.append(all_chunks[0])
            elif s == "responsibilities" and len(all_chunks) > 1:
                result.append(all_chunks[1])
            elif s == "hard_requirements":
                start = 2
                end = min(start + hard_count, len(all_chunks))
                result.extend(all_chunks[start:end])
            elif s == "soft_requirements":
                start = 2 + hard_count
                end = min(start + soft_count, len(all_chunks))
                result.extend(all_chunks[start:end])
            elif s == "keywords" and len(all_chunks) > 0:
                result.append(all_chunks[-1])
    
    return list(dict.fromkeys(result))


def main():
    jds = load_jds()
    cases = load_test_cases()
    jd_maps = build_jd_maps(jds)

    rag_cases = []
    for case in cases:
        tools = case.get("eval_context", {}).get("expected_tools", [])
        if not any(t in tools for t in ["kb_retrieve", "qa_synthesize", "global_rank"]):
            continue
        
        case_id = case["session_id"]
        ann = CASE_ANNOTATIONS.get(case_id, {})
        
        if not ann:
            print(f"[WARN] No annotation for {case_id}, skipping")
            continue
        
        golden_chunks = resolve_chunk_ids(ann, jd_maps)
        
        rag_case = {
            "case_id": case_id,
            "original_query": case["message"],
            "rewritten_query": ann.get("rewritten_query", case["message"]),
            "search_keywords": ann.get("search_keywords", ""),
            "retrieval_type": ann.get("retrieval_type", ""),
            "golden_jd_ids": ann.get("golden_jd_ids", []),
            "golden_chunk_ids": golden_chunks,
            "critical_sections": ann.get("critical_sections", []),
            "relevance_scores": ann.get("relevance_scores", {}),
            "annotation_notes": ann.get("notes", ""),
        }
        rag_cases.append(rag_case)

    output_path = "eval/rag_test_dataset.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for case in rag_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"Generated {len(rag_cases)} RAG test cases -> {output_path}")

    # 统计
    from collections import Counter
    type_counts = Counter(c["retrieval_type"] for c in rag_cases)
    print("\n=== Retrieval Type Distribution ===")
    for t, cnt in type_counts.most_common():
        print(f"  {t}: {cnt}")
    
    print(f"\nTotal golden chunks annotated: {sum(len(c['golden_chunk_ids']) for c in rag_cases)}")
    print(f"Cases with empty golden set: {sum(1 for c in rag_cases if not c['golden_jd_ids'])}")


if __name__ == "__main__":
    main()
