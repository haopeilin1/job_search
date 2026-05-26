"""
知识库实体列表管理模块。

统一维护公司和岗位名称列表，支持从 jds.json 动态加载 + 硬编码兜底。
避免在 intent.py / llm_intent.py / llm_planner.py 中重复定义。
"""

import json
import logging
from pathlib import Path
from typing import Tuple, List

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════
# 硬编码兜底（常见互联网公司及其简称）
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


# ═══════════════════════════════════════════════════════
# 动态加载（从 jds.json 实时读取）
# ═══════════════════════════════════════════════════════

_JDS_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "jds.json"


def _load_kb_entities() -> Tuple[List[str], List[str]]:
    """从 jds.json 动态加载公司名和岗位名，硬编码作为兜底。"""
    companies = set(KB_COMPANIES)
    positions = set(KB_POSITIONS)
    try:
        if _JDS_FILE.exists():
            jds = json.loads(_JDS_FILE.read_text(encoding="utf-8"))
            for jd in jds:
                c = jd.get("company", "")
                if c:
                    companies.add(c)
                p = jd.get("position", "") or jd.get("title", "")
                if p:
                    positions.add(p)
                    # 常见简称映射
                    if "产品经理" in p and "产品岗" not in positions:
                        positions.add("产品岗")
                    if "AI产品经理" in p and "AI产品岗" not in positions:
                        positions.add("AI产品岗")
                    if "后端" in p and "后端" not in positions:
                        positions.add("后端")
                    if "Java" in p and "Java" not in positions:
                        positions.add("Java")
    except Exception as e:
        logger.warning(f"[_load_kb_entities] 加载 jds.json 失败，使用硬编码兜底: {e}")
    return sorted(companies), sorted(positions)


def get_kb_entities_summary() -> Tuple[List[str], List[str], int]:
    """
    返回 (公司列表, 岗位列表, 总JD数)。
    供 Planner / 意图识别模块调用。
    """
    companies, positions = _load_kb_entities()
    jd_count = 0
    try:
        if _JDS_FILE.exists():
            jds = json.loads(_JDS_FILE.read_text(encoding="utf-8"))
            jd_count = len(jds)
    except Exception:
        pass
    return companies, positions, jd_count
