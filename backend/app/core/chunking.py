"""
JD Chunk 切分策略 —— 四种实现用于对比实验

设计决策：
1. 保留多种策略，面试时展示从"固定大小"到"语义切分"的演进故事
2. 每个 chunk 必须携带完整的 metadata，便于后续向量检索时做元数据过滤
3. chunk 内容要自包含（带上上下文提示），提高 embedding 质量
"""

import logging
import re
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """
    Chunk 数据结构。

    content: 切分后的文本内容（用于 embedding）
    metadata: 附加信息（用于 ChromaDB 元数据过滤）
    """
    content: str
    metadata: dict


# ──────────────────────────── 策略 A：固定大小滑动窗口（已验证存在问题，仅作对比） ────────────────────────────

def chunk_fixed_size(
    jd: dict,
    size: int = 512,
    overlap: int = 50,
) -> List[Chunk]:
    """
    第一版策略：固定长度滑动窗口切分

    ⚠️ 此版本已验证存在切断问题，仅用于对比实验：
       - 会把"硬性要求"拦腰切断，导致"3年以上经验"被切成"3年"和"以上经验"
       - 会把句子中间切断，embedding 失去语义完整性
       - 不同 chunk 之间需要大量 overlap 才能缓解，但增加了冗余

    Args:
        jd: 结构化 JD 字典（JDSchema 格式）
        size: 每个 chunk 的最大字符数
        overlap: 相邻 chunk 的重叠字符数

    Returns:
        Chunk 列表
    """
    raw_text = jd.get("raw_text", "")
    if not raw_text:
        return []

    chunks = []
    start = 0
    index = 0
    jd_id = jd.get("jd_id", "")
    company = jd.get("company", "")
    position = jd.get("position", "")

    while start < len(raw_text):
        end = min(start + size, len(raw_text))
        content = raw_text[start:end]

        chunks.append(Chunk(
            content=content,
            metadata={
                "jd_id": jd_id,
                "company": company,
                "position": position,
                "strategy": "fixed",
                "index": index,
                "start": start,
                "end": end,
                # 标记此策略已知的问题，便于 A/B 实验分析
                "note": "fixed_size: may cut sentences in half",
            },
        ))

        start += size - overlap
        index += 1

    logger.info(f"[chunk_fixed_size] jd_id={jd_id} generated {len(chunks)} chunks (size={size}, overlap={overlap})")
    return chunks


# ──────────────────────────── 策略 B：语义切分（推荐策略） ────────────────────────────

def chunk_semantic(jd: dict) -> List[Chunk]:
    """
    第二版策略：基于 JD 结构化字段的语义切分

    设计理由：
    - JD 天然具有结构化特征（职责、要求、加分项），按字段切分保证语义完整性
    - 每个 chunk 自包含上下文（带上"这是XX公司的XX岗位的XX要求"前缀），提高 embedding 质量
    - 硬性要求和软性要求分开存储，检索时可以按意图做 metadata 过滤
    - 不会出现固定大小切分的"拦腰切断"问题

    切分规则：
    1. 基础信息 chunk：公司名 + 岗位名 + 地点 + 薪资（用于快速匹配公司和岗位）
    2. 岗位职责 chunk：完整职责原文（上下文前缀增强）
    3. 硬性要求 chunks：每条硬性要求单独成 chunk（高优先级检索）
    4. 软性要求 chunks：每条软性要求/加分项单独成 chunk
    5. 关键词 chunk：所有关键词汇总（用于技能匹配）

    Args:
        jd: 结构化 JD 字典（JDSchema 格式）

    Returns:
        Chunk 列表
    """
    chunks = []
    jd_id = jd.get("jd_id", "")
    company = jd.get("company", "未知公司")
    position = jd.get("position", "未知岗位")
    location = jd.get("location") or "地点未说明"
    salary_range = jd.get("salary_range") or "薪资面议"
    sections = jd.get("sections", {})
    keywords = jd.get("keywords", [])

    # 提取结构化摘要（用于粗筛层元数据过滤）
    structured_summary = jd.get("structured_summary", {}) or {}

    # 1. 基础信息 chunk（携带结构化摘要元数据，供粗筛使用）
    basic_meta = {
        "jd_id": jd_id,
        "company": company,
        "position": position,
        "strategy": "semantic",
        "section": "basic_info",
        "index": 0,
    }
    # 粗筛层元数据：仅当值不为 None 时才放入，避免 ChromaDB 序列化失败
    for _key in ("min_years", "max_years", "min_education", "category", "domain"):
        _val = structured_summary.get(_key)
        if _val is not None:
            basic_meta[_key] = _val

    chunks.append(Chunk(
        content=f"公司：{company}，岗位：{position}，地点：{location}，薪资：{salary_range}",
        metadata=basic_meta,
    ))

    # 2. 岗位职责 chunk
    responsibilities = sections.get("responsibilities", "")
    if responsibilities:
        chunks.append(Chunk(
            content=f"【{company} · {position} 岗位职责】\n{responsibilities}",
            metadata={
                "jd_id": jd_id,
                "company": company,
                "position": position,
                "strategy": "semantic",
                "section": "responsibilities",
                "index": 1,
            },
        ))

    # 3. 硬性要求 chunks（每条单独成 chunk，高优先级）
    hard_requirements = sections.get("hard_requirements", [])
    for i, req in enumerate(hard_requirements):
        chunks.append(Chunk(
            content=f"【{company} · {position} 硬性要求】\n{req}",
            metadata={
                "jd_id": jd_id,
                "company": company,
                "position": position,
                "strategy": "semantic",
                "section": "hard_requirements",
                "index": i + 2,
                "priority": "high",  # 硬性要求标记为高优先级
            },
        ))

    # 4. 软性要求/加分项 chunks
    soft_requirements = sections.get("soft_requirements", [])
    for i, req in enumerate(soft_requirements):
        chunks.append(Chunk(
            content=f"【{company} · {position} 软性要求/加分项】\n{req}",
            metadata={
                "jd_id": jd_id,
                "company": company,
                "position": position,
                "strategy": "semantic",
                "section": "soft_requirements",
                "index": i + 2 + len(hard_requirements),
                "priority": "medium",
            },
        ))

    # 5. 关键词 chunk
    if keywords:
        chunks.append(Chunk(
            content=f"【{company} · {position} 关键词】\n{', '.join(keywords)}",
            metadata={
                "jd_id": jd_id,
                "company": company,
                "position": position,
                "strategy": "semantic",
                "section": "keywords",
                "index": 2 + len(hard_requirements) + len(soft_requirements),
            },
        ))

    logger.info(f"[chunk_semantic] jd_id={jd_id} generated {len(chunks)} chunks")
    return chunks


# ──────────────────────────── 策略 C：递归分块（recursive） ────────────────────────────

def _recursive_split(text: str, max_length: int) -> List[str]:
    r"""
    递归切分辅助函数。

    核心原则：只有文本长度超过 max_length 时，才进行降级切分。

    切分层级（从粗到细，逐级降级）：
        1. 段落分隔：\n\n
        2. 换行分隔：\n
        3. 句末标点：。！？.!?
        4. 句中标点：，；;,
        5. 空格分隔：\s+
        6. 强制按字符截断（最后一级）

    切分后合并相邻短片段，使每个 chunk 尽量接近 max_length 但不超过。
    """
    # 分隔符按优先级排序（从语义完整的粗粒度到细粒度）
    _SEPARATORS = [r'\n\n', r'\n', r'[。！？.!?]', r'[，；;,]', r'\s+']

    def _split(t: str, depth: int = 0) -> List[str]:
        # 核心原则：未超过 max_length 的文本无需切分
        if len(t) <= max_length:
            return [t]

        if depth >= len(_SEPARATORS):
            # 最后一级：强制按字符截断
            return [t[i:i + max_length] for i in range(0, len(t), max_length)]

        sep = _SEPARATORS[depth]
        parts = re.split(sep, t)

        result = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) <= max_length:
                result.append(part)
            else:
                # 超过 max_length，继续降级切分
                result.extend(_split(part, depth + 1))
        return result

    fragments = _split(text)

    # 合并相邻短片段（严格保证合并后不超过 max_length）
    merged = []
    current = ""
    for frag in fragments:
        if not current:
            current = frag
        elif len(current) + 1 + len(frag) <= max_length:
            current += "\n" + frag
        else:
            merged.append(current)
            current = frag
    if current:
        merged.append(current)

    return merged


def chunk_recursive(jd: dict, max_length: int = 512) -> List[Chunk]:
    """
    第三版策略：递归分块

    设计理由：
    - 先按 section 切分，保证文档结构层级不被破坏
    - section 内部再按语义边界逐级降级切分（换行 -> 逗号 -> 强制截断）
    - 切分后合并相邻短片段，使 chunk 大小尽量均匀
    - 通过调整 max_length 可灵活控制粒度（如 512 vs 256）

    Args:
        jd: 结构化 JD 字典（JDSchema 格式）
        max_length: 每个 chunk 的最大字符数（默认 512，可对比 256）

    Returns:
        Chunk 列表
    """
    chunks = []
    jd_id = jd.get("jd_id", "")
    company = jd.get("company", "未知公司")
    position = jd.get("position", "未知岗位")
    location = jd.get("location") or "地点未说明"
    salary_range = jd.get("salary_range") or "薪资面议"
    sections = jd.get("sections", {})
    keywords = jd.get("keywords", [])

    # 1. 基础信息 chunk（通常较短，无需递归切分）
    basic_info = f"公司：{company}，岗位：{position}，地点：{location}，薪资：{salary_range}"
    if basic_info:
        chunks.append(Chunk(
            content=basic_info,
            metadata={
                "jd_id": jd_id,
                "company": company,
                "position": position,
                "strategy": "recursive",
                "max_length": max_length,
                "section": "basic_info",
                "index": 0,
            },
        ))

    # 2. 岗位职责（长文本，递归切分）
    responsibilities = sections.get("responsibilities", "")
    if responsibilities:
        resp_parts = _recursive_split(responsibilities, max_length)
        for i, part in enumerate(resp_parts):
            chunks.append(Chunk(
                content=f"【{company} · {position} 岗位职责】\n{part}",
                metadata={
                    "jd_id": jd_id,
                    "company": company,
                    "position": position,
                    "strategy": "recursive",
                    "max_length": max_length,
                    "section": "responsibilities",
                    "index": i + 1,
                },
            ))

    # 3. 硬性要求（数组，每条可能很长，各自递归切分）
    hard_requirements = sections.get("hard_requirements", [])
    hr_index = 1 + len(resp_parts)
    for req in hard_requirements:
        req_parts = _recursive_split(req, max_length)
        for i, part in enumerate(req_parts):
            chunks.append(Chunk(
                content=f"【{company} · {position} 硬性要求】\n{part}",
                metadata={
                    "jd_id": jd_id,
                    "company": company,
                    "position": position,
                    "strategy": "recursive",
                    "max_length": max_length,
                    "section": "hard_requirements",
                    "index": hr_index,
                    "priority": "high",
                },
            ))
            hr_index += 1

    # 4. 软性要求/加分项（数组，各自递归切分）
    soft_requirements = sections.get("soft_requirements", [])
    sr_index = hr_index
    for req in soft_requirements:
        req_parts = _recursive_split(req, max_length)
        for i, part in enumerate(req_parts):
            chunks.append(Chunk(
                content=f"【{company} · {position} 软性要求/加分项】\n{part}",
                metadata={
                    "jd_id": jd_id,
                    "company": company,
                    "position": position,
                    "strategy": "recursive",
                    "max_length": max_length,
                    "section": "soft_requirements",
                    "index": sr_index,
                    "priority": "medium",
                },
            ))
            sr_index += 1

    # 5. 关键词 chunk
    if keywords:
        kw_text = ", ".join(keywords)
        kw_parts = _recursive_split(kw_text, max_length)
        for i, part in enumerate(kw_parts):
            chunks.append(Chunk(
                content=f"【{company} · {position} 关键词】\n{part}",
                metadata={
                    "jd_id": jd_id,
                    "company": company,
                    "position": position,
                    "strategy": "recursive",
                    "max_length": max_length,
                    "section": "keywords",
                    "index": sr_index + i,
                },
            ))

    logger.info(f"[chunk_recursive] jd_id={jd_id} max_length={max_length} generated {len(chunks)} chunks")
    return chunks


# ──────────────────────────── 策略 D：基于文档结构的分块（section） ────────────────────────────

def chunk_by_section(jd: dict) -> List[Chunk]:
    """
    第四版策略：基于文档结构的分块

    设计理由：
    - 直接按 section 切分，每个 section 作为一个独立 chunk
    - 最大程度保留文档的原始结构层级
    - 不做过细的拆分，适合需要完整上下文的检索场景
    - 硬性要求和软性要求各自合并为一个整体 chunk（而非逐条拆分）

    Args:
        jd: 结构化 JD 字典（JDSchema 格式）

    Returns:
        Chunk 列表
    """
    chunks = []
    jd_id = jd.get("jd_id", "")
    company = jd.get("company", "未知公司")
    position = jd.get("position", "未知岗位")
    location = jd.get("location") or "地点未说明"
    salary_range = jd.get("salary_range") or "薪资面议"
    sections = jd.get("sections", {})
    keywords = jd.get("keywords", [])

    # 1. 基础信息 chunk
    chunks.append(Chunk(
        content=f"公司：{company}，岗位：{position}，地点：{location}，薪资：{salary_range}",
        metadata={
            "jd_id": jd_id,
            "company": company,
            "position": position,
            "strategy": "section",
            "section": "basic_info",
            "index": 0,
        },
    ))

    # 2. 岗位职责 chunk（整体作为一个 chunk）
    responsibilities = sections.get("responsibilities", "")
    if responsibilities:
        chunks.append(Chunk(
            content=f"【{company} · {position} 岗位职责】\n{responsibilities}",
            metadata={
                "jd_id": jd_id,
                "company": company,
                "position": position,
                "strategy": "section",
                "section": "responsibilities",
                "index": 1,
            },
        ))

    # 3. 硬性要求 chunk（所有要求合并为一个 chunk）
    hard_requirements = sections.get("hard_requirements", [])
    if hard_requirements:
        content = f"【{company} · {position} 硬性要求】\n" + "\n".join(f"- {r}" for r in hard_requirements)
        chunks.append(Chunk(
            content=content,
            metadata={
                "jd_id": jd_id,
                "company": company,
                "position": position,
                "strategy": "section",
                "section": "hard_requirements",
                "index": 2,
                "priority": "high",
            },
        ))

    # 4. 软性要求/加分项 chunk（所有要求合并为一个 chunk）
    soft_requirements = sections.get("soft_requirements", [])
    if soft_requirements:
        content = f"【{company} · {position} 软性要求/加分项】\n" + "\n".join(f"- {r}" for r in soft_requirements)
        chunks.append(Chunk(
            content=content,
            metadata={
                "jd_id": jd_id,
                "company": company,
                "position": position,
                "strategy": "section",
                "section": "soft_requirements",
                "index": 3,
                "priority": "medium",
            },
        ))

    # 5. 关键词 chunk
    if keywords:
        chunks.append(Chunk(
            content=f"【{company} · {position} 关键词】\n{', '.join(keywords)}",
            metadata={
                "jd_id": jd_id,
                "company": company,
                "position": position,
                "strategy": "section",
                "section": "keywords",
                "index": 4,
            },
        ))

    logger.info(f"[chunk_by_section] jd_id={jd_id} generated {len(chunks)} chunks")
    return chunks
