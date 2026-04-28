from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List
from datetime import datetime
import uuid
import logging
import base64
import json
import os

from app.schemas import JDSchema, JDCreateSchema, JDUpdateSchema
from app.core.llm_client import LLMClient
from app.core.jd_parser import JDParser
from app.core.vector_store import VectorStore
from app.core.embedding import EmbeddingClient
from app.core.chunking import chunk_fixed_size, chunk_semantic, chunk_recursive, chunk_by_section

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/knowledge-base", tags=["knowledge-base"])

# 全局向量库实例（生命周期随应用）
vector_store = VectorStore(embedding_client=None)

# ──────────────────────────── 持久化存储 ────────────────────────────

# JSON 数据文件路径
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
DATA_DIR = os.path.abspath(DATA_DIR)
os.makedirs(DATA_DIR, exist_ok=True)
JD_DATA_FILE = os.path.join(DATA_DIR, "jds.json")


def _serialize_datetime(obj):
    """JSON 序列化时处理 datetime 对象"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _deserialize_datetime(obj):
    """JSON 反序列化时把 ISO 格式字符串转回 datetime"""
    if isinstance(obj, str):
        # 尝试解析 datetime 格式
        try:
            return datetime.fromisoformat(obj)
        except ValueError:
            pass
    return obj


def _load_jds() -> list:
    """从 JSON 文件加载 JD 列表"""
    if not os.path.exists(JD_DATA_FILE):
        # 首次启动：写入默认 demo 数据
        default_jds = [
            {"id": 1, "jd_id": str(uuid.uuid4()), "company": "ByteDance", "title": "AI 平台产品经理", "description": "负责大模型基础设施产品设计。", "location": "北京", "salary": "30k-60k", "color": "bg-gray-100 text-gray-600", "created_at": datetime.now(), "meta": {"source_type": "mock", "created_at": datetime.now().isoformat(), "updated_at": datetime.now().isoformat(), "chunk_ids": []}},
            {"id": 2, "jd_id": str(uuid.uuid4()), "company": "Baidu", "title": "大模型应用 PM", "description": "探索文心一言在垂直行业的落地场景。", "location": "上海", "salary": "25k-50k", "color": "bg-gray-100 text-gray-600", "created_at": datetime.now(), "meta": {"source_type": "mock", "created_at": datetime.now().isoformat(), "updated_at": datetime.now().isoformat(), "chunk_ids": []}},
            {"id": 3, "jd_id": str(uuid.uuid4()), "company": "Meituan", "title": "搜索推荐 AI PM", "description": "基于多模态理解提升搜索意图识别准确率。", "location": "北京", "salary": "35k-65k", "color": "bg-gray-100 text-gray-600", "created_at": datetime.now(), "meta": {"source_type": "mock", "created_at": datetime.now().isoformat(), "updated_at": datetime.now().isoformat(), "chunk_ids": []}},
            {"id": 4, "jd_id": str(uuid.uuid4()), "company": "ByteDance", "title": "抖音算法 product", "description": "负责短视频分发策略优化。", "location": "杭州", "salary": "40k-70k", "color": "bg-gray-100 text-gray-600", "created_at": datetime.now(), "meta": {"source_type": "mock", "created_at": datetime.now().isoformat(), "updated_at": datetime.now().isoformat(), "chunk_ids": []}},
        ]
        _save_jds(default_jds)
        return default_jds

    try:
        with open(JD_DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 把 created_at 字符串转回 datetime
        for jd in data:
            if isinstance(jd.get("created_at"), str):
                jd["created_at"] = datetime.fromisoformat(jd["created_at"])
        logger.info(f"[KnowledgeBase] loaded {len(data)} JDs from {JD_DATA_FILE}")
        return data
    except Exception as e:
        logger.error(f"[KnowledgeBase] failed to load JDs: {e}")
        return []


def _save_jds(jds: list):
    """保存 JD 列表到 JSON 文件"""
    try:
        with open(JD_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(jds, f, ensure_ascii=False, indent=2, default=_serialize_datetime)
    except Exception as e:
        logger.error(f"[KnowledgeBase] failed to save JDs: {e}")


# 启动时加载已有数据
MOCK_JDS = _load_jds()


@router.get("/", response_model=List[JDSchema])
async def list_jds():
    return [JDSchema(**jd) for jd in MOCK_JDS]


@router.post("/", response_model=JDSchema)
async def create_jd(data: JDCreateSchema):
    """添加新的 JD 到知识库（结构化存储 + 向量化入库）"""
    new_id = max([j["id"] for j in MOCK_JDS], default=0) + 1
    now = datetime.now()

    # 构造完整结构化 JD（兼容前端传入的解析结果）
    sections_data = data.sections.dict() if data.sections else {
        "responsibilities": data.description,
        "hard_requirements": [],
        "soft_requirements": [],
    }
    jd_record = {
        "id": new_id,
        "jd_id": str(uuid.uuid4()),
        "company": data.company,
        "title": data.title,
        "position": data.position or data.title,
        "description": data.description,
        "location": data.location,
        "salary": data.salary,
        "salary_range": data.salary_range or data.salary,
        "color": data.color,
        "sections": sections_data,
        "keywords": data.keywords or [],
        "structured_summary": data.structured_summary.dict() if data.structured_summary else {},
        "raw_text": data.raw_text or data.description,
        "meta": {
            "source_type": data.meta.source_type if data.meta else "paste",
            "created_at": data.meta.created_at if data.meta else now.isoformat(),
            "updated_at": now.isoformat(),
            "chunk_ids": [],
        },
        "created_at": now,
    }

    # 向量库入库：切分 + 向量化（默认使用 semantic 策略）
    vector_indexed = False
    try:
        # 延迟初始化 embedding_client（首次入库时）
        if vector_store.embedding_client is None:
            vector_store.embedding_client = EmbeddingClient.from_config()

        chunk_ids = await vector_store.add_jd(jd_record, strategy="semantic")
        jd_record["meta"]["chunk_ids"] = chunk_ids
        vector_indexed = len(chunk_ids) > 0

        if vector_indexed:
            logger.info(f"[create_jd] ✅ 入库成功 | jd_id={jd_record['jd_id']} company={jd_record['company']} position={jd_record['position']} chunks={len(chunk_ids)}")
        else:
            logger.warning(f"[create_jd] ⚠️ 入库异常 | jd_id={jd_record['jd_id']} 未生成有效 chunk")
    except Exception as e:
        logger.error(f"[create_jd] ❌ 入库失败 | jd_id={jd_record['jd_id']} error={e}")
        # 向量入库失败不阻断主流程，只记录日志

    jd_record["vector_indexed"] = vector_indexed
    MOCK_JDS.insert(0, jd_record)
    _save_jds(MOCK_JDS)
    logger.info(f"[create_jd] 💾 数据已持久化 | total_jds={len(MOCK_JDS)}")
    return JDSchema(**jd_record)


@router.delete("/{jd_id}")
async def delete_jd(jd_id: int):
    """删除指定 JD（同步清理向量库）"""
    global MOCK_JDS
    original_len = len(MOCK_JDS)

    # 找到要删除的 JD，清理向量库
    target = next((j for j in MOCK_JDS if j["id"] == jd_id), None)
    vector_cleaned = False
    if target:
        internal_jd_id = target.get("jd_id")
        company = target.get("company", "未知公司")
        position = target.get("position", "未知岗位")
        if internal_jd_id:
            try:
                vector_cleaned = vector_store.delete_jd(internal_jd_id)
                if vector_cleaned:
                    logger.info(f"[delete_jd] ✅ 向量库清理成功 | jd_id={internal_jd_id} company={company} position={position}")
                else:
                    logger.warning(f"[delete_jd] ⚠️ 向量库清理异常 | jd_id={internal_jd_id} company={company} position={position}")
            except Exception as e:
                logger.error(f"[delete_jd] ❌ 向量库清理失败 | jd_id={internal_jd_id} company={company} position={position} error={e}")
        else:
            logger.warning(f"[delete_jd] ⚠️ 该 JD 无内部 jd_id，跳过向量库清理 | id={jd_id}")

    MOCK_JDS = [j for j in MOCK_JDS if j["id"] != jd_id]
    if len(MOCK_JDS) == original_len:
        raise HTTPException(status_code=404, detail="JD not found")

    _save_jds(MOCK_JDS)
    logger.info(f"[delete_jd] ✅ JD 已删除 | id={jd_id} company={target.get('company') if target else '?'} vector_cleaned={vector_cleaned}")
    logger.info(f"[delete_jd] 💾 数据已持久化 | total_jds={len(MOCK_JDS)}")
    return {"message": "JD deleted successfully", "vector_cleaned": vector_cleaned}


@router.post("/parse-text")
async def parse_jd_text(data: dict):
    """文本粘贴方式解析 JD（调用 LLM 真实解析）"""
    raw_text = data.get("raw_text", "").strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="raw_text is required")

    llm = LLMClient.from_config("planner")
    parser = JDParser(llm_client=llm)
    parsed = await parser.parse(raw_text, source_type="paste")
    return {"parsed_schema": parsed, "status": "pending_validation"}


# 多模态 LLM 提取 JD 文本的 prompt
_VISION_JD_EXTRACT_PROMPT = """你是一位专业的 JD（岗位描述）信息提取助手。

任务：仔细查看用户上传的 JD 截图，将图片中的所有文字内容完整提取出来。

要求：
1. 尽可能完整地提取图片中的全部文字，不要遗漏任何信息
2. 保持原文的段落结构和列表格式
3. 如果图片中有表格，用文本方式描述表格内容
4. 只输出提取到的文字内容，不要添加任何解释或评论
5. 如果某些文字因模糊无法识别，用 [模糊] 标记

请直接输出提取到的文本："""


@router.post("/parse-image")
async def parse_jd_image(source_image: UploadFile = File(...)):
    """图片上传方式解析 JD（接入多模态 LLM 提取文本后解析）"""
    content = await source_image.read()
    filename = source_image.filename or "unknown"

    # 检测图片类型
    content_type = source_image.content_type or "image/jpeg"
    if content_type not in ("image/jpeg", "image/png", "image/webp", "image/gif"):
        # 尝试从文件名推断
        ext = filename.split(".")[-1].lower() if "." in filename else "jpeg"
        mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                    "webp": "image/webp", "gif": "image/gif"}
        content_type = mime_map.get(ext, "image/jpeg")

    # 转为 base64 data URI
    b64 = base64.b64encode(content).decode("utf-8")
    data_uri = f"data:{content_type};base64,{b64}"

    # 调用多模态 LLM 提取文本
    try:
        vision_llm = LLMClient.from_vision_config()
        logger.info(f"[parse_jd_image] calling vision LLM for {filename} ({len(content)} bytes, {content_type})")
        extracted_text = await vision_llm.vision_chat(
            system_prompt=_VISION_JD_EXTRACT_PROMPT,
            image_data_uris=[data_uri],
            temperature=0.3,
            max_tokens=4096,
        )
        logger.info(f"[parse_jd_image] vision LLM extracted text ({len(extracted_text)} chars):")
        logger.info(f"{extracted_text}")
    except Exception as e:
        logger.error(f"[parse_jd_image] vision LLM failed: {e}")
        raise HTTPException(status_code=500, detail=f"多模态解析失败: {e}")

    # 用提取的文本进行 JD 结构化解析
    llm = LLMClient.from_config("planner")
    parser = JDParser(llm_client=llm)
    parsed = await parser.parse(extracted_text, source_type="image")

    # 如果从图片文件名能推断公司名，做个兜底
    if parsed.get("company") == "未知公司":
        name_guess = filename.split(".")[0][:15]
        if name_guess and len(name_guess) > 2:
            parsed["company"] = name_guess

    logger.info(f"[parse_jd_image] final parsed schema for {filename}:")
    logger.info(f"company={parsed.get('company')} position={parsed.get('position')} keywords={parsed.get('keywords')}")

    return {"parsed_schema": parsed, "extracted_text": extracted_text, "status": "pending_validation"}


# ──────────────────────────── Chunk 策略对比测试接口 ────────────────────────────

@router.post("/chunk-test")
async def chunk_strategy_test(data: dict):
    """
    测试接口：传入 JD 原文 -> LLM 解析 -> 多种 chunk 策略切分 -> 返回对比结果

    请求体：
        {"raw_text": "岗位职责：...\n硬性要求：..."}

    返回：
        {
            "parsed_schema": {...},
            "fixed_size": {"stats": {...}, "chunks": [...]},
            "semantic":   {"stats": {...}, "chunks": [...]},
            "recursive_512": {"stats": {...}, "chunks": [...]},
            "recursive_256": {"stats": {...}, "chunks": [...]},
            "section":    {"stats": {...}, "chunks": [...]}
        }
    """
    raw_text = data.get("raw_text", "").strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="raw_text is required")

    # 1. 调用 LLM 解析为结构化 Schema
    try:
        llm = LLMClient.from_config("planner")
        parser = JDParser(llm_client=llm)
        parsed = await parser.parse(raw_text, source_type="test")
    except Exception as e:
        logger.error(f"[chunk_strategy_test] LLM parse failed: {e}")
        raise HTTPException(status_code=500, detail=f"LLM 解析失败: {e}")

    # 2. 五种策略切分
    chunks_fixed = chunk_fixed_size(parsed, size=512, overlap=50)
    chunks_semantic = chunk_semantic(parsed)
    chunks_rec_512 = chunk_recursive(parsed, max_length=512)
    chunks_rec_256 = chunk_recursive(parsed, max_length=256)
    chunks_section = chunk_by_section(parsed)

    def _fmt_chunks(chunks):
        return [
            {
                "index": c.metadata.get("index", i),
                "section": c.metadata.get("section", "-"),
                "priority": c.metadata.get("priority", "-"),
                "strategy": c.metadata.get("strategy", "-"),
                "length": len(c.content),
                "content": c.content,
            }
            for i, c in enumerate(chunks)
        ]

    def _stats(chunks):
        if not chunks:
            return {"count": 0, "avg_len": 0, "max_len": 0, "min_len": 0}
        lengths = [len(c.content) for c in chunks]
        return {
            "count": len(chunks),
            "avg_len": sum(lengths) // len(lengths),
            "max_len": max(lengths),
            "min_len": min(lengths),
        }

    logger.info(
        f"[chunk_strategy_test] fixed={len(chunks_fixed)} semantic={len(chunks_semantic)} "
        f"rec512={len(chunks_rec_512)} rec256={len(chunks_rec_256)} section={len(chunks_section)} "
        f"jd_company={parsed.get('company')} position={parsed.get('position')}"
    )

    return {
        "parsed_schema": parsed,
        "fixed_size": {
            "stats": _stats(chunks_fixed),
            "chunks": _fmt_chunks(chunks_fixed),
        },
        "semantic": {
            "stats": _stats(chunks_semantic),
            "chunks": _fmt_chunks(chunks_semantic),
        },
        "recursive_512": {
            "stats": _stats(chunks_rec_512),
            "chunks": _fmt_chunks(chunks_rec_512),
        },
        "recursive_256": {
            "stats": _stats(chunks_rec_256),
            "chunks": _fmt_chunks(chunks_rec_256),
        },
        "section": {
            "stats": _stats(chunks_section),
            "chunks": _fmt_chunks(chunks_section),
        },
    }
