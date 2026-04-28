import json
import logging
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.core.file_ops import FileOps
from app.core.llm_client import LLMClient
from app.core.prompts import RESUME_PARSE_PROMPT
from app.core import state as app_state
from app.schemas.resume import (
    ResumeSchema,
    ResumeUploadResponse,
    ResumeCurrentResponse,
    ResumeListResponse,
    ResumeListItem,
    ResumeUpdateRequest,
    ResumeUpdateResponse,
    ResumeActivateResponse,
    ResumeMeta,
    ExtractedPreview,
    BasicInfo,
    EducationItem,
    WorkExperienceItem,
    ProjectItem,
    SkillsBucket,
    CertificationItem,
)

router = APIRouter(prefix="/api/v1/resumes", tags=["resumes"])
logger = logging.getLogger(__name__)


def _safe_get(data: dict, key: str, default=None):
    """安全地从字典取值"""
    if data is None:
        return default
    return data.get(key, default)


def _as_list(val, item_transform=str):
    """将任意值安全转为列表"""
    if val is None:
        return []
    if isinstance(val, list):
        return [item_transform(x) for x in val if x is not None]
    return []


def _build_resume_schema(parsed: Optional[dict], raw_text: str, source_type: str) -> ResumeSchema:
    """将 LLM 返回的 JSON 安全转换为 ResumeSchema"""
    if parsed is None:
        parsed = {}

    confidence = _safe_get(parsed, "confidence_score")
    if confidence is None:
        confidence = 0.85 if parsed else 0.0

    # ---------- basic_info ----------
    bi = _safe_get(parsed, "basic_info", {})
    basic_info = BasicInfo(
        name=_safe_get(bi, "name") or None,
        phone=_safe_get(bi, "phone") or None,
        email=_safe_get(bi, "email") or None,
        years_exp=_safe_get(bi, "years_exp"),
        current_company=_safe_get(bi, "current_company") or None,
        current_title=_safe_get(bi, "current_title") or None,
        location=_safe_get(bi, "location") or None,
        target_locations=_as_list(_safe_get(bi, "target_locations")),
        target_salary_min=_safe_get(bi, "target_salary_min"),
        target_salary_max=_safe_get(bi, "target_salary_max"),
        availability=_safe_get(bi, "availability") or None,
    )

    # ---------- education ----------
    education = []
    for edu in _safe_get(parsed, "education", []) or []:
        if not edu:
            continue
        education.append(
            EducationItem(
                school=_safe_get(edu, "school") or "",
                school_level=_safe_get(edu, "school_level") or None,
                degree=_safe_get(edu, "degree") or None,
                major=_safe_get(edu, "major") or "",
                major_category=_safe_get(edu, "major_category") or None,
                graduation_year=_safe_get(edu, "graduation_year"),
            )
        )

    # ---------- work_experience ----------
    work_experience = []
    for we in _safe_get(parsed, "work_experience", []) or []:
        if not we:
            continue
        work_experience.append(
            WorkExperienceItem(
                company=_safe_get(we, "company") or "",
                title=_safe_get(we, "title") or "",
                department=_safe_get(we, "department") or None,
                start_date=_safe_get(we, "start_date") or None,
                end_date=_safe_get(we, "end_date") or None,
                is_current=bool(_safe_get(we, "is_current", False)),
                description=_safe_get(we, "description") or "",
                achievements=_as_list(_safe_get(we, "achievements")),
                team_size=_safe_get(we, "team_size"),
                keywords=_as_list(_safe_get(we, "keywords")),
            )
        )

    # ---------- projects ----------
    projects = []
    for proj in _safe_get(parsed, "projects", []) or []:
        if not proj:
            continue
        projects.append(
            ProjectItem(
                name=_safe_get(proj, "name") or "",
                role=_safe_get(proj, "role") or "",
                company=_safe_get(proj, "company") or None,
                description=_safe_get(proj, "description") or "",
                tech_keywords=_as_list(_safe_get(proj, "tech_keywords")),
                business_keywords=_as_list(_safe_get(proj, "business_keywords")),
                metrics=_as_list(_safe_get(proj, "metrics")),
                duration=_safe_get(proj, "duration") or None,
                is_key_project=bool(_safe_get(proj, "is_key_project", False)),
            )
        )

    # ---------- skills ----------
    sk = _safe_get(parsed, "skills", {})
    if isinstance(sk, list):
        # 兼容旧格式：如果是列表，全部放入 technical
        sk = {"technical": [str(x) for x in sk if x], "business": [], "soft": [], "proficiency_map": {}}
    proficiency_map = _safe_get(sk, "proficiency_map", {})
    if not isinstance(proficiency_map, dict):
        proficiency_map = {}
    skills = SkillsBucket(
        technical=_as_list(_safe_get(sk, "technical")),
        business=_as_list(_safe_get(sk, "business")),
        soft=_as_list(_safe_get(sk, "soft")),
        proficiency_map={k: v for k, v in proficiency_map.items() if v},
    )

    # ---------- certifications ----------
    certifications = []
    for cert in _safe_get(parsed, "certifications", []) or []:
        if not cert:
            continue
        certifications.append(
            CertificationItem(
                name=_safe_get(cert, "name") or "",
                issuer=_safe_get(cert, "issuer") or None,
            )
        )

    # ---------- advantages ----------
    advantages = _as_list(_safe_get(parsed, "advantages"))

    return ResumeSchema(
        basic_info=basic_info,
        education=education,
        work_experience=work_experience,
        projects=projects,
        skills=skills,
        certifications=certifications,
        advantages=advantages,
        meta=None,
    )


def _make_resume_meta(
    resume_id: str, raw_text: str, is_active: bool,
    parser_version: str, confidence_score: float, source_type: str
) -> ResumeMeta:
    return ResumeMeta(
        resume_id=resume_id,
        raw_text=raw_text,
        parsed_at=datetime.now().isoformat(),
        parser_version=parser_version,
        confidence_score=confidence_score,
        is_active=is_active,
        source_type=source_type,
    )


@router.post("/upload", response_model=ResumeUploadResponse)
async def upload_resume(
    file: UploadFile = File(...),
    force_update: bool = Form(False),
    parser_mode: str = Form("text"),
):
    """上传简历文件并解析为结构化数据。
    
    解析策略：
    1. 先尝试用 FileOps.extract_text() 提取纯文本
    2. 若文本提取成功，调用文字处理 LLM（chat 模型）解析
    3. 若文本提取失败 或 文字 LLM 解析失败，使用多模态 LLM（vision 模型）兜底
       - 仅 PDF 支持多模态兜底（先转图片再送入 vision 模型）
    """
    file_bytes = await file.read()
    filename = file.filename or "resume.pdf"
    ext = filename.lower().split(".")[-1] if "." in filename else ""
    source_type_map = {"pdf": "pdf", "docx": "docx", "doc": "docx", "txt": "txt"}
    source_type = source_type_map.get(ext, "txt")
    logger.info(f"[Upload] received file={filename}, size={len(file_bytes)} bytes")

    # ---------- 1. 文本提取（两种模式都需要原始文本用于保底）----------
    extracted_text = ""
    text_ok = False
    try:
        extracted_text = FileOps.extract_text(file_bytes, filename)
        if extracted_text and len(extracted_text.strip()) > 20:
            text_ok = True
            logger.info(f"[Upload] text extracted, length={len(extracted_text)}")
        else:
            logger.warning("[Upload] extracted text too short, treat as failure")
    except Exception as e:
        logger.warning(f"[Upload] text extraction failed: {e}")

    # ---------- 2. 获取 LLM 配置 ----------
    cfg = app_state.llm_config_store
    parsed_data: Optional[dict] = None
    parser_version = ""

    if parser_mode == "vision":
        # ---------- Vision 直接模式：跳过文本 LLM，直接使用多模态模型 ----------
        logger.info("[Upload] parser_mode=vision, using vision LLM directly")
        try:
            if not cfg.vision.api_key:
                raise RuntimeError("vision API key empty")
            if not filename.lower().endswith(".pdf"):
                raise RuntimeError("vision mode only supports PDF")

            images = FileOps.pdf_to_images(file_bytes)
            if not images:
                raise RuntimeError("PDF converted to 0 images")

            source_type = "image"
            data_uris = FileOps.images_to_base64_data_uris(images)
            raw = LLMClient.call_vision(
                base_url=cfg.vision.base_url,
                api_key=cfg.vision.api_key,
                model=cfg.vision.model,
                system_prompt=RESUME_PARSE_PROMPT,
                image_data_uris=data_uris,
            )
            parsed_data = LLMClient.safe_parse_json(raw)
            if parsed_data:
                parser_version = f"vision-llm:{cfg.vision.model}"
                logger.info("[Upload] vision LLM parse succeeded")
            else:
                logger.warning("[Upload] vision LLM returned non-JSON")
        except Exception as e:
            logger.warning(f"[Upload] vision parse failed: {e}")
    else:
        # ---------- Text 模式：原有逻辑（文本提取 → 文本 LLM → Vision 兜底）----------
        # ---------- 3. 策略 A：文本 LLM ----------
        if text_ok:
            try:
                if not cfg.core.api_key:
                    logger.warning("[Upload] core API key empty, skip text LLM")
                else:
                    raw = LLMClient.call_text(
                        base_url=cfg.core.base_url,
                        api_key=cfg.core.api_key,
                        model=cfg.core.model,
                        system_prompt=RESUME_PARSE_PROMPT,
                        user_text=extracted_text,
                    )
                    parsed_data = LLMClient.safe_parse_json(raw)
                    if parsed_data:
                        parser_version = f"text-llm:{cfg.core.model}"
                        logger.info("[Upload] text LLM parse succeeded")
                    else:
                        logger.warning("[Upload] text LLM returned non-JSON")
            except Exception as e:
                logger.warning(f"[Upload] text LLM parse failed: {e}")

        # ---------- 4. 策略 B：多模态兜底 ----------
        if parsed_data is None:
            try:
                if not cfg.vision.api_key:
                    raise RuntimeError("vision API key empty")
                if not filename.lower().endswith(".pdf"):
                    raise RuntimeError("fallback vision only supports PDF")

                logger.info("[Upload] entering vision fallback")
                images = FileOps.pdf_to_images(file_bytes)
                if not images:
                    raise RuntimeError("PDF converted to 0 images")

                source_type = "image"
                data_uris = FileOps.images_to_base64_data_uris(images)
                raw = LLMClient.call_vision(
                    base_url=cfg.vision.base_url,
                    api_key=cfg.vision.api_key,
                    model=cfg.vision.model,
                    system_prompt=RESUME_PARSE_PROMPT,
                    image_data_uris=data_uris,
                )
                parsed_data = LLMClient.safe_parse_json(raw)
                if parsed_data:
                    parser_version = f"vision-llm:{cfg.vision.model}"
                    logger.info("[Upload] vision LLM parse succeeded")
                else:
                    logger.warning("[Upload] vision LLM returned non-JSON")
            except Exception as e:
                logger.warning(f"[Upload] vision fallback failed: {e}")

    # ---------- 5. 无 LLM 兜底：直接保存原始文本 ----------
    if parsed_data is None:
        logger.info("[Upload] no LLM available, saving raw text only")
        parser_version = "raw-text-only"
        # 用原始文本构造一个最小化的 parsed_data，让后续流程正常走完
        parsed_data = {
            "confidence_score": 0.5,
            "basic_info": {},
            "education": [],
            "work_experience": [],
            "projects": [],
            "skills": {"technical": [], "business": [], "soft": [], "proficiency_map": {}},
            "certifications": [],
            "advantages": [],
        }

    # ---------- 5. 组装结果并入库 ----------
    resume_id = str(uuid.uuid4())
    is_active = force_update or (app_state.active_resume_id is None)
    confidence = _safe_get(parsed_data, "confidence_score", 0.85) if parsed_data else 0.0

    # 如果文本提取很少且最终使用了 vision 模型，在 raw_text 前加提示，避免用户觉得"解析内容与原始文本不匹配"
    raw_text_for_meta = extracted_text
    if not text_ok and (parser_mode == "vision" or parser_version.startswith("vision-llm")):
        raw_text_for_meta = (
            f"[该文件为扫描版/图片型PDF，原始文本提取不完整。"
            f"以下结构化数据由多模态模型直接基于图片解析生成]\n\n"
            f"{extracted_text or ''}"
        )

    schema = _build_resume_schema(parsed_data, extracted_text, source_type)
    schema.meta = _make_resume_meta(
        resume_id, raw_text_for_meta, is_active, parser_version, confidence, source_type
    )

    app_state.resumes_db[resume_id] = {
        "id": resume_id,
        "parsed_schema": schema.model_dump(),
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }

    if is_active:
        app_state.active_resume_id = resume_id
        for rid, item in app_state.resumes_db.items():
            ps = item["parsed_schema"]
            if ps.get("meta"):
                ps["meta"]["is_active"] = (rid == resume_id)

    # 持久化到磁盘
    app_state.save_resumes()

    preview = ExtractedPreview(
        name=schema.basic_info.name,
        years_exp=schema.basic_info.years_exp,
        skill_count=len(schema.skills.technical) + len(schema.skills.business) + len(schema.skills.soft),
        project_count=len(schema.projects),
        work_count=len(schema.work_experience),
    )

    return ResumeUploadResponse(
        resume_id=resume_id,
        parsed_schema=schema,
        is_active=is_active,
        extracted_preview=preview,
        status="parsed",
    )


@router.get("/current", response_model=ResumeCurrentResponse)
async def get_current_resume():
    """获取当前生效简历"""
    if not app_state.active_resume_id or app_state.active_resume_id not in app_state.resumes_db:
        empty_schema = ResumeSchema(
            basic_info=BasicInfo(),
            education=[],
            work_experience=[],
            projects=[],
            skills=SkillsBucket(),
            certifications=[],
            advantages=[],
            meta=None,
        )
        return ResumeCurrentResponse(
            resume_id="",
            parsed_schema=empty_schema,
            updated_at=datetime.now().isoformat(),
        )

    item = app_state.resumes_db[app_state.active_resume_id]
    return ResumeCurrentResponse(
        resume_id=app_state.active_resume_id,
        parsed_schema=ResumeSchema(**item["parsed_schema"]),
        updated_at=item["updated_at"],
    )


@router.get("/", response_model=ResumeListResponse)
async def list_resumes():
    """获取简历列表（支持历史版本）"""
    items = []
    for rid, item in app_state.resumes_db.items():
        items.append(
            ResumeListItem(
                resume_id=rid,
                parsed_schema=ResumeSchema(**item["parsed_schema"]),
                created_at=item["created_at"],
            )
        )
    return ResumeListResponse(items=items, active_id=app_state.active_resume_id)


@router.put("/{resume_id}", response_model=ResumeUpdateResponse)
async def update_resume(resume_id: str, req: ResumeUpdateRequest):
    """更新简历解析结果（用户手动修正）"""
    if resume_id not in app_state.resumes_db:
        raise HTTPException(status_code=404, detail="Resume not found")

    parsed = req.parsed_schema.model_dump()
    old_meta = app_state.resumes_db[resume_id]["parsed_schema"].get("meta")
    parsed["meta"] = old_meta

    app_state.resumes_db[resume_id]["parsed_schema"] = parsed
    app_state.resumes_db[resume_id]["updated_at"] = datetime.now().isoformat()

    # 持久化到磁盘
    app_state.save_resumes()

    return ResumeUpdateResponse(
        resume_id=resume_id,
        parsed_schema=ResumeSchema(**parsed),
        updated_at=app_state.resumes_db[resume_id]["updated_at"],
    )


@router.post("/{resume_id}/reparse", response_model=ResumeUpdateResponse)
async def reparse_resume(resume_id: str, raw_text: str = Form(...)):
    """基于用户编辑后的原始文本重新解析简历。

    1. 调用文本 LLM 重新解析 raw_text
    2. 保留原简历的 resume_id / is_active / source_type
    3. 更新 parser_version / parsed_at / confidence_score
    """
    if resume_id not in app_state.resumes_db:
        raise HTTPException(status_code=404, detail="Resume not found")

    old_item = app_state.resumes_db[resume_id]
    old_schema = old_item.get("parsed_schema", {})
    old_meta = old_schema.get("meta", {}) or {}

    cfg = app_state.llm_config_store
    parsed_data: Optional[dict] = None
    parser_version = ""

    try:
        if not cfg.core.api_key:
            raise RuntimeError("core API key empty")
        raw = LLMClient.call_text(
            base_url=cfg.core.base_url,
            api_key=cfg.core.api_key,
            model=cfg.core.model,
            system_prompt=RESUME_PARSE_PROMPT,
            user_text=raw_text,
        )
        parsed_data = LLMClient.safe_parse_json(raw)
        if parsed_data:
            parser_version = f"text-llm:{cfg.core.model}"
            logger.info(f"[Reparse] text LLM parse succeeded | resume_id={resume_id}")
        else:
            logger.warning("[Reparse] text LLM returned non-JSON")
    except Exception as e:
        logger.warning(f"[Reparse] text LLM parse failed: {e}")

    if parsed_data is None:
        logger.info("[Reparse] no LLM available, saving raw text only")
        parser_version = "raw-text-only"
        parsed_data = {
            "confidence_score": 0.5,
            "basic_info": {},
            "education": [],
            "work_experience": [],
            "projects": [],
            "skills": {"technical": [], "business": [], "soft": [], "proficiency_map": {}},
            "certifications": [],
            "advantages": [],
        }

    confidence = _safe_get(parsed_data, "confidence_score", 0.85) if parsed_data else 0.0
    source_type = old_meta.get("source_type", "txt")

    schema = _build_resume_schema(parsed_data, raw_text, source_type)
    schema.meta = ResumeMeta(
        resume_id=resume_id,
        raw_text=raw_text,
        parsed_at=datetime.now().isoformat(),
        parser_version=parser_version,
        confidence_score=confidence,
        is_active=old_meta.get("is_active", False),
        source_type=source_type,
    )

    app_state.resumes_db[resume_id]["parsed_schema"] = schema.model_dump()
    app_state.resumes_db[resume_id]["updated_at"] = datetime.now().isoformat()
    app_state.save_resumes()

    return ResumeUpdateResponse(
        resume_id=resume_id,
        parsed_schema=schema,
        updated_at=app_state.resumes_db[resume_id]["updated_at"],
    )


@router.put("/{resume_id}/activate", response_model=ResumeActivateResponse)
async def activate_resume(resume_id: str):
    """切换生效简历"""
    if resume_id not in app_state.resumes_db:
        raise HTTPException(status_code=404, detail="Resume not found")

    app_state.active_resume_id = resume_id
    for rid, item in app_state.resumes_db.items():
        ps = item["parsed_schema"]
        if ps.get("meta"):
            ps["meta"]["is_active"] = (rid == resume_id)

    # 持久化到磁盘
    app_state.save_resumes()

    return ResumeActivateResponse(active_id=resume_id)


@router.post("/match", response_model=dict)
async def match_jd(
    file: UploadFile = File(None),
    jd_text: str = Form(""),
):
    """上传 JD 并与简历进行匹配分析（TODO: 接入真实 LLM 匹配逻辑）"""
    return {
        "score": 88,
        "label": "优秀契合",
        "advantage": "Agent 架构经验",
        "weakness": "SQL 深度优化",
        "questions": [
            {
                "tag": "场景考察",
                "text": "你曾主导过 Agent 项目，请详细说明在模型幻觉严重时，你如何设计闭环反馈路径？",
            },
            {
                "tag": "压力测试",
                "text": "如果业务方要求 2 周内上线一个多模态检索模型，但数据集质量极差，你会如何权衡？",
            },
        ],
    }
