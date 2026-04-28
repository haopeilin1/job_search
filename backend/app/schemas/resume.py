from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict


class BasicInfo(BaseModel):
    name: Optional[str] = Field(None, description="姓名")
    phone: Optional[str] = Field(None, description="电话")
    email: Optional[str] = Field(None, description="邮箱")
    years_exp: Optional[float] = Field(None, description="工作年限，精确到0.5年，如 3.5")
    current_company: Optional[str] = Field(None, description="当前公司")
    current_title: Optional[str] = Field(None, description="当前职位，如'高级产品经理'")
    location: Optional[str] = Field(None, description="当前城市")
    target_locations: Optional[List[str]] = Field(None, description="期望城市")
    target_salary_min: Optional[int] = Field(None, description="期望薪资下限（k）")
    target_salary_max: Optional[int] = Field(None, description="期望薪资上限（k）")
    availability: Optional[str] = Field(None, description="到岗时间：随时|1个月内|看机会")


class EducationItem(BaseModel):
    school: str = Field(..., description="学校名称")
    school_level: Optional[str] = Field(None, description="985|211|海外|双一流|普通")
    degree: Optional[str] = Field(None, description="本科/硕士/博士/大专")
    major: str = Field(..., description="专业")
    major_category: Optional[str] = Field(None, description="CS|EE|Math|Business|Design|Other")
    graduation_year: Optional[int] = Field(None, description="毕业年份")


class WorkExperienceItem(BaseModel):
    company: str = Field(..., description="公司名称")
    title: str = Field(..., description="职位名称")
    department: Optional[str] = Field(None, description="部门，如'AI平台部'")
    start_date: Optional[str] = Field(None, description="入职时间，如'2022-03'")
    end_date: Optional[str] = Field(None, description="离职时间，如'2024-05'")
    is_current: bool = Field(False, description="是否当前在职")
    description: str = Field(default="", description="职责原文")
    achievements: List[str] = Field(default_factory=list, description="量化成果，如'DAU提升30%'")
    team_size: Optional[int] = Field(None, description="管理幅度，带团队人数")
    keywords: List[str] = Field(default_factory=list, description="业务/技术关键词")


class ProjectItem(BaseModel):
    name: str = Field(..., description="项目名称")
    role: str = Field(..., description="担任角色：产品经理|项目负责人|核心成员")
    company: Optional[str] = Field(None, description="关联哪段工作经历")
    description: str = Field(default="", description="详细描述")
    tech_keywords: List[str] = Field(default_factory=list, description="技术关键词：RAG、LangChain")
    business_keywords: List[str] = Field(default_factory=list, description="业务关键词：大模型、推荐系统")
    metrics: Optional[List[str]] = Field(None, description="量化成果，用于压力题验证")
    duration: Optional[str] = Field(None, description="起止时间")
    is_key_project: bool = Field(False, description="是否核心项目（匹配时加权）")


class SkillsBucket(BaseModel):
    technical: List[str] = Field(default_factory=list, description="硬技能：Python、SQL、A/B测试、RAG")
    business: List[str] = Field(default_factory=list, description="业务技能：需求分析、数据分析")
    soft: List[str] = Field(default_factory=list, description="软技能：跨部门协作、项目管理")
    proficiency_map: Dict[str, Optional[str]] = Field(default_factory=dict, description='技能熟练度：{"Python": "精通", "SQL": "熟练"}')


class CertificationItem(BaseModel):
    name: str = Field(..., description="证书名称：PMP、NPDP、CFA、软考等")
    issuer: Optional[str] = Field(None, description="颁发机构")


class ResumeMeta(BaseModel):
    resume_id: str = Field(..., description="UUID")
    raw_text: str = Field(..., description="原始解析文本备份")
    parsed_at: str = Field(..., description="ISO 8601 时间戳")
    parser_version: str = Field(..., description="解析器版本")
    confidence_score: Optional[float] = Field(None, description="解析置信度 0-1")
    is_active: bool = Field(..., description="是否为当前生效简历")
    source_type: Optional[str] = Field(None, description="pdf|docx|txt|image")


class ResumeSchema(BaseModel):
    basic_info: BasicInfo = Field(default_factory=BasicInfo, description="基础身份与求职意向")
    education: List[EducationItem] = Field(default_factory=list, description="教育背景")
    work_experience: List[WorkExperienceItem] = Field(default_factory=list, description="工作经历")
    projects: List[ProjectItem] = Field(default_factory=list, description="项目经历")
    skills: SkillsBucket = Field(default_factory=SkillsBucket, description="技能栈")
    certifications: List[CertificationItem] = Field(default_factory=list, description="证书与加分项")
    advantages: List[str] = Field(default_factory=list, description="个人优势/自我评价")
    meta: Optional[ResumeMeta] = Field(None, description="系统元数据")


class ExtractedPreview(BaseModel):
    name: Optional[str] = Field(None, description="姓名")
    years_exp: Optional[float] = Field(None, description="工作年限")
    skill_count: int = Field(0, description="技能数量")
    project_count: int = Field(0, description="项目数量")
    work_count: int = Field(0, description="工作经历数量")


class ResumeUploadResponse(BaseModel):
    resume_id: str = Field(..., description="UUID")
    parsed_schema: ResumeSchema = Field(..., description="结构化解析结果")
    is_active: bool = Field(..., description="是否设为当前生效简历")
    extracted_preview: ExtractedPreview = Field(..., description="前端快速展示用")
    status: str = Field(..., description="parsed | parsing | error")


class ResumeCurrentResponse(BaseModel):
    resume_id: str = Field(..., description="UUID")
    parsed_schema: ResumeSchema = Field(..., description="结构化解析结果")
    updated_at: str = Field(..., description="ISO 8601 时间戳")


class ResumeListItem(BaseModel):
    resume_id: str = Field(..., description="UUID")
    parsed_schema: ResumeSchema = Field(..., description="结构化解析结果")
    created_at: str = Field(..., description="创建时间")


class ResumeListResponse(BaseModel):
    items: List[ResumeListItem] = Field(default_factory=list, description="简历列表")
    active_id: Optional[str] = Field(None, description="当前生效简历 ID")


class ResumeUpdateRequest(BaseModel):
    parsed_schema: ResumeSchema = Field(..., description="用户编辑后的完整结构")


class ResumeUpdateResponse(BaseModel):
    resume_id: str = Field(..., description="UUID")
    parsed_schema: ResumeSchema = Field(..., description="更新后的结构化结果")
    updated_at: str = Field(..., description="ISO 8601 时间戳")


class ResumeActivateResponse(BaseModel):
    active_id: str = Field(..., description="当前生效简历 ID")
