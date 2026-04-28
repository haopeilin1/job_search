from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List


class JDSections(BaseModel):
    """JD 结构化内容"""
    responsibilities: str = Field(default="", description="岗位职责完整原文")
    hard_requirements: List[str] = Field(default_factory=list, description="硬性要求（逐条数组）")
    soft_requirements: List[str] = Field(default_factory=list, description="软性要求/加分项（逐条数组）")


class JDStructuredSummary(BaseModel):
    """JD 结构化摘要（用于粗筛层快速匹配）"""
    min_years: Optional[int] = Field(None, description="最低工作年限要求")
    max_years: Optional[int] = Field(None, description="最高工作年限上限")
    min_education: Optional[str] = Field(None, description="最低学历要求（博士/硕士/本科/大专）")
    category: Optional[str] = Field(None, description="岗位大类（技术/产品/运营/设计/市场/销售/职能/其他）")
    domain: Optional[str] = Field(None, description="业务领域（AI/电商/金融/教育等）")


class JDMeta(BaseModel):
    """JD 系统元数据"""
    source_type: str = Field(default="paste", description="来源类型: paste/image/pdf")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="存入时间")
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="更新时间")
    chunk_ids: List[str] = Field(default_factory=list, description="指向 ChromaDB 的 chunk ID 列表")
    fallback_reason: Optional[str] = Field(default=None, description="fallback 原因（如有）")


class JDCreateSchema(BaseModel):
    """创建 JD 请求模型"""
    company: str = Field(..., description="公司名称")
    title: str = Field(..., description="职位名称")
    description: str = Field(..., description="职位描述（向后兼容）")
    location: Optional[str] = Field("远程", description="工作地点")
    salary: Optional[str] = Field("薪资面议", description="薪资范围")
    color: Optional[str] = Field("bg-gray-100 text-gray-600", description="UI 配色类名")
    # 新增结构化字段（可选，前端传入解析结果时使用）
    position: Optional[str] = Field(None, description="岗位名（与 title 兼容）")
    salary_range: Optional[str] = Field(None, description="薪资范围（与 salary 兼容）")
    sections: Optional[JDSections] = Field(None, description="结构化内容")
    keywords: Optional[List[str]] = Field(None, description="关键词列表")
    structured_summary: Optional[JDStructuredSummary] = Field(None, description="结构化摘要（用于粗筛）")
    raw_text: Optional[str] = Field(None, description="原始文本备份")
    meta: Optional[JDMeta] = Field(None, description="系统元数据")


class JDUpdateSchema(BaseModel):
    """更新 JD 请求模型"""
    company: Optional[str] = Field(None, description="公司名称")
    title: Optional[str] = Field(None, description="职位名称")
    description: Optional[str] = Field(None, description="职位描述")
    location: Optional[str] = Field(None, description="工作地点")
    salary: Optional[str] = Field(None, description="薪资范围")


class JDSchema(BaseModel):
    """JD 响应模型（完整结构化）"""
    id: int = Field(..., description="JD 唯一标识（展示用）")
    jd_id: Optional[str] = Field(None, description="系统内部 UUID")
    company: str = Field(..., description="公司名称")
    title: str = Field(..., description="职位名称")
    position: Optional[str] = Field(None, description="岗位名")
    description: str = Field(..., description="职位描述")
    location: Optional[str] = Field(None, description="工作地点")
    salary: Optional[str] = Field(None, description="薪资范围")
    salary_range: Optional[str] = Field(None, description="薪资范围（结构化字段）")
    color: Optional[str] = Field(None, description="UI 配色类名")
    sections: Optional[JDSections] = Field(None, description="结构化内容")
    keywords: Optional[List[str]] = Field(None, description="关键词列表")
    raw_text: Optional[str] = Field(None, description="原始文本备份")
    meta: Optional[JDMeta] = Field(None, description="系统元数据")
    created_at: datetime = Field(..., description="创建时间")
    # 向量库入库状态（仅用于前后端状态同步展示）
    vector_indexed: Optional[bool] = Field(None, description="是否成功写入向量库")

    class Config:
        from_attributes = True
