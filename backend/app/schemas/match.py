from pydantic import BaseModel, Field
from typing import List, Optional


class QuestionSchema(BaseModel):
    tag: str = Field(..., description="题目类型标签，如场景考察/压力测试")
    text: str = Field(..., description="面试题内容")


class MatchRequestSchema(BaseModel):
    resume_id: Optional[int] = Field(None, description="简历 ID，不传则使用默认简历")
    jd_text: Optional[str] = Field(None, description="JD 文本内容")
    jd_file_name: Optional[str] = Field(None, description="上传的 JD 文件名")


class MatchResultSchema(BaseModel):
    score: int = Field(..., ge=0, le=100, description="匹配分数 0-100")
    label: str = Field(..., description="匹配等级标签，如优秀契合/良好/一般")
    advantage: str = Field(..., description="关键优势")
    weakness: str = Field(..., description="待补充项")
    questions: List[QuestionSchema] = Field(..., description="专属面试题预测列表")
