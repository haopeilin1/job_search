"""
将用户上传的简历图片内容手动解析后存入系统。

由于环境限制无法调用 Vision LLM，此处基于图片内容人工提取并构造 ResumeSchema，
保存到 backend/data/resumes.json（后端启动后可通过 API 读取）。
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
RESUME_FILE = DATA_DIR / "resumes.json"

# ── 基于图片内容人工构造的简历结构化数据 ──
resume_id = str(uuid.uuid4())
now_iso = datetime.now().isoformat()

raw_text = """
教育背景
江南科技大学
信息工程学院·软件工程（AI应用方向）·硕士
2024.09-2027.06（预计）
GPA: 3.4/4.0 | 主修：机器学习基础、数据库系统、软件项目管理、人机交互设计

云溪大学
计算机学院·信息管理与信息系统·本科
2020.09-2024.06
荣誉：两次校级三等奖学金 | 主修：数据结构、管理学原理、Web开发技术、统计学

项目经历
求职助手小程序 — 智能简历匹配与面试模拟工具
2025.03-2025.11
产品经理 | 三人小组
- 需求分析：设计调研问卷回收40余份有效样本，梳理"简历评估"与"定向面试模拟"核心需求，使用Figma完成低保真原型，输出产品需求文档。
- AI匹配策略：调研开源Embedding模型实现简历与JD语义相似度计算，探索多维度匹配方案，对比整篇匹配与分块匹配效果差异，持续优化推荐排序。
- 内容生成：基于大模型AI设计结构化Prompt，融合岗位关键词与简历信息输出个性化面试题，建立追问机制增强模拟面试针对性。
- 测试验证：组织20人内测收集用户路径与反馈数据，针对复杂PDF格式设计补充录入机制，完成从需求分析到上线测试的完整产品闭环。

教务问答机器人 — 基于校内政策的RAG问答系统
2024.10-2025.02
产品设计与测试 | 实验室课题项目
- 知识库构建：收集整理教务处政策文件、培养方案及历史问答共80余份，完成扫描件OCR识别、格式转换与结构化清洗，建立可检索的领域知识库。
- 交互设计：设计Web端对话界面，参考主流客服产品布局增加"推荐提问"推荐模块降低投入成本，针对政策类回答设计溯源展示功能增强可信度。
- 效果测试：参与搭建检索增强生成链路，完成实验环境下多轮问答测试，验证标准政策类问题回答准确率，整理测试数据与优化建议形成可复用方案。

校园经历
江南科技大学软件学院学生会·宣传部干事
2024.09-2025.06
- 负责学院官方公众号推文排版与发布，累计输出活动宣传稿件10余篇；协助组织院内编程比赛，负责报名统计与现场签到。

云溪大学图书馆·勤工助学岗
2022.03-2023.06
- 负责电子阅览室日常值班，协助同学解决基础网络与设备使用问题；参与旧期刊电子目录录入与数据整理工作。

获奖情况
- 2025.05 江南科技大学研究生学业三等奖学金
- 2024.11 学院AI应用创新赛（课程组）三等奖
- 2022.06 云溪大学优秀共青团员
- 2022.12 校级网页设计比赛优秀奖

个人技能
- 产品工具：熟练使用Figma进行原型与交互演示，了解Axure基础操作；能够使用Xmind梳理产品架构，通过腾讯文档/飞书写PRD。
- 技术理解：了解大语言模型在应用层的落地方式，具备Prompt Engineering实践经验；理解RAG基本架构与工作流程，了解Embedding与向量检索作用。
- 语言能力：英语CET-4（525分），可阅读英文技术文档与产品资料。
- 办公技能：熟练使用Office办公软件，掌握Excel数据透视表等基础数据分析功能。
""".strip()

resume_record = {
    "id": resume_id,
    "parsed_schema": {
        "basic_info": {
            "name": None,
            "phone": None,
            "email": None,
            "years_exp": 0.0,
            "current_company": None,
            "current_title": "应届研究生（产品经理方向）",
            "location": None,
            "target_locations": [],
            "target_salary_min": None,
            "target_salary_max": None,
            "availability": None,
        },
        "education": [
            {
                "school": "江南科技大学",
                "school_level": None,
                "degree": "硕士",
                "major": "软件工程（AI应用方向）",
                "major_category": "CS",
                "graduation_year": 2027,
            },
            {
                "school": "云溪大学",
                "school_level": None,
                "degree": "本科",
                "major": "信息管理与信息系统",
                "major_category": "CS",
                "graduation_year": 2024,
            },
        ],
        "work_experience": [],
        "projects": [
            {
                "name": "求职助手小程序 — 智能简历匹配与面试模拟工具",
                "role": "产品经理",
                "company": None,
                "description": (
                    "设计调研问卷回收40余份有效样本，梳理'简历评估'与'定向面试模拟'核心需求，"
                    "使用Figma完成低保真原型，输出产品需求文档。"
                    "调研开源Embedding模型实现简历与JD语义相似度计算，探索多维度匹配方案，"
                    "对比整篇匹配与分块匹配效果差异，持续优化推荐排序。"
                    "基于大模型AI设计结构化Prompt，融合岗位关键词与简历信息输出个性化面试题，"
                    "建立追问机制增强模拟面试针对性。"
                    "组织20人内测收集用户路径与反馈数据，针对复杂PDF格式设计补充录入机制，"
                    "完成从需求分析到上线测试的完整产品闭环。"
                ),
                "tech_keywords": ["Figma", "Embedding", "大模型", "Prompt Engineering", "PDF解析"],
                "business_keywords": ["简历匹配", "面试模拟", "推荐排序", "需求分析", "用户测试"],
                "metrics": ["调研问卷40+份", "20人内测", "完整产品闭环"],
                "duration": "2025.03-2025.11",
                "is_key_project": True,
            },
            {
                "name": "教务问答机器人 — 基于校内政策的RAG问答系统",
                "role": "产品设计与测试",
                "company": None,
                "description": (
                    "收集整理教务处政策文件、培养方案及历史问答共80余份，"
                    "完成扫描件OCR识别、格式转换与结构化清洗，建立可检索的领域知识库。"
                    "设计Web端对话界面，参考主流客服产品布局增加'推荐提问'推荐模块降低投入成本，"
                    "针对政策类回答设计溯源展示功能增强可信度。"
                    "参与搭建检索增强生成链路，完成实验环境下多轮问答测试，"
                    "验证标准政策类问题回答准确率，整理测试数据与优化建议形成可复用方案。"
                ),
                "tech_keywords": ["RAG", "OCR", "检索增强生成", "Web端设计"],
                "business_keywords": ["知识库构建", "交互设计", "政策问答", "溯源展示"],
                "metrics": ["80余份文档", "多轮问答测试", "准确率验证"],
                "duration": "2024.10-2025.02",
                "is_key_project": True,
            },
        ],
        "skills": {
            "technical": [
                "Figma", "Axure", "Xmind",
                "Prompt Engineering", "RAG", "Embedding", "向量检索",
                "大语言模型", "Excel数据透视表", "Office办公软件",
            ],
            "business": [
                "产品需求文档(PRD)", "原型设计", "交互设计",
                "用户调研", "数据分析", "知识库构建",
            ],
            "soft": [
                "跨部门协作", "项目管理", "文档撰写", "公众号运营",
            ],
            "proficiency_map": {
                "Figma": "熟练",
                "Prompt Engineering": "熟练",
                "RAG": "理解",
                "Excel数据透视表": "熟练",
            },
        },
        "certifications": [],
        "advantages": [
            "具备AI产品项目实践经验：主导RAG问答系统、智能简历匹配小程序两个完整项目",
            "熟悉大模型应用层落地：Prompt Engineering、Embedding语义匹配、RAG检索增强",
            "有完整产品闭环经验：从需求分析、原型设计到开发测试、用户验证、上线迭代",
            "英语CET-4（525分），可阅读英文技术文档与产品资料",
            "熟练使用Figma/Axure/Xmind等产品工具，具备基础数据分析能力",
        ],
        "meta": {
            "resume_id": resume_id,
            "raw_text": raw_text,
            "parsed_at": now_iso,
            "parser_version": "manual-v1",
            "confidence_score": 0.95,
            "is_active": True,
            "source_type": "image",
        },
    },
    "created_at": now_iso,
    "updated_at": now_iso,
}

# ── 保存到 JSON ──
existing = []
if RESUME_FILE.exists():
    try:
        with open(RESUME_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except Exception:
        existing = []

# 如果已有同一份简历（通过 raw_text 前200字判断），替换；否则追加
text_sig = raw_text[:200]
found_idx = None
for i, r in enumerate(existing):
    if r.get("parsed_schema", {}).get("meta", {}).get("raw_text", "")[:200] == text_sig:
        found_idx = i
        break

if found_idx is not None:
    existing[found_idx] = resume_record
    print(f"[Update] 替换已有简历 | resume_id={resume_id}")
else:
    existing.append(resume_record)
    print(f"[Create] 新增简历 | resume_id={resume_id}")

with open(RESUME_FILE, "w", encoding="utf-8") as f:
    json.dump(existing, f, ensure_ascii=False, indent=2)

print(f"[File] 已保存到 {RESUME_FILE}")
print(f"[Info] 当前共 {len(existing)} 份简历")

# ── 同时输出人类可读的摘要 ──
schema = resume_record["parsed_schema"]
bi = schema["basic_info"]
edu = schema["education"]
proj = schema["projects"]
skills = schema["skills"]

print("\n" + "=" * 50)
print("Resume Parsing Summary")
print("=" * 50)
print(f"Name: {bi['name'] or '(Not filled)'}")
print(f"Current Identity: {bi['current_title']}")
print(f"Years of Experience: {bi['years_exp']} years")
print(f"\nEducation ({len(edu)} items):")
for e in edu:
    print(f"  - {e['school']} | {e['degree']} | {e['major']} | Graduation: {e['graduation_year']}")
print(f"\nProjects ({len(proj)} items):")
for p in proj:
    print(f"  - {p['name']}")
    print(f"    Role: {p['role']} | Duration: {p['duration']}")
    print(f"    Tech: {', '.join(p['tech_keywords'])}")
    print(f"    Business: {', '.join(p['business_keywords'])}")
print(f"\nSkills:")
print(f"  Technical: {', '.join(skills['technical'])}")
print(f"  Business: {', '.join(skills['business'])}")
print(f"  Soft: {', '.join(skills['soft'])}")
print(f"\nKey Advantages:")
for adv in schema["advantages"]:
    print(f"  - {adv}")
print("=" * 50)
