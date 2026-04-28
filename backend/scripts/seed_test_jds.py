"""
种子脚本：向知识库插入5条测试JD（带 structured_summary，用于双层召回测试）。

用法：
    cd backend && python scripts/seed_test_jds.py
"""

import asyncio
import json
import os
import sys
import uuid
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.chunking import chunk_semantic
from app.core.embedding import EmbeddingClient
from app.core.vector_store import VectorStore

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
JD_DATA_FILE = os.path.join(DATA_DIR, "jds.json")

# 5条测试JD
TEST_JDS = [
    {
        "company": "字节跳动",
        "title": "AI产品经理",
        "position": "AI产品经理",
        "description": "负责AI平台产品设计，定义Agent框架，重塑LUI交互。",
        "location": "北京",
        "salary": "30k-60k",
        "salary_range": "30k-60k",
        "sections": {
            "responsibilities": "1. 定义Agent框架：设计Planning、Reflection、Memory策略。2. 重塑LUI交互：探索意图驱动的流式UI与动态组件生成。3. 编排超级工作流：深入客服、数据、ERP等实战场景。4. 模型精细化调教：编写结构化Prompt，配合RAG、Function Calling。5. 数据驱动进化：建立交互好感度与任务成功率双重指标。",
            "hard_requirements": [
                "本科及以上学历，兼具C端AIGC敏锐度与B端SaaS/RPA逻辑功底",
                "熟练掌握Transformer、Token、RAG等核心概念",
                "深度使用过LangChain、Dify、Coze或LangGraph等框架",
                "具备极强的逻辑拆解力，能像指挥交响乐一样控制模型输出",
                "24小时高频冲浪，对新技术有雷达般的嗅觉",
            ],
            "soft_requirements": [
                "有LLM/Agent落地案例者优先",
            ],
        },
        "keywords": ["Python", "PyTorch", "LangChain", "RAG", "Agent", "产品经理", "AI"],
        "structured_summary": {
            "min_years": 3,
            "max_years": 5,
            "min_education": "本科",
            "category": "产品",
            "domain": "AI",
        },
    },
    {
        "company": "百度",
        "title": "大模型应用PM",
        "position": "大模型应用PM",
        "description": "探索文心一言在垂直行业的落地场景。",
        "location": "上海",
        "salary": "25k-50k",
        "salary_range": "25k-50k",
        "sections": {
            "responsibilities": "1. 负责大模型应用的产品规划与设计。2. 深入理解行业需求，将业务经验抽象为大模型可执行的逻辑流与SOP。3. 模型精细化调教：编写结构化Prompt，配合RAG、Function Calling。4. 数据驱动进化：建立交互好感度与任务成功率双重指标。",
            "hard_requirements": [
                "硕士及以上学历，计算机、人工智能、数据科学或相关专业",
                "5年以上AI产品经验，有大模型落地案例",
                "深度理解Transformer、RAG、Prompt Engineering",
                "具备成本评估意识与良好的合规风险判断能力",
            ],
            "soft_requirements": [
                "有搜索/推荐/内容分发领域实战经验者优先",
            ],
        },
        "keywords": ["大模型", "LLM", "RAG", "Prompt", "产品经理", "AI", "NLP"],
        "structured_summary": {
            "min_years": 5,
            "max_years": None,
            "min_education": "硕士",
            "category": "产品",
            "domain": "AI",
        },
    },
    {
        "company": "阿里巴巴",
        "title": "后端开发",
        "position": "后端开发",
        "description": "负责电商核心系统开发。",
        "location": "杭州",
        "salary": "35k-65k",
        "salary_range": "35k-65k",
        "sections": {
            "responsibilities": "1. 负责电商核心系统的设计与开发。2. 高并发系统架构设计与性能优化。3. 分布式系统开发与微服务治理。4. 数据库设计与优化。",
            "hard_requirements": [
                "3年以上Java开发经验",
                "精通Spring Boot、MySQL、Redis、Kafka",
                "熟悉分布式系统设计与微服务架构",
                "本科及以上学历",
            ],
            "soft_requirements": [
                "有电商行业经验优先",
                "熟悉K8s、Docker优先",
            ],
        },
        "keywords": ["Java", "Spring Boot", "MySQL", "Redis", "Kafka", "微服务", "电商"],
        "structured_summary": {
            "min_years": 3,
            "max_years": None,
            "min_education": "本科",
            "category": "技术",
            "domain": "电商",
        },
    },
    {
        "company": "美团",
        "title": "搜索推荐AI PM",
        "position": "搜索推荐AI PM",
        "description": "基于多模态理解提升搜索意图识别准确率。",
        "location": "北京",
        "salary": "35k-65k",
        "salary_range": "35k-65k",
        "sections": {
            "responsibilities": "1. 负责搜索推荐全链路的产品演进。2. 利用Agent工作流、意图编程与自动化评估技术，重构从海量供给到亿级用户决策的完整分发链路。3. 智能诊断与进化体系：主导生成式策略的Benchmark建设。4. 全链路用户体验重塑：深入洞察用户在搜索推荐场景中的行为路径。",
            "hard_requirements": [
                "本科及以上学历，计算机、人工智能、数据科学等相关专业",
                "3-5年搜索/推荐/内容分发领域实战经验",
                "熟悉大模型基础原理，理解RAG、Prompt Engineering或Agent工作流",
                "具备AI原生交互天赋，能将复杂业务需求精准转化为Agent可执行的任务逻辑",
            ],
            "soft_requirements": [
                "有搜索/推荐/内容分发领域实战经验者优先",
            ],
        },
        "keywords": ["搜索推荐", "AI", "Agent", "Prompt", "RAG", "多模态", "产品经理"],
        "structured_summary": {
            "min_years": 3,
            "max_years": 5,
            "min_education": "本科",
            "category": "产品",
            "domain": "AI",
        },
    },
    {
        "company": "某小公司",
        "title": "Java后端",
        "position": "Java后端",
        "description": "负责传统企业内部系统开发。",
        "location": "成都",
        "salary": "10k-20k",
        "salary_range": "10k-20k",
        "sections": {
            "responsibilities": "1. 负责传统企业内部管理系统的设计与开发。2. 数据库CRUD操作与报表开发。3. 系统日常维护与bug修复。",
            "hard_requirements": [
                "1年以上Java开发经验",
                "熟悉Spring Boot、MySQL",
                "大专及以上学历",
            ],
            "soft_requirements": [
                "有ERP系统开发经验优先",
            ],
        },
        "keywords": ["Java", "Spring Boot", "MySQL", "ERP", "后端"],
        "structured_summary": {
            "min_years": 1,
            "max_years": 3,
            "min_education": "大专",
            "category": "技术",
            "domain": "传统",
        },
    },
]


def load_jds() -> list:
    if not os.path.exists(JD_DATA_FILE):
        return []
    try:
        with open(JD_DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] 读取 jds.json 失败: {e}，将创建新文件")
        return []


def save_jds(jds: list):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(JD_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(jds, f, ensure_ascii=False, indent=2, default=str)


async def seed():
    jds = load_jds()
    existing_ids = {j.get("jd_id") for j in jds}
    existing_companies = {j.get("company") for j in jds}

    # 向量库初始化
    vs = VectorStore()
    vs.embedding_client = EmbeddingClient.from_config()

    added = 0
    for i, data in enumerate(TEST_JDS, start=1):
        # 去重：如果同公司同岗位已存在，跳过
        dup = [j for j in jds if j.get("company") == data["company"] and j.get("position") == data["position"]]
        if dup:
            print(f"[SKIP] {data['company']} · {data['position']} 已存在")
            continue

        new_id = max([j["id"] for j in jds], default=0) + 1
        jd_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        jd_record = {
            "id": new_id,
            "jd_id": jd_id,
            "company": data["company"],
            "title": data["title"],
            "position": data["position"],
            "description": data["description"],
            "location": data["location"],
            "salary": data["salary"],
            "salary_range": data["salary_range"],
            "color": "bg-gray-100 text-gray-600",
            "sections": data["sections"],
            "keywords": data["keywords"],
            "structured_summary": data["structured_summary"],
            "raw_text": json.dumps(data["sections"], ensure_ascii=False),
            "meta": {
                "source_type": "seed",
                "created_at": now,
                "updated_at": now,
                "chunk_ids": [],
            },
            "created_at": datetime.now(),
            "vector_indexed": False,
        }

        # 向量库入库
        try:
            chunk_ids = await vs.add_jd(jd_record, strategy="semantic")
            jd_record["meta"]["chunk_ids"] = chunk_ids
            jd_record["vector_indexed"] = len(chunk_ids) > 0
            print(f"[VECTOR] {data['company']} · {data['position']} → {len(chunk_ids)} chunks")
        except Exception as e:
            print(f"[WARN] {data['company']} 向量入库失败: {e}")

        jds.insert(0, jd_record)
        added += 1
        print(f"[ADD] id={new_id} | {data['company']} · {data['position']} | "
              f"年限={data['structured_summary']['min_years']}-{data['structured_summary']['max_years']} | "
              f"学历={data['structured_summary']['min_education']} | "
              f"领域={data['structured_summary']['domain']}")

    save_jds(jds)
    print(f"\n完成：新增 {added} 条JD，知识库共 {len(jds)} 条")


if __name__ == "__main__":
    asyncio.run(seed())
