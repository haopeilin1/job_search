"""
一键修复 jds.json 中已知的小瑕疵。
"""

import json
from datetime import datetime
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "jds.json"


def main():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 建立 id -> record 映射
    records = {r["id"]: r for r in data}

    # ── 1. id=30 美团：清理引言和福利段落 ──
    r30 = records[30]
    hard30 = r30["sections"]["hard_requirements"]
    # 删除引言/福利
    remove30 = {
        "需同时符合以下三个条件：",
        "【加入我们可以获得什么？】",
        "1.前沿的业务方向：100%精力投入AI，兼具发展前景与挑战性，具有丰富的业务落地场景和海量的用户；",
        "2.与懂AI的人同行：AI领域专家直接带教，团队人才云集，日常在用优秀的模型/工具，成长迅速；",
        "3.产研合作效率高：不写复杂的PRD，自己可以写Demo，快速做实验，并支持去生产环境做实验。",
    }
    r30["sections"]["hard_requirements"] = [h for h in hard30 if h not in remove30]
    soft30 = r30["sections"]["soft_requirements"]
    r30["sections"]["soft_requirements"] = [s for s in soft30 if s != "【我们欢迎什么样的你？】"]
    # 补充关键词
    r30["keywords"] = [
        "AI", "PRD", "产品经理", "美团", "转正实习",
        "AI编程", "Demo", "大模型", "工业实践",
        "好奇心", "学习能力", "逻辑思考"
    ]
    print(f"[Fix] id=30 美团 | hard={len(r30['sections']['hard_requirements'])} soft={len(r30['sections']['soft_requirements'])}")

    # ── 2. id=32 MiniMax：删除引言 ──
    r32 = records[32]
    hard32 = r32["sections"]["hard_requirements"]
    r32["sections"]["hard_requirements"] = [
        h for h in hard32
        if "我们相信，才华与热爱远比经验更重要" not in h
    ]
    print(f"[Fix] id=32 MiniMax | hard={len(r32['sections']['hard_requirements'])} soft={len(r32['sections']['soft_requirements'])}")

    # ── 3. id=34 360：拆分合并的要求 ──
    r34 = records[34]
    # 原来 soft[0] 是4条粘在一起的，拆出来
    merged = r34["sections"]["soft_requirements"][0] if r34["sections"]["soft_requirements"] else ""
    split_360 = []
    if merged:
        # 按 "数字." 或 "数字、" 拆分
        import re
        parts = re.split(r'(?=\d\.)', merged)
        for p in parts:
            p = p.strip()
            if p and len(p) > 5:
                # 清理末尾可能粘着的下一个要求的开始
                p = re.sub(r'(?=\d\.).*', '', p).strip()
                if p:
                    split_360.append(p)
    
    # 360 原始文本中 1-4 是基本要求（无优先字样），5-6 是硬性时间/语言，7 是加分项
    # 重新整理：1-6 放 hard，7 放 soft
    r34["sections"]["hard_requirements"] = [
        "1.教育背景:本科或研究生在读，计算机、人工智能、软件工程、信息管理等相关专业优先。",
        "2.AI基础与兴趣:对生成式AI、机器学习或NLP有浓厚兴趣，了解大模型及RAG基本原理",
        "3.分析与写作能力:逻辑清晰，擅长信息检索、结构化笔记与PPT可视化呈现。",
        "4.协作与沟通:积极主动，善于跨团队沟通，熟练使用协作工具。",
        "5.时间投入:每周至少4天，连续实习6个月及以上，支持远程。",
        "6.语言能力:具备良好的中英文阅读与书面表达能力。",
    ]
    r34["sections"]["soft_requirements"] = [
        "7.加分项:熟练应用AI工具，持有360人工智能智能体工程师认证及相关AI证书优先;参与过AI相关竞赛或开源项目;具备Python/JavaScript基础;熟悉PromptEngineering或LLM微调流程;对SaaS或企业级软件有研究或实习经历。"
    ]
    # 修正 title，去掉末尾编号
    r34["title"] = "AI产品实习生"
    r34["position"] = "AI产品实习生"
    print(f"[Fix] id=34 360 | hard={len(r34['sections']['hard_requirements'])} soft={len(r34['sections']['soft_requirements'])}")

    # ── 4. id=35 滴滴：清理 description 和 responsibilities ──
    r35 = records[35]
    # description 清理
    r35["description"] = "协助负责跟踪产品核心指标，完善指标体系；参与数据处理和策略分析，产出可落地的产品需求；参与case分析，推动优化；负责行业调研、产品体验分析等。"
    # responsibilities 截断掉 "任职要求:" 之后的内容
    resp = r35["sections"]["responsibilities"]
    if "任职要求:" in resp:
        resp = resp.split("任职要求:")[0].strip()
    r35["sections"]["responsibilities"] = resp
    # 拆分 hard/soft（原来 soft 是合并的3条+4条+5条）
    r35["sections"]["hard_requirements"] = [
        "4、良好的沟通协调能力，有效进行业务需求沟通、开发沟通、推动项目落地；",
        "5、责任心强，对数据的准确性、及时性负责。"
    ]
    r35["sections"]["soft_requirements"] = [
        "1、2027届在校生，硕士及以上学历，理工科专业背景优先；",
        "2、熟悉SQL、python、机器学习等数据技能优先，有大数据分析或数据处理相关经验者优先，有算法、物联网相关经验者优先；",
        "3、良好的思维能力和快速学习能力，能准确理解复杂的业务逻辑和抽象的系统逻辑；"
    ]
    print(f"[Fix] id=35 滴滴 | hard={len(r35['sections']['hard_requirements'])} soft={len(r35['sections']['soft_requirements'])}")

    # ── 5. id=33 OPPO：补充关键词 ──
    r33 = records[33]
    r33["keywords"] = [
        "AI", "OPPO", "产品经理", "ColorOS", "手机AI",
        "海外产品", "英语", "互联网产品", "AI功能设计",
        "视觉类AI", "语言类AI", "产品策划", "用户研究"
    ]
    print(f"[Fix] id=33 OPPO | keywords={len(r33['keywords'])}")

    # ── 6. id=24-29 字节跳动：微调 hard/soft 分配 ──
    # id=27 TikTok Shop：第4条"学习能力..."不含优先，应是 hard
    r27 = records[27]
    hard27 = [h for h in r27["sections"]["hard_requirements"] if "学习能力" not in h]
    soft27 = r27["sections"]["soft_requirements"]
    if "4、学习能力、沟通和协作能力、项目管理能力，能够确保项目按时按质完成；" not in hard27:
        hard27.append("4、学习能力、沟通和协作能力、项目管理能力，能够确保项目按时按质完成；")
    r27["sections"]["hard_requirements"] = hard27
    r27["sections"]["soft_requirements"] = [s for s in soft27 if "学习能力" not in s]
    print(f"[Fix] id=27 字节跳动-TikTok | hard={len(r27['sections']['hard_requirements'])} soft={len(r27['sections']['soft_requirements'])}")

    # id=28 Data AML：第3条"具备良好的产品认知..."不含优先，应是 hard
    r28 = records[28]
    hard28 = [h for h in r28["sections"]["hard_requirements"] if "产品认知" not in h]
    soft28 = r28["sections"]["soft_requirements"]
    if "3、具备良好的产品认知，能从用户视角审视管理员配置体验和终端用户使用体验，具备中英文工作能力，能阅读英文技术文档和API reference，善于结构化整理信息，能输出清晰的对接文档和进度看板。" not in hard28:
        hard28.append("3、具备良好的产品认知，能从用户视角审视管理员配置体验和终端用户使用体验，具备中英文工作能力，能阅读英文技术文档和API reference，善于结构化整理信息，能输出清晰的对接文档和进度看板。")
    r28["sections"]["hard_requirements"] = hard28
    r28["sections"]["soft_requirements"] = [s for s in soft28 if "产品认知" not in s]
    print(f"[Fix] id=28 字节跳动-Data AML | hard={len(r28['sections']['hard_requirements'])} soft={len(r28['sections']['soft_requirements'])}")

    # 保存
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=lambda o: o.isoformat() if isinstance(o, datetime) else str(o))

    print(f"\n[Done] 已保存到 {DATA_FILE}")


if __name__ == "__main__":
    main()
