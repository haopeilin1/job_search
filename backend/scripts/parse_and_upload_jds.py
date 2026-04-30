"""
批量解析用户粘贴的 JD 文本，追加到 knowledge base 的 jds.json 中。

用法：
    python backend/scripts/parse_and_upload_jds.py

（将原始 JD 文本写入 backend/scripts/raw_jds_input.txt 后运行）
"""

import json
import re
import uuid
from datetime import datetime
from pathlib import Path

# ── 配置 ──
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "jds.json"
RAW_INPUT = Path(__file__).resolve().parent / "raw_jds_input.txt"

# 用户这次粘贴的原始文本（直接内嵌，避免手动创建文件）
RAW_JDS_TEXT = r'''
字节跳动AI产品实习生（代码方向）-抖音AI
北京 | 实习 | 产品-产品经理 | ByteIntern | 职位ID: A08585A
职位描述
ByteIntern：面向2027届毕业生（2026年9月-2027年8月期间毕业），为符合岗位要求的同学提供转正机会。
团队介绍：抖音AI团队主要负责抖音项目以及团队内的AI落地相关基础建设工作。我们的职责是用前沿的AI技术去赋能抖音应用以及抖音团队。我们希望在AI技术的加持下，能够更好的给抖音用户带来更好的信息消费体验，也能够让抖音这个大型组织能够更加高效的运转，从而更加及时的满足用户的各项诉求。团队主要负责抖音AI基础建设，包括但不限于模型训练、Agent相关的工程链路开发、通用Agent工具抽象以及AI Native的各类产品开发工作。AGI信仰强，以发展成为AI Native组织为目标，努力拓展传统协作模式的职责边界，充分给同学们提供自身探索的空间。
1、构建代码大模型与Agent在抖音前端、服务端、客户端软件工程开发真实任务场景下的效果评估体系，定义效果评测集、评估标准、指标体系，产出评测结论与归因分析，推动效果迭代，探索代码Agent在真实业务场景中的效果上限；
2、协同研发、测试、数据科学等团队，基于产品定位、用户反馈、离线评测、AB实验和日志观测等方式，识别效果缺陷与改进机会点，给出可执行的优化方向与评估策略；
3、跟进工业界和学术界的最新评测研究与方法论，结合真实研发需求场景迭代评测方案，探索更反映真实开发者用户体验和价值的评测方法。
职位要求
1、2027届本科及以上学历在读，计算机、软件、通信、软件工程、电子信息工程、人工智能、数据科学、统计学、数学等相关专业优先；
2、有前端、服务端、客户端研发实习经验及AI评测产品实习经验者优先；
3、具备结构化分析与复杂问题拆解的能力，对数据和指标体系敏感，能基于实验、日志、线上表现快速定位模型问题，并熟练制定合理的验证方案与评测集；
4、对AI技术发展有兴趣或基础理解，跟进最新研究论文、直接体验产品，并将其转化为评估与产品改进思路；
5、具备较强的好奇心、自驱力和结果导向意识，主动承担有挑战性的任务，推动复杂问题落地。

字节跳动**AI产品实习生（商家经营方向）-抖音生活服务**

上海 | 实习 | 产品 | ByteIntern | 职位ID: A150961

**职位描述**

ByteIntern：面向2027届毕业生（2026年9月-2027年8月期间毕业），为符合岗位要求的同学提供转正机会。

团队介绍：抖音生活服务依托抖音POI（Point of Interest）、短视频、直播、搜索等，以"激发线下生意，丰富美好生活"为使命，致力于成为最值得信赖的生活服务平台，为用户提供更丰富、独特的生活体验，为合作伙伴提供包容且公平的健康营商生态。截至2025年底，抖音生活服务业务已覆盖400+城市，超过1500万家动销门店。抖音生活服务将继续携手生态伙伴，为用户提供值得心动的商品和服务。

1、AI Agent设计与行业落地，能够参与并深入调研生活服务（餐饮、游玩、服务零售等）细分行业，理解商家入驻、经营过程中的需求痛点，设计帮助商家更好经营的Agent；

2、深度参与Agent迭代，包括但不限于：业务场景挖掘、Prompt编写与调优、模型微调策略、Tools/Skills构建、记忆库构建等，以此提升模型回答的准确性与业务转化效果；

3、业务场景挖掘，负责分析商家行为和经营数据，挖掘重点场景并推动优化模型策略或业务流程；

4、模型测评，定义可量化的测评标准，构建Agent回复的评测集，并参与端到端的模型测评，对回答质量进行量化分析并输出优化建议；

5、负责监控AI沟通效果，参与建设数据监控看板，确保业务过程数据可观测。

**职位要求**

1、2027届本科及以上学历在读，计算机、统计、数学或相关专业优先；

2、具备良好的逻辑推导能力与文字表达能力，能够产出高质量的需求文档，熟练使用SQL及Excel进行数据处理，具备基础的数据敏感度；

3、熟悉大语言模型（LLM）、Agent基本原理，对Prompt Engineering有实际操作经验，了解如何通过结构化指令优化模型输出；

4、了解模型评测逻辑，能够针对业务场景设计合理的评测规则和优化建议；

5、有使用相关AI工具/框架经验者优先；

6、具备优秀的自驱力和跨团队协同的能力，全勤实习4个月以上。

字节跳动**AI产品实习生（平台产品方向）-抖音生活服务**

北京 | 实习 | 产品 | ByteIntern | 职位ID: A57419

**职位描述**

ByteIntern：面向2027届毕业生（2026年9月-2027年8月期间毕业），为符合岗位要求的同学提供转正机会。

团队介绍：抖音生活服务依托抖音POI（Point of Interest）、短视频、直播、搜索等，以"激发线下生意，丰富美好生活"为使命，致力于成为最值得信赖的生活服务平台，为用户提供更丰富、独特的生活体验，为合作伙伴提供包容且公平的健康营商生态。截至2025年底，抖音生活服务业务已覆盖400+城市，超过1500万家动销门店。抖音生活服务将继续携手生态伙伴，为用户提供值得心动的商品和服务。

1、负责AI营销、AIGC、AI销售等方向的产品功能定义、需求调研与流程梳理等工作；

2、明确模型应用能力的定义、标准，构建合理的评估与反馈体系，推动各项关键能力的建设与提升；

3、协调跨多职能团队合作，协同研发、模型团队提升场景可用性，保证项目高质量的实现和交付。

**职位要求**

1、2027届本科及以上学历在读；

2、学习能力优秀，逻辑思维清晰，有较好业务场景理解能力，对大模型以及AI应用领域有较大热情，对大模型的特性与动态有跟踪和理解；

3、具备较好的协调沟通能力，有较好的复杂项目推进经验；

4、在AI营销领域有模型应用相关项目实践者加分；

5、每周出勤5天，实习时长6个月者优先。

字节跳动**AI产品实习生-TikTok Shop**

上海 | 实习 | 产品-产品经理 | ByteIntern | 职位ID: A239954

**职位描述**

ByteIntern：面向2027届毕业生（2026年9月-2027年8月期间毕业），为符合岗位要求的同学提供转正机会。

团队介绍：TikTok Shop 是 TikTok 旗下的内容电商。平台汇聚全球优质商家与创作者，通过短视频、直播等多场景连接消费者，让新奇好物畅销全球，让美好生活触手可得。目前团队分布在美国、英国、法国、印尼、墨西哥、中国等全球多个国家和地区，在这里你将有机会深入国际场景，面向全世界商家及用户，和跨区域团队协作，共同探索创新购物模式。期待和优秀的你一起创造更多可能！

1、全面参与文本、图片、语音大模型的落地，支持多场景达成业务目标；与算法、工程、QA、标注团队合作，定义算法任务，优化模型效果，组织算法评测；

2、参与A/B实验，深入分析实验结果，结合模型评测、用户调研等各手段，不断迭代算法策略；

3、参与负责AI评测工具建设，调研用户实际痛点，产出产品方案，组织产品上线和运营推广；

4、挖掘和探索电商领域AIGC能力的新场景，追踪行业最新动态，调研各类产品并产出报告。

**职位要求**

1、2027届本科及以上学历在读，计算机科学、软件工程、人工智能等相关专业优先；

2、对AI有热情，对大模型行业有较好理解，深度体验或使用过大模型产品者优先；

3、能够使用SQL、Python等进行数据分析和工作流搭建者优先；

4、学习能力、沟通和协作能力、项目管理能力，能够确保项目按时按质完成；

5、至少每周到岗4天及以上，连续实习6个月及以上，能长期实习者优先。

字节跳动**AI产品实习生（搜索方向）-Data AML**

北京 | 实习 | 产品 | ByteIntern | 职位ID: A112201A

**职位描述**

ByteIntern：面向2027届毕业生（2026年9月-2027年8月期间毕业），为符合岗位要求的同学提供转正机会。

团队介绍：Data AML是字节跳动的机器学习中台，为抖音/今日头条/西瓜视频等业务提供推荐/广告/CV/语音/NLP的训练和推理系统。为公司内业务部门提供强大的机器学习算力，并在这些业务的问题上研究一些具有通用性和创新性的算法。同时，也通过火山引擎将一些机器学习/推荐系统的核心能力提供给外部企业客户。

1、参与Connector Marketplace的管理后台和用户体验设计，关注Connector的发现、配置、授权、调试全流程的易用性，产出交互方案和体验优化建议，推动迭代；

2、负责按优先级推进企业数据源的对接落地，具体包括：理解目标平台的API能力与限制，按照Agent框架实现标准化的Connector的输入输出定义，协调研发完成开发、联调与上线，协同策略PM定义数据源知识利用场景的效果要求，同时覆盖中国和国际市场两条线的数据源；

3、持续跟踪企业SaaS生态和AI Agent工具链的发展动态（MCP/A2A协议演进、产品生态建设、新兴数据源平台），输出调研结论，为数据源接入路线图的优先级排序提供数据支撑。

**职位要求**

1、2027届本科及以上学历在读，计算机科学、人工智能、软件工程、电子信息、通信工程等相关专业优先；

2、日常使用AI编程工具，能独立完成中等复杂度的开发任务，熟悉至少一种主流编程语言（Python/TypeScript优先），了解RESTful API设计基础，能读懂OpenAPI/Swagger文档；

3、具备良好的产品认知，能从用户视角审视管理员配置体验和终端用户使用体验，具备中英文工作能力，能阅读英文技术文档和API reference，善于结构化整理信息，能输出清晰的对接文档和进度看板。

加分项

1、有AI产品（搜索、RAG、Agent）相关的产品或研发经历；

2、有开发经验，或为AI Agent写过自定义Tool/Plugin；

3、了解OAuth2.0/OAuth2.1授权流程，有AI产品第三方平台对接经验。

字节跳动**AI产品（AI coding方向）实习生-飞书**

北京 | 实习 | 产品 | ByteIntern | 职位ID: A222130B

**职位描述**

ByteIntern：面向2027届毕业生（2026年9月-2027年8月期间毕业），为符合岗位要求的同学提供转正机会。

团队介绍：飞书是AI时代先进生产力平台，提供一站式工作协同、组织管理、业务提效工具和深入企业场景的AI能力，助力企业能增长，有巧降。从互联网、高科技、消费零售，到制造、金融、医疗健康等，各行各业先进企业都在选择飞书，与飞书共创行业最佳实践。先进团队，先用飞书。

1、参与设计和开发先进的AI Coding应用搭建平台，为企业应用搭建提效，提升开发体验；

2、调研、抽象用户需求，分析数据，挖掘洞察，撰写PRD，协调项目组实现设计并不断迭代；

3、主导、参与AI应用场景的研究、规划，将需求转化为高品质的产品设计或解决方案；

4、多维度评估需求产品效果质量，不断关注用户反馈，行业进展。

**职位要求**

1、2027届本科及以上学历在读，人工智能、软件工程、计算机科学等相关专业优先；

2、对企业生产力工具领域的需求和场景有深入理解，能够将技术和业务需求相结合，制定合适的产品策略；

3、具备优秀的项目管理和团队协作能力，良好的沟通和表达能力，能够激励和指导团队成员，共同实现产品目标；

4、具备数据分析和数据驱动决策的能力，对产品的用户行为、性能和满意度等关键指标进行跟踪和分析，以指导产品优化；

5、关注用户体验，能够站在用户的角度思考问题，确保产品的易用性和体验满意度。

美团**【转正实习】AI产品经理**

实习·校园招聘

工作地点：北京市、上海市

最多可投递3个职位，但同一时间仅有1个职位流程处于进行中，第一志愿将被优先考虑。

**岗位职责**

我们期望用AI让世界变更好，推动人类发展进步！

在这里，你将会和AI核心团队共同参与兼具发展前景与挑战性的项目，由专家直接进行带教，自己写Demo、快速做实验，基于丰富的业务落地场景和海量的用户，想你所想，做你想做！

**任职要求**

【我们欢迎什么样的你？】

需同时符合以下三个条件：

1.秉持AI改善世界的初心：希望用AI让世界变更好，并因此愿意主动学习、实践AI知识；

2.软性素质和能力：具备极强的好奇心、学习能力、逻辑思考能力和动手能力；

3.对AI产品有足够的理解和实践经验：

1）优秀AI产品的重度用户：日常使用的模型为常用AI模型；

2）理解基础理论：习惯阅读和学习与自己工作相关的AI论文、技术报告，并把它用在工作改进中；

3）具备AI编程能力：是AI编程工具的深度用户，并实际开发过Demo、小工具，调过头部模型的API；

4）有AI产品的工业实践：参与过一款有实际用户的AI产品的设计或开发过程。

【加入我们可以获得什么？】

1.前沿的业务方向：100%精力投入AI，兼具发展前景与挑战性，具有丰富的业务落地场景和海量的用户；

2.与懂AI的人同行：AI领域专家直接带教，团队人才云集，日常在用优秀的模型/工具，成长迅速；

3.产研合作效率高：不写复杂的PRD，自己可以写Demo，快速做实验，并支持去生产环境做实验。

MiniMax**大模型产品实习生-B端业务**

上海、北京 | 校招 | 实习 | 产品/策划/项目 - 产品经理 | 日常实习

**职位描述**

1、参与公司B端业务的产品规划和设计工作，结合业务场景及客户需求推进模型优化；

2、与算法密切合作，并能基于业务充分发挥AI能力；

3、与研发和业务团队紧密配合，进行高质量迭代。

**职位要求**

1、本科以上学历，熟悉人工智能各方向技术，并有在各场景上应用的知识储备，具备AIGC相关产品背景优先；

2、具备丰富的实践经历，有产品化思维，能拆解客户需求并定义问题，完成业务-技术的语言转；

3、对AIGC赛道有热情，理解模型能力边界、优化方向。

MiniMax**大模型产品经理-实习-Top Talent**

北京、上海 | 校招 | 实习 | 产品/策划/项目 - 产品经理 | 2027届实习生招聘

**职位描述**

我们正在寻找对大模型产品充满好奇心的未来领袖。根据你的天赋与兴趣，你将深度参与以下一个或多个方向的工作：

1、驱动核心产品规划：参与甚至主导我们下一代大模型产品，将前沿技术转化为具有巨大用户价值的产品；

2、定义用户体验与视觉语言：主导或深度参与AIGC产品（图文/视频/音频生成等）的交互与视觉设计，确保其国际一线的体验标准。

3、负责增长与传播：基于对用户与数据的洞察，或你对内容与社区的敏感度，驱动产品的迭代、增长与市场传播。

4、洞察全球市场与用户：为业务注入国际化视野，助力我们的产品在全球领先。

**职位要求**

我们相信，才华与热爱远比经验更重要。我们期待你具备以下任一特质：

1、卓越的技术理解力，具备良好的数据结构和算法基础，熟练掌握Python/SQL等工具者优先；

2、出色的产品思维，对AI产品的交互设计、能力边界等问题有自己独到的思考；

3、对美有极致的追求，对流行有天然的嗅觉。你的作品集或你的社交平台内容，能证明你的审美与网感；

4、真正的国际化视野，能快速捕捉全球信息，理解不同市场的用户与文化并转化成商业价值。

**AI产品经理**

深圳市 | 产品类 | 实习生 (2027届寻梦实习招聘) | 2026-03-06 发布

OPPO尊重并保护第三方保密信息，请勿非法披露您前雇主（及其客户）或他人的保密信息。

**岗位职责**

在这里，你将获得以AI技术为原点的，为亿级海外手机用户做产品规划和策划机会，你将对这些用户的生活方式带来改变，你将接触行业最先进最顶尖的AI技术，你将拥有最简单轻松的工作关系。你需要洞察AI行业发展趋势，掌握行业最新技术，深入挖掘用户生活方式和场景，分析用户潜在需求和痛点，打造高用户价值的手机AI产品。

方向一：从事AI产品、互联网产品的策划工作，为用户及开发者提供优秀的AI产品体验；

方向二：分析行业发展趋势，洞察用户及开发者的痛点和需求，在此基础上策划和输出产品方案，并跟进实现；

方向三：分析产品上线后的数据和用户反馈，持续迭代和优化产品；

方向四：主导AI产品的研发，与多种AI技术的研发、手机整机、ColorOS等业务部门紧密合作，形成可落地产品。

**任职要求**

1.有一定的AI相关知识学习基础，语言类&视觉类AI相关经验均可，计算机、软件工程、电子信息、通信工程等相关专业优先；

2.具综合素质扎实，具有优秀的学习能力、行业洞察分析能力、逻辑分析能力和文字组织表达能力；

3.对产品经理的工作有一定的了解并感兴趣，对产品有独到见解和创新的想法；

4.在校期间，有海外产品相关实习经验，英语沟通表达流畅，有海外留学经验者优先。

5.懂AI功能设计，理解AI能力的边界和可能性，能设计出以AI为核心功能或增强功能的产品/方案。了解行业最新AI产品、技术发展，了解AI的最佳实践。

360   AI产品实习生 (北京)-4872(J12150)
实习生招聘 实习北京市
2026-03-03发布
工作职责
1.AI产品调研与竞品分析:跟踪国内外LLM/RAG/Agent等技术动态，整理市场与用户痛点。
2.汇报材料与文档撰写:协助撰写PRD、方案PPT、Roadmap等，支持内外部评审与汇报。
3.资源协调与进度跟进:与算法、前端、后端、设计、测试团队对接，更新任务看板(Jira飞书等)，推动里程碑按期完成。
4.数据与结果输出:产出竞品分析报告、需求洞察清单、项目周报/Sprint回顾等可交付成果。
任职资格
1.教育背景:本科或研究生在读，计算机、人工智能、软件工程、信息管理等相关专业优先。2.AI基础与兴趣:对生成式AI、机器学习或NLP有浓厚兴趣，了解大模型及RAG基本原理3.分析与写作能力:逻辑清晰，擅长信息检索、结构化笔记与PPT可视化呈现。4.协作与沟通:积极主动，善于跨团队沟通，熟练使用协作工具。
5.时间投入:每周至少4天，连续实习6个月及以上，支持远程。
6.语言能力:具备良好的中英文阅读与书面表达能力。
7.加分项:熟练应用AI工具，持有360人工智能智能体工程师认证及相关A证书优先;参与过 A1相关竞赛或开源项目;具备PytouJavaScript基础;熟悉PromptEngineering或LLM微调流程;对SaaS或企业级软件有研究或实习经历。

滴滴：
27届秋储-AI产品实习生(AIoT产品事业部)
实习|产品类|CTO线|北京市
发布于2026-04-02
分享
申请职位
职位描述
岗位职责:1、协助负责跟踪产品核心指标，关注指标异常，分析原因产出认知;负责完善产品的指标体系，产出数据需求并推动上线;2、参与目标达成所需的数据处理和策略分析，产出具体可落地的产品需求，负责需求管理，与各端研发、算法等团队沟通保障需求按期上线;3、参与具体case分析，协助看清影响产品效果的问题，形成产品解决方案并推动优化;4、负责行业调研、产品体验分析等，产出分析报告。
任职要求:
1、2027届在校生，硕士及以上学历，理工科专业背景优先;2、熟悉SQL、python、机器学习等数据技能优先，有大数据分析或数据处理相关经验者优先，有算法、物联网相关经验者优先;3、良好的思维能力和快速学习能力，能准确理解复杂的业务逻辑和抽象的系统逻辑;
4、良好的沟通协调能力，有效进行业务需求沟通、开发沟通、推动项目落地;
5、责任心强，对数据的准确性、及时性负责。
'''


# ── 工具函数 ──

def _now_iso() -> str:
    return datetime.now().isoformat()


def _extract_location(text: str) -> str:
    """从JD文本中提取地点"""
    # 常见模式
    patterns = [
        r'^(北京|上海|深圳|杭州|广州|成都|武汉|西安|南京|苏州)[\s|、，,\\/]',
        r'工作地点[：:]\s*(.+?)(?:\n|$)',
        r'工作地点[：:]\s*(.+?)(?:\s|$)',
        r'([\u4e00-\u9fa5]{2,5})\s*\|\s*实习',
        r'([\u4e00-\u9fa5]{2,5})\s*\|\s*(?:校招|日常)',
        r'([\u4e00-\u9fa5]{2,5})\s*\|\s*产品',
        r'实习生招聘\s+实习\s*([\u4e00-\u9fa5]{2,5})',
    ]
    for p in patterns:
        m = re.search(p, text, re.MULTILINE)
        if m:
            loc = m.group(1).strip()
            if loc in ('北京', '上海', '深圳', '杭州', '广州', '成都', '武汉', '西安', '南京', '苏州', '北京市', '上海市', '深圳市', '杭州市', '广州市'):
                return loc.replace('市', '')
            return loc
    # 全文搜索城市
    cities = ['北京', '上海', '深圳', '杭州', '广州', '成都', '武汉', '西安', '南京', '苏州']
    found = []
    for c in cities:
        if c in text:
            found.append(c)
    if found:
        return '/'.join(found)
    return None


def _extract_salary(text: str) -> str:
    """提取薪资"""
    m = re.search(r'(\d+)[kK]-(\d+)[kK]', text)
    if m:
        return f"{m.group(1)}k-{m.group(2)}k"
    if '薪资面议' in text:
        return '薪资面议'
    return None


def _split_hard_soft(requirements: list) -> tuple:
    """将要求列表拆分为 hard / soft"""
    hard, soft = [], []
    soft_markers = ['优先', '加分', '更佳', '欢迎', '有兴趣', '热爱', '热情', '好奇心']
    for r in requirements:
        r_clean = r.strip()
        if not r_clean:
            continue
        is_soft = any(m in r_clean for m in soft_markers)
        # 学历、年限等硬性条件归到 hard
        if re.search(r'(本科|硕士|博士|学历|在读|毕业生|届|年以上|经验)', r_clean):
            is_soft = False
        if is_soft:
            soft.append(r_clean)
        else:
            hard.append(r_clean)
    return hard, soft


def _extract_keywords(text: str, position: str, company: str) -> list:
    """提取关键词"""
    tech_kws = [
        'Python', 'SQL', 'Java', 'Go', 'C++', 'JavaScript', 'TypeScript',
        'React', 'Vue', 'Node.js', 'MySQL', 'Redis', 'MongoDB', 'Kafka',
        'Docker', 'Kubernetes', 'A/B实验', 'AB实验',
        'AI', '大模型', 'LLM', 'NLP', '机器学习', '深度学习', '算法',
        'Agent', 'RAG', 'Prompt', 'AIGC', '多模态', '智能体',
        '产品经理', '数据分析', '用户研究', '运营', '增长',
        'PRD', 'Figma', 'Sketch', 'Axure', '竞品分析',
        'MCP', 'A2A', 'OAuth', 'RESTful', 'Swagger',
        'SaaS', 'RPA', '低代码', '工作流', 'Workflow',
        '推荐', '搜索', '广告', 'CV', '语音', '图像',
        '飞书', '抖音', 'TikTok', '火山引擎', 'ByteIntern',
        '代码大模型', 'coding', '评测', '评估',
    ]
    kws = set()
    for kw in tech_kws:
        if kw.lower() in text.lower():
            kws.add(kw)
    # 业务方向关键词
    biz_kws = ['商家经营', '平台产品', '生活服务', '电商', '搜索', 'AI Coding', 'IoT', 'AIoT', 'B端', 'C端']
    for kw in biz_kws:
        if kw in text:
            kws.add(kw)
    # 公司名
    if company and company not in ('未知公司',):
        kws.add(company)
    # 岗位名中的核心词
    if position:
        for w in ['产品经理', '产品实习生', '实习生']:
            if w in position:
                kws.add(w)
    return sorted(list(kws))[:20]


def _clean_lines(lines: list) -> list:
    """清理行，去除 markdown 标记和空行"""
    cleaned = []
    for l in lines:
        l = l.strip()
        l = l.lstrip('*').lstrip('-').lstrip('•').strip()
        l = l.replace('**', '')
        if l and len(l) > 3:
            cleaned.append(l)
    return cleaned


# ── 主解析器 ──

def split_jds(raw: str) -> list:
    """将原始文本分割为单个JD"""
    # 支持两种格式：
    # 1. 旧格式：公司名在行首（字节跳动、美团等）
    # 2. 新格式：以 "公司：" 开头，用 "---" 分隔
    
    # 先按 "---" 分隔（新格式）
    if '---' in raw:
        parts = [p.strip() for p in raw.split('---') if p.strip()]
        return parts
    
    # 旧格式：按公司名分割
    company_pattern = r'(?:^|\n\s*\n)(?=字节跳动|美团|MiniMax|OPPO|360\s|滴滴[：:]|ACG\s)'
    parts = re.split(company_pattern, raw.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def parse_single_jd(text: str) -> dict:
    """解析单个JD为结构化Schema。支持两种格式：旧格式（公司名行首）和新格式（公司：XXX）。"""
    lines = text.split('\n')
    lines = [l.strip() for l in lines]
    
    # 判断格式：新格式以 "公司：" 开头
    is_new_format = lines[0].startswith('公司：') if lines else False
    
    company = "未知公司"
    title = "未知岗位"
    location = None
    
    if is_new_format:
        # ========== 新格式 ==========
        # 公司：XXX
        # 岗位：XXX
        # 地点：XXX
        for l in lines[:5]:
            if l.startswith('公司：'):
                company = l.replace('公司：', '').strip()
            elif l.startswith('岗位：'):
                title = l.replace('岗位：', '').strip()
            elif l.startswith('地点：'):
                location = l.replace('地点：', '').strip()
    else:
        # ========== 旧格式 ==========
        first_line = lines[0] if lines else ""
        if '字节跳动' in first_line:
            company = '字节跳动'
        elif '美团' in first_line:
            company = '美团'
        elif 'MiniMax' in first_line:
            company = 'MiniMax'
        elif 'OPPO' in first_line:
            company = 'OPPO'
        elif first_line.startswith('360'):
            company = '360'
        elif '滴滴' in first_line:
            company = '滴滴'
        elif 'ACG' in first_line:
            company = 'ACG'
        
        title_match = re.search(r'(?:字节跳动|美团|MiniMax|OPPO|360|滴滴|ACG)[：:\*\s]*(.+?)(?:\n|$)', first_line)
        if title_match:
            title = title_match.group(1).strip('*').strip()
        else:
            title = first_line.replace('字节跳动', '').replace('美团', '').replace('MiniMax', '').replace('OPPO', '').replace('360', '').replace('滴滴', '').replace('ACG', '').replace('：', '').replace('**', '').strip()
        title = re.sub(r'^[：:\*\s]+', '', title)
        if not title or len(title) < 3:
            for l in lines[1:4]:
                if '实习生' in l or '产品经理' in l or '产品' in l:
                    title = l.strip('*').strip()
                    break
        location = _extract_location(text)
    
    # 提取薪资
    salary_range = _extract_salary(text)
    
    # ========== 按编号提取职责、要求、加分项 ==========
    responsibilities = []
    hard_requirements = []
    soft_requirements = []
    
    # 段落标记
    desc_markers = ['工作职责：', '工作职责:', '职位描述', '岗位职责', '工作职责']
    req_markers = ['职责要求：', '职责要求:', '职位要求', '任职要求', '任职资格']
    plus_markers = ['加分项：', '加分项:', '加分项']
    
    current_section = None
    
    for l in lines:
        l_stripped = l.strip()
        if not l_stripped:
            continue
        
        # 检测段落切换
        if any(l_stripped.startswith(m) for m in desc_markers):
            current_section = 'desc'
            continue
        if any(l_stripped.startswith(m) for m in req_markers):
            current_section = 'req'
            continue
        if any(l_stripped.startswith(m) for m in plus_markers):
            current_section = 'plus'
            continue
        
        # 收集编号行
        if re.match(r'^\d+[\.、．]\s*', l_stripped):
            item = re.sub(r'^\d+[\.、．]\s*', '', l_stripped).strip()
            if current_section == 'desc':
                responsibilities.append(item)
            elif current_section == 'req':
                hard_requirements.append(item)
            elif current_section == 'plus':
                soft_requirements.append(item)
    
    # Fallback：如果没有编号行，按旧逻辑处理
    if not responsibilities and not hard_requirements:
        # 旧格式的段落提取
        in_desc = False
        in_req = False
        desc_lines = []
        req_lines = []
        
        for l in lines:
            l_clean = l.strip().replace('*', '').replace('**', '')
            if l_clean in desc_markers or l_clean.startswith('工作职责'):
                in_desc = True
                in_req = False
                continue
            if l_clean in req_markers or l_clean.startswith('职责要求'):
                in_desc = False
                in_req = True
                continue
            if l_clean in ('加分项', '加分', '加分项：', '加分：'):
                in_desc = False
                in_req = False
                continue
            if in_desc:
                desc_lines.append(l)
            elif in_req:
                req_lines.append(l)
        
        responsibilities = _clean_lines(desc_lines)
        requirements = _clean_lines(req_lines)
        hard_requirements, soft_requirements = _split_hard_soft(requirements)
        if not hard_requirements and requirements:
            hard_requirements = requirements
    
    # 合并职责为一段（用于 description 字段）
    responsibilities_text = '\n'.join(responsibilities)
    if not responsibilities_text:
        responsibilities_text = text[:500]
    
    # 关键词
    keywords = _extract_keywords(text, title, company)
    
    now = _now_iso()
    jd_id = str(uuid.uuid4())
    
    return {
        "jd_id": jd_id,
        "company": company,
        "position": title,
        "location": location,
        "salary_range": salary_range,
        "sections": {
            "responsibilities": responsibilities_text,
            "hard_requirements": hard_requirements,
            "soft_requirements": soft_requirements,
        },
        "keywords": keywords,
        "raw_text": text,
        "meta": {
            "source_type": "paste",
            "created_at": now,
            "updated_at": now,
            "chunk_ids": [],
        },
    }


def main():
    # 1. 加载现有数据
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        existing = json.load(f)

    max_id = max([j.get("id", 0) for j in existing], default=0)

    # 2. 读取 JD 文本（优先从文件，fallback 到变量）
    raw_text = RAW_JDS_TEXT
    if RAW_INPUT.exists():
        raw_text = RAW_INPUT.read_text(encoding="utf-8")
        print(f"[Parser] 从文件读取: {RAW_INPUT}")
    
    if not raw_text.strip():
        print("[Parser] 没有 JD 文本可处理")
        return

    # 3. 分割并解析
    jd_texts = split_jds(raw_text)
    print(f"[Parser] 识别到 {len(jd_texts)} 条 JD")

    new_records = []
    for idx, jd_text in enumerate(jd_texts, 1):
        parsed = parse_single_jd(jd_text)
        max_id += 1
        record = {
            "id": max_id,
            "jd_id": parsed["jd_id"],
            "company": parsed["company"],
            "title": parsed["position"],
            "position": parsed["position"],
            "description": parsed["sections"]["responsibilities"][:300] + "..." if len(parsed["sections"]["responsibilities"]) > 300 else parsed["sections"]["responsibilities"],
            "location": parsed["location"] or "未知",
            "salary": parsed["salary_range"] or "薪资面议",
            "salary_range": parsed["salary_range"] or "薪资面议",
            "color": "bg-gray-100 text-gray-600",
            "sections": parsed["sections"],
            "keywords": parsed["keywords"],
            "raw_text": parsed["raw_text"],
            "meta": parsed["meta"],
            "created_at": datetime.now().isoformat(),
            "vector_indexed": False,
        }
        new_records.append(record)
        print(f"  [{idx}] {record['company']} | {record['position']} | {record['location']} | keywords={len(record['keywords'])}")

    # 3. 追加并保存
    existing.extend(new_records)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2, default=lambda o: o.isoformat() if isinstance(o, datetime) else str(o))

    print(f"\n[Done] 已追加 {len(new_records)} 条 JD，当前库共 {len(existing)} 条")
    print(f"[File] {DATA_FILE}")


if __name__ == "__main__":
    main()
