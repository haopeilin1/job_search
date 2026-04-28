import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

from app.core.intent import RuleClassifier, IntentType

rc = RuleClassifier()

tests = [
    ("分析我的简历与字节的匹配度", [], IntentType.MATCH_SINGLE, "引用式匹配"),
    ("对比知识库中阿里与字节的要求", [], IntentType.RAG_QA, "RAG属性查询"),
    ("帮我看看适合哪家", [], IntentType.GLOBAL_MATCH, "全局对比"),
    ("岗位职责：\n1. 负责公司核心推荐算法的设计与优化，提升用户点击率与停留时长；\n2. 深入理解业务场景，将机器学习技术落地到搜索、推荐、广告等场景；\n3. 跟踪学术界与工业界最新进展，推动算法迭代与技术创新；\n4. 与产品、工程团队紧密协作，推动算法方案从 0 到 1 上线。\n\n硬性要求：\n1. 计算机、数学、统计等相关专业本科及以上学历，3 年以上算法经验；\n2. 扎实的编程能力，精通 Python/C++，熟悉 TensorFlow/PyTorch 等深度学习框架；\n3. 熟悉主流推荐算法（协同过滤、矩阵分解、深度神经网络等），有大规模推荐系统实战经验；\n4. 具备良好的数据敏感度，能够独立完成数据分析、特征工程、模型训练与效果评估。", [], IntentType.MATCH_SINGLE, "JD长文本"),
    ("你好", [], IntentType.GENERAL, "问候"),
    ("字节需要什么技能", [], IntentType.RAG_QA, "RAG属性查询"),
    ("阿里在招什么岗位", [], IntentType.RAG_QA, "RAG公司查询"),
    ("面试技巧有哪些", [], IntentType.GENERAL, "面试通用"),
]

passed = 0
for msg, atts, expected, desc in tests:
    intent, conf, rule = rc.classify(msg, atts)
    ok = intent == expected
    status = "PASS" if ok else "FAIL"
    intent_str = intent.value if intent else "None"
    print(f"[{status}] {desc}: '{msg[:30]}...' -> {intent_str} (rule={rule})")
    if ok:
        passed += 1

print(f"\nResult: {passed}/{len(tests)} passed")
print(f"Rule stats: {rc.registry.get_stats()}")
