# v2 ReAct Agent 测试集设计建议（单轮对话版）

> 多轮对话暂不做，所有用例均为单轮，但可通过 `eval_context.injected_history_slots` 模拟多轮上下文状态，测试 evidence_cache 复用和 resolved_refs。

---

## 设计原则

1. **工具全覆盖**：8 个工具每个至少被覆盖 2 次（kb_retrieve, external_search, match_analyze, global_rank, qa_synthesize, interview_gen, evidence_relevance_check, general_chat）
2. **意图全覆盖**：6 种意图每种至少 3 条（position_explore, match_assess, attribute_verify, interview_prepare, resume_manage, general_chat）
3. **流程模式全覆盖**：串行、并行、依赖链、条件分支、fallback、Replan 触发
4. **简历差异化**：3 种简历背景（AI算法/产品、Java后端、空简历）均匀分布
5. **一句话原则**：每条 message 独立完整，不依赖真实多轮历史

---

## 一、单轮对话单意图（~18 条）

### 1.1 岗位探索（position_explore）— 6 条

| # | 用户消息 | 简历 | gold_intents | expected_tools | 测试重点 |
|---|---------|------|-------------|----------------|---------|
| S01 | "最近AI产品岗多不多？帮我筛几个能投的" | 空简历 | explore | kb_retrieve, global_rank | 无简历宽泛探索，测试粗筛降序 |
| S02 | "我这种情况能投哪些公司啊？" | AI简历 | explore | kb_retrieve, global_rank | 口语化，测试意图稳定性 |
| S03 | "给我看看符合我背景的JD" | AI简历 | explore | kb_retrieve, global_rank | 反问句式，测试鲁棒性 |
| S04 | "帮我推几个Java后端的岗" | Java简历 | explore | kb_retrieve, global_rank | 简历与搜索方向一致，测试高匹配排序 |
| S05 | "Python相关的岗位有哪些" | AI简历 | explore | kb_retrieve, global_rank | 技能关键词匹配 |
| S06 | "随便看看有什么岗" | 空简历 | explore | kb_retrieve, global_rank | 无明确方向，测试hybrid_score兜底 |

### 1.2 匹配评估（match_assess）— 5 条

| # | 用户消息 | 简历 | gold_intents | expected_tools | 测试重点 |
|---|---------|------|-------------|----------------|---------|
| S07 | "字节那个AI产品经理，我够格不？" | AI简历 | assess | kb_retrieve, match_analyze | 公司+岗位明确，知识库命中 |
| S08 | "百度那个大模型PM要求太高了？你看看我能不能行" | AI简历 | assess | kb_retrieve, match_analyze | 口语化+反问，测试槽位提取 |
| S09 | "帮我分析一下蚂蚁集团的算法工程师" | AI简历 | assess | kb_retrieve, match_analyze | 公司名需标准化（蚂蚁→蚂蚁集团） |
| S10 | "这个岗位我能去吗？" | AI简历 | assess | kb_retrieve, match_analyze | **注入history_slots**: `{"position": "AI产品经理", "company": "字节跳动"}`，测试上下文引用解析 |
| S11 | "字节和百度的产品岗，哪个我更合适？" | AI简历 | assess | kb_retrieve, match_analyze | **多公司对比**，应触发多次检索或多实体assess |

### 1.3 属性核实（attribute_verify）— 3 条

| # | 用户消息 | 简历 | gold_intents | expected_tools | 测试重点 |
|---|---------|------|-------------|----------------|---------|
| S12 | "字节AI产品岗工资大概多少？" | AI简历 | verify | kb_retrieve, qa_synthesize | 薪资属性，RAG问答 |
| S13 | "百度那个大模型PM要不要求RAG经验？" | AI简历 | verify | kb_retrieve, qa_synthesize | 技能属性，测试布尔型问答 |
| S14 | "这个岗位加班多吗？" | AI简历 | verify | kb_retrieve, qa_synthesize | **注入history_slots**: `{"position": "AI产品经理", "company": "字节跳动"}`，测试上下文引用+属性推断 |

### 1.4 面试准备（interview_prepare）— 2 条

| # | 用户消息 | 简历 | gold_intents | expected_tools | 测试重点 |
|---|---------|------|-------------|----------------|---------|
| S15 | "面字节AI产品岗，我得准备点啥？" | AI简历 | prepare | kb_retrieve, interview_gen | 指定岗位+公司，生成针对性面试题 |
| S16 | "AI产品面试一般会问啥？给我模拟几个" | AI简历 | prepare | interview_gen | 泛化岗位，无公司名，测试泛化能力 |

### 1.5 简历管理（resume_manage）— 1 条

> 现有测试集完全没有覆盖 resume_manage 意图！

| # | 用户消息 | 简历 | gold_intents | expected_tools | 测试重点 |
|---|---------|------|-------------|----------------|---------|
| S17 | "帮我优化一下简历，突出我的项目经验" | AI简历 | resume_manage | file_ops, general_chat | 简历管理意图，测试文件操作+LLM改写 |

### 1.6 通用对话（general_chat）— 1 条

| # | 用户消息 | 简历 | gold_intents | expected_tools | 测试重点 |
|---|---------|------|-------------|----------------|---------|
| S18 | "你好，这个机器人能干嘛？" | 任意 | chat | general_chat | 问候语+功能介绍 |

---

## 二、单轮对话多意图（~10 条）

### 2.1 探索 + 评估（explore + assess）— 2 条

| # | 用户消息 | 简历 | gold_intents | expected_tools | 测试重点 |
|---|---------|------|-------------|----------------|---------|
| M01 | "先帮我挑几个岗，再重点看看字节那个AI产品我能不能去" | AI简历 | explore, assess | kb_retrieve, global_rank, match_analyze | **串行依赖**：assess依赖explore的top结果 |
| M02 | "推几个我能投的，顺便看看第一个我匹配不" | AI简历 | explore, assess | kb_retrieve, global_rank, match_analyze | 含"第一个"引用，测试resolved_refs |

### 2.2 探索 + 面试准备（explore + prepare）— 2 条

| # | 用户消息 | 简历 | gold_intents | expected_tools | 测试重点 |
|---|---------|------|-------------|----------------|---------|
| M03 | "推几个我能投的，顺便告诉我面试要准备啥" | AI简历 | explore, prepare | kb_retrieve, global_rank, interview_gen | **并行执行**：explore和prepare可并行 |
| M04 | "先帮我看看有什么合适的岗，top几个的面试题帮我准备一下" | AI简历 | explore, prepare | kb_retrieve, global_rank, interview_gen | **依赖链**：prepare依赖explore的top结果 |

### 2.3 评估 + 属性查询（assess + verify）— 2 条

| # | 用户消息 | 简历 | gold_intents | expected_tools | 测试重点 |
|---|---------|------|-------------|----------------|---------|
| M05 | "字节AI产品我匹配不？另外他们工资开多少？" | AI简历 | assess, verify | kb_retrieve, match_analyze, qa_synthesize | **共享kb_retrieve**：assess和verify共用同一份JD |
| M06 | "分析一下这个岗，顺便说下加班情况和薪资" | AI简历 | assess, verify | kb_retrieve, match_analyze, qa_synthesize | 上下文引用+多属性查询 |

### 2.4 评估 + 面试准备（assess + prepare）— 2 条

| # | 用户消息 | 简历 | gold_intents | expected_tools | 测试重点 |
|---|---------|------|-------------|----------------|---------|
| M07 | "分析一下这个岗，再给我模拟几个面试题" | AI简历 | assess, prepare | kb_retrieve, match_analyze, interview_gen | **串行依赖**：interview_gen依赖match_analyze结果 |
| M08 | "帮我看看能不能冲，再准备下面试题" | AI简历 | assess, prepare | kb_retrieve, match_analyze, interview_gen | 口语化，测试意图切分 |

### 2.5 探索 + 属性查询（explore + verify）— 2 条

| # | 用户消息 | 简历 | gold_intents | expected_tools | 测试重点 |
|---|---------|------|-------------|----------------|---------|
| M09 | "给我推几个岗，顺便说下他们的薪资和要求" | AI简历 | explore, verify | kb_retrieve, global_rank, qa_synthesize | verify针对explore推荐的top岗位 |
| M10 | "推荐几个AI岗，再告诉我每个的学历要求" | AI简历 | explore, verify | kb_retrieve, global_rank, qa_synthesize | 多岗位属性批量查询 |

---

## 三、按工具调用分类的测试用例（~12 条）

> 确保每个工具都被充分覆盖，包括 Replan fallback 场景

### 3.1 kb_retrieve 覆盖（已在上述用例中充分覆盖，此处补充特殊场景）

| # | 用户消息 | 简历 | expected_tools | 测试重点 |
|---|---------|------|----------------|---------|
| T01 | "火星科技有什么岗？" | 任意 | kb_retrieve | **知识库无命中**，测试空结果处理 |
| T02 | "把所有JD都给我分析一遍" | AI简历 | kb_retrieve, global_rank | **大范围请求**，测试top-k截断 |

### 3.2 external_search 覆盖（Replan 触发场景）

> external_search 是 Replan 时自动插入的，正常流程不会直接调用

| # | 用户消息 | 简历 | expected_tools | 测试重点 |
|---|---------|------|----------------|---------|
| T03 | "帮我查查特斯拉在中国的招聘" | 任意 | kb_retrieve, external_search | **知识库无命中 → Replan触发external_search** |
| T04 | "OpenAI的工程师岗位有什么要求" | 任意 | kb_retrieve, external_search | 外企名，知识库大概率无命中，测试Replan |

### 3.3 match_analyze 覆盖

| # | 用户消息 | 简历 | expected_tools | 测试重点 |
|---|---------|------|----------------|---------|
| T05 | "字节AI产品经理，匹配度如何" | AI简历 | kb_retrieve, match_analyze | 标准匹配分析 |
| T06 | "这个岗适合我吗？" | AI简历 | kb_retrieve, match_analyze | **注入history_slots**: `{"company": "百度", "position": "大模型PM"}`，测试上下文引用 |
| T07 | "Java后端开发，我大专学历2年经验能去吗" | Java简历 | kb_retrieve, match_analyze | 低匹配度输入，测试match_analyze评分客观性 |

### 3.4 global_rank 覆盖

| # | 用户消息 | 简历 | expected_tools | 测试重点 |
|---|---------|------|----------------|---------|
| T08 | "帮我推荐5个最合适的岗位" | AI简历 | kb_retrieve, global_rank | 明确top-k请求 |
| T09 | "按匹配度排序，看看我适合什么" | AI简历 | kb_retrieve, global_rank | 排序意图 |

### 3.5 qa_synthesize 覆盖

| # | 用户消息 | 简历 | expected_tools | 测试重点 |
|---|---------|------|----------------|---------|
| T10 | "字节AI产品岗需要出差吗" | AI简历 | kb_retrieve, qa_synthesize | 布尔型属性问答 |
| T11 | "百度大模型PM的工作内容是什么" | AI简历 | kb_retrieve, qa_synthesize | 描述型属性问答 |

### 3.6 interview_gen 覆盖

| # | 用户消息 | 简历 | expected_tools | 测试重点 |
|---|---------|------|----------------|---------|
| T12 | "准备一下字节AI产品的面试题" | AI简历 | kb_retrieve, interview_gen | 有match_result输入的面试题生成 |
| T13 | "给我出3道Python面试题" | 任意 | interview_gen | 泛化面试题，不依赖具体JD |

### 3.7 evidence_relevance_check 覆盖

> 该工具在 evidence_cache 复用决策时调用

| # | 用户消息 | 简历 | expected_tools | 测试重点 |
|---|---------|------|----------------|---------|
| T14 | "那薪资呢？" | AI简历 | evidence_relevance_check, qa_synthesize | **注入history_slots+evidence_cache**：上一轮是"字节AI产品岗工资多少？"，本轮追问薪资细节，测试缓存复用 |
| T15 | "这个岗需要加班吗？" | AI简历 | evidence_relevance_check, kb_retrieve, qa_synthesize | **注入history_slots+evidence_cache**：上一轮是"帮我推荐岗位"，缓存与当前query不相关，应重新检索 |

### 3.8 general_chat 覆盖

| # | 用户消息 | 简历 | expected_tools | 测试重点 |
|---|---------|------|----------------|---------|
| T16 | "产品经理以后能干到啥级别？" | 任意 | general_chat | 职业规划咨询 |
| T17 | "你觉得我现在转行做AI来得及吗" | 任意 | general_chat | 主观建议类对话 |

---

## 四、复杂流程编排（~8 条）

> 测试 Planner 生成的 DAG 结构是否合理

### 4.1 简单串行（A→B）

| # | 用户消息 | expected_tools | 期望DAG |
|---|---------|----------------|--------|
| F01 | "分析一下字节AI产品岗我能不能去" | kb_retrieve → match_analyze | T0(kb) → T1(match) |

### 4.2 简单并行（A→B, A→C）

| # | 用户消息 | expected_tools | 期望DAG |
|---|---------|----------------|--------|
| F02 | "字节AI产品我匹配不？另外工资多少？" | kb_retrieve → match_analyze + qa_synthesize | T0(kb) → T1(match), T0 → T2(qa) |

### 4.3 依赖链（A→B→C）

| # | 用户消息 | expected_tools | 期望DAG |
|---|---------|----------------|--------|
| F03 | "先帮我挑几个岗，top1的面试题准备一下" | kb_retrieve → global_rank → interview_gen | T0(kb) → T1(rank) → T2(interview) |

### 4.4 多意图混合编排

| # | 用户消息 | expected_tools | 期望DAG |
|---|---------|----------------|--------|
| F04 | "推几个我能投的，分析一下第一个，再出几道面试题" | kb_retrieve → global_rank → match_analyze → interview_gen | T0(kb) → T1(rank) → T2(match) → T3(interview) |
| F05 | "帮我筛一圈，再看看字节和百度我更适合哪个" | kb_retrieve → global_rank → match_analyze(x2) | T0(kb) → T1(rank) → T2(match字节), T0 → T3(match百度) |

### 4.5 Replan 触发（检索不足 → 外部搜索）

| # | 用户消息 | expected_tools | 期望DAG |
|---|---------|----------------|--------|
| F06 | "查查Meta的AI岗位" | kb_retrieve → external_search → qa_synthesize | T0(kb, fail) → [Replan] → T1(ext_search) → T2(qa) |

### 4.6 条件分支（缓存复用 vs 重新检索）

| # | 用户消息 | expected_tools | 期望DAG |
|---|---------|----------------|--------|
| F07 | "那工资范围呢？"（expand类型，缓存相关） | evidence_relevance_check → qa_synthesize | T0(evidence_check, success) → T1(qa, 复用缓存) |
| F08 | "帮我看看Java后端岗"（switch类型，缓存不相关） | kb_retrieve → global_rank | T0(evidence_check, fail) → T1(kb, 重新检索) → T2(rank) |

---

## 五、边界情况（~10 条）

| # | 用户消息 | 预期行为 | 测试重点 |
|---|---------|---------|---------|
| B01 | ""（空消息） | 返回友好提示，不崩溃 | 空输入处理 |
| B02 | "asdfghjkl12345" | 识别为chat，不触发工具 | 无意义输入 |
| B03 | "   "（纯空格） | 同空消息处理 | 空白输入 |
| B04 | "帮我分析"（意图模糊） | 触发澄清或fallback到chat | 意图不明确 |
| B05 | "字节跳动字节跳动字节跳动..."（重复100次） | 正常处理，不崩溃 | 超长/异常输入 |
| B06 | "火星科技有什么岗？" | kb_retrieve返回空，graceful处理 | 知识库无命中 |
| B07 | "字节、百度、阿里、腾讯、美团、京东、网易、快手、小红书、滴滴的产品岗我都适合吗？" | 处理前N个公司，不超时 | 超长实体列表 |
| B08 | "Java Python Go Rust C++ 后端开发工程师" | 提取多个技能，正确匹配 | 多技能混合 |
| B09 | "分析"（无上下文+无实体） | 触发澄清 | 完全无信息 |
| B10 | "我不找工作，就想聊聊天" | 识别为chat，不触发工具 | 明确拒绝服务 |

---

## 六、测试集总览

| 分类 | 条数 | 覆盖工具 | 覆盖意图 |
|------|------|---------|---------|
| 单轮单意图 | 18 | 全部8个 | 全部6种 |
| 单轮多意图 | 10 | 全部8个 | 组合覆盖 |
| 工具专项 | 12 | 每个工具≥2次 | 混合 |
| 复杂编排 | 8 | 全部8个 | 混合 |
| 边界情况 | 10 | general_chat, kb_retrieve | chat, explore |
| **总计** | **58** | — | — |

---

## 七、eval_context 字段规范

每条用例的 `eval_context` 应包含以下字段：

```json
{
  "gold_intents": ["explore"],
  "gold_slots": {"company": "字节跳动", "position": "AI产品经理"},
  "expected_tools": ["kb_retrieve", "match_analyze"],
  "scenario": "场景描述",
  "notes": "备注",
  "resume_id": "eval_resume_ai",
  "injected_history_slots": {"company": "字节跳动"},
  "injected_evidence_cache": [{"content": "...", "metadata": {}}],
  "follow_up_type": "expand"
}
```

| 字段 | 必填 | 说明 |
|------|------|------|
| `gold_intents` | 是 | 期望的意图列表（测试集标签：explore/assess/verify/prepare/chat/clarification） |
| `gold_slots` | 否 | 期望提取的槽位 |
| `expected_tools` | 是 | 期望执行的工具列表（按执行顺序或集合均可） |
| `scenario` | 是 | 场景描述，便于问题定位 |
| `notes` | 否 | 特殊说明 |
| `resume_id` | 否 | 指定测试简历（eval_resume_ai / eval_resume_java / eval_resume_empty） |
| `injected_history_slots` | 否 | **单轮模拟多轮**：注入历史槽位，测试 resolved_refs |
| `injected_evidence_cache` | 否 | **单轮模拟多轮**：注入证据缓存，测试 evidence_relevance_check |
| `follow_up_type` | 否 | **单轮模拟多轮**：指定 follow_up 类型（expand/switch/clarify/none） |

---

## 八、与现有测试集的对比

| 维度 | 现有 46 条 | 建议 58 条 | 改进点 |
|------|-----------|-----------|--------|
| resume_manage 意图 | **0 条** | 1 条 | 补充遗漏意图 |
| evidence_relevance_check 工具 | **0 条** | 2 条 | 补充遗漏工具 |
| 多意图用例 | 8 条 | 10 条 | 增加依赖链和混合编排 |
| Replan 触发 | 1 条 | 2 条 | 增加 external_search fallback |
| 边界情况 | 6 条 | 10 条 | 增加超长输入、多实体等 |
| 单轮模拟多轮 | 0 条 | 4 条 | 新增 injected_history/evidence_cache |
| 多公司/多岗位 | 1 条 | 3 条 | 增加对比分析场景 |
