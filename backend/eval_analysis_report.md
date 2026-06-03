# 求职雷达 Agent v3 评测结果逐条分析报告

> 分析对象：`eval/v3_merged_1779859571.json`（54条case）
> 分析时间：2026-05-27

---

## 一、全局系统性问题

### 1.1 ❌ 意图识别层 resume_available 系统性错误

**严重级别：高**

在全部 54 个 case 中，意图识别层（intent_result）几乎**全部错误地**将 `resume_available` 标记为 `false` 或 `null`，即使 `resume_text` 实际上已完整传入。

**影响：**
- `match_assess` 等依赖简历的工具无法获取简历信息
- 系统回复频繁出现"缺少您的简历信息"等错误表述
- 大量 assess/match_analyze 场景的核心任务无法完成

**典型case：**
| case | 消息 | resume_available | 实际resume_text |
|------|------|------------------|-----------------|
| eval_li_02 | 阿里巴巴后端开发我匹配吗 | `false` | ✅ 李工程师完整简历 |
| eval_chen_02 | 字节跳动的AI产品经理我够格吗 | `false` | ✅ 陈雨桐完整简历 |
| eval_chen_05 | 阿里巴巴的AI Agent产品经理匹配吗 | `false` | ✅ 陈雨桐完整简历 |
| eval_chen_12 | 快手搜索推荐AI PM我能去吗 | `false` | ✅ 陈雨桐完整简历 |
| eval_chen_08 | 小米AI培训方向产品实习生我匹配不 | `false` | ✅ 陈雨桐完整简历 |
| eval_wang_05 | 我从品牌设计转UI，能投AI产品岗吗 | `false` | ✅ 王设计师完整简历 |
| eval_li_08 | 字节AI产品经理我能转行去吗 | `false` | ✅ 李工程师完整简历 |

**根因推测：** 意图识别 LLM prompt 中没有正确传递 resume_text，或 LLM 未理解已传入简历的上下文。

---

## 二、逐条问题分析

### 【eval_li_02】阿里巴巴后端开发我匹配吗

| 维度 | 详情 |
|------|------|
| 期望意图 | assess |
| 实际意图 | match_assess ✅ |
| 简历 | ✅ 已传入李工程师简历 |
| **核心问题** | 系统回复"**暂无匹配结论，缺少您的简历信息**" |
| judge | ❌ `resolved=False`, source=llm, accuracy=6, completeness=3 |

**详细分析：**
- intent 识别正确（match_assess），但 `resume_available=false`
- match_analyze 工具因认为无简历，未执行匹配分析
- 最终回复仅列出JD硬性要求，让用户"自测"
- **这不是评测集问题，是系统 resume_available 识别的系统性 bug**

---

### 【eval_wang_10】快手搜索推荐AI PM我匹配吗

| 维度 | 详情 |
|------|------|
| 期望意图 | assess |
| 实际意图 | **demands=[]，needs_clarification=true** ❌ |
| 执行工具 | **0 个** |
| **核心问题** | 用户已明确说"快手搜索推荐AI PM"，系统却要求"请提供公司名和岗位名" |
| judge | ✅ `resolved=True`, source=**rule**, reason="正确触发澄清" |

**详细分析：**
- 意图识别严重错误：用户消息中已包含完整公司和岗位信息，却触发澄清
- **未执行任何工具**（kb_retrieve/match_analyze 均未调用）
- 最终回复为空
- **judge 规则覆盖问题**：`run_eval_v3.py` 第 683 行逻辑——只要 `needs_clarification=true` 就直接给 `resolved=True, source=rule`
- 但此处澄清是**错误触发**的，不应判定为"正确触发澄清"

---

### 【eval_chen_12】快手搜索推荐AI PM我能去吗

| 维度 | 详情 |
|------|------|
| 期望意图 | assess |
| 实际意图 | match_assess ✅ |
| **核心问题** | 检索到**美团**而非快手；`resume_available=false` |
| judge | ✅ `resolved=True`, source=llm, accuracy=7, completeness=7 |

**详细分析：**
- 知识库中**确有快手JD**（ID=9：快手 - AI产品经理实习生-【电商】）
- 但检索未召回任何快手相关结果，反而召回美团搜索推荐AI PM
- 系统回复"检索到的岗位信息属于美团，而非快手"
- **标注问题辨析**：
  - notes 说"知识库中无快手搜索推荐AI PM"——**这个表述本身是对的**，因为确实没有"快手搜索推荐AI PM"这个具体岗位
  - 但知识库**有快手公司JD**，检索应该至少能搜到快手，然后告知"有快手JD但不是搜索推荐方向"
  - 用户提到的"jd_9明明是快手jd，检索却说没有快手相关"——**这是检索策略的问题**，不是标注错误
- `resume_available=false` 导致无法进行匹配分析

---

### 【eval_chen_18】百度大模型应用PM和快手搜索推荐AI PM哪个更适合我

| 维度 | 详情 |
|------|------|
| 期望意图 | assess（多JD对比） |
| 实际意图 | match_assess ✅ |
| **核心问题** | 快手JD未命中；`resume_available=null`；无法完成对比 |
| judge | ✅ `resolved=True`, source=llm, accuracy=6, completeness=4 |

**详细分析：**
- 百度大模型应用PM检索成功
- 快手搜索推荐AI PM检索失败，系统自己承认"系统误检索为美团搜索推荐AI PM"
- 因无法获取快手JD + 简历缺失，无法完成核心对比任务
- judge 评分为 accuracy=6, completeness=4，但因未编造信息仍给 resolved=True

---

### 【eval_li_03】阿里后端岗工资开多少

| 维度 | 详情 |
|------|------|
| 期望意图 | verify（薪资查询） |
| 实际意图 | attribute_verify ✅ |
| **核心问题** | 检索到"某小公司"而非阿里巴巴 |
| judge | ❌ `resolved=False`, source=llm, accuracy=5, completeness=2, citation=2 |

**详细分析：**
- 知识库中阿里巴巴JD（ID=1）明确有薪资范围 35k-65k
- 但检索结果只命中"某小公司"Java后端（10k-20k）
- 系统回复"无法找到阿里相关证据，可能因检索关键词匹配偏差导致"
- **这是检索/召回策略的问题**，BM25/向量检索未正确命中阿里巴巴JD

---

### 【eval_chen_03】百度的AI产品实习生要求什么学历

| 维度 | 详情 |
|------|------|
| 期望意图 | verify（学历查询） |
| 实际意图 | attribute_verify ✅ |
| **核心问题** | 使用了错误的JD（百度大模型应用PM，ID=2）而非百度AI产品实习生 |
| judge | ❌ `resolved=False`, source=llm, accuracy=2, completeness=3, citation=1 |

**详细分析：**
- 知识库中百度相关JD有多条：
  - ID=2：百度 - 大模型应用PM
  - ID=28：百度 - AI应用-产品经理（安全与企业效率平台）
- 系统检索命中了 ID=2（大模型应用PM），但用户问的是"AI产品实习生"
- 系统中没有"百度AI产品实习生"这个精确岗位（最接近的是 ID=28 或 ID=2）
- 回复编造了"计算机科学、人工智能、数据科学、产品设计"等专业要求
- judge 严厉扣分（accuracy=2），说明确实用了错误JD

---

### 【eval_li_15】阿里巴巴后端开发

| 维度 | 详情 |
|------|------|
| 期望意图 | verify（澄清后） |
| 实际意图 | **demands=[]，needs_clarification=true** |
| 执行工具 | **0 个** |
| judge | ✅ `resolved=True`, source=**rule**, reason="正确触发澄清" |

**详细分析：**
- 用户仅说"阿里巴巴后端开发"，意图确实不明确（是查询、匹配还是面试准备？）
- 系统触发澄清"能再详细说说你的需求吗？"——**这是合理的**
- 与 eval_wang_10 不同，这里的澄清是**正确**的
- 但同样被 rule 直接给 True，未进入 LLM judge

---

### 【eval_chen_13】分析一下这个岗

| 维度 | 详情 |
|------|------|
| 期望意图 | clarification |
| 实际意图 | **demands=[]，needs_clarification=true** ✅ |
| judge | ✅ `resolved=True`, source=**rule**, reason="正确触发澄清" |

**详细分析：**
- "这个岗"无上下文指代，正确触发澄清
- 这是 3 条 rule=True 中**唯一真正正确触发澄清**的 case

---

### 【eval_gen_04】帮我列出已上传的简历

| 维度 | 详情 |
|------|------|
| 期望意图 | manage（简历管理） |
| 实际意图 | resume_manage ✅ |
| 执行工具 | **0 个** |
| **核心问题** | 系统回复"当前系统未检索到任何已上传的简历文件"——但实际已上传 |
| judge | ✅ `resolved=True`, source=llm |

**详细分析：**
- expected_tools=['file_ops']，但未执行任何工具
- 系统声称没有简历，但实际上传了陈雨桐简历
- 这是 resume 存储/查询接口的问题

---

### 【eval_wang_12】我的4年互联网设计经验对产品岗有帮助吗

| 维度 | 详情 |
|------|------|
| 期望意图 | verify |
| 实际意图 | attribute_verify ✅ |
| **核心问题** | 回复为空或极短 |
| judge | ❌ `resolved=False`, source=llm, rule_hit=**empty_reply** |

**详细分析：**
- `rule_hit=empty_reply`，judge 直接给 0 分
- 这是执行异常导致的空回复

---

### 【eval_li_09】我的MySQL分库分表经验对哪个岗最有价值

| 维度 | 详情 |
|------|------|
| 期望意图 | explore |
| 实际意图 | position_explore + attribute_verify ✅ |
| **核心问题** | 回复为空或极短 |
| judge | ❌ `resolved=False`, source=llm, rule_hit=**empty_reply** |

**详细分析：**
- 同 eval_wang_12，执行异常导致空回复
- rule_hit=empty_reply 直接判 False

---

## 三、JUDGE 评判机制详细分析

### 3.1 评判结果分布

| 类型 | 数量 | 说明 |
|------|------|------|
| source=rule | 3 | 直接规则判定，不经过LLM judge |
| source=llm + resolved=True | 39 | LLM judge 判定通过 |
| source=llm + resolved=False | 12 | LLM judge 判定不通过 |

### 3.2 source=rule 的 3 条

| case | 触发条件 | 是否合理 |
|------|----------|----------|
| eval_chen_13 | needs_clarification=true | ✅ 合理，"这个岗"确实需要澄清 |
| eval_li_15 | needs_clarification=true | ✅ 合理，"阿里巴巴后端开发"意图不明确 |
| eval_wang_10 | needs_clarification=true | ❌ **不合理**，用户已明确"快手搜索推荐AI PM" |

**规则漏洞**：`run_eval_v3.py` 第 683 行无条件将 `needs_clarification=true` 判定为"正确触发澄清"，未区分澄清是否正确/必要。

### 3.3 LLM judge=False 但被 code_override 为 True 的 6 条

| case | LLM原判 | override原因 | 各维度评分 |
|------|---------|--------------|------------|
| eval_gen_03 | False | acc=7≥4, comp=4≥4, rel=5≥4 | acc=7, comp=4, cit=5, rel=5 |
| eval_sup_03 | False | acc=8≥4, comp=4≥4, rel=8≥4 | acc=8, comp=4, cit=4, rel=8 |
| eval_wang_05 | False | acc=6≥4, comp=5≥4, rel=7≥4 | acc=6, comp=5, cit=4, rel=7 |
| eval_li_10 | False | acc=5≥4, comp=4≥4, rel=5≥4 | acc=5, comp=4, cit=3, rel=5 |
| eval_li_11 | False | acc=5≥4, comp=6≥4, rel=9≥4 | acc=5, comp=6, cit=3, rel=9 |
| eval_chen_17 | False | acc=5≥4, comp=7≥4, rel=9≥4 | acc=5, comp=7, cit=2, rel=9 |

**code_override 规则**：`run_eval_v3.py` 中当 `accuracy≥4 && completeness≥4 && relevance≥4` 时，即使 LLM 判 False，代码也会 override 为 True。

**分析**：
- 这 6 条的 citation 普遍较低（2-5分），说明引用质量差
- eval_chen_17 甚至 accuracy=5, citation=2（编造信息），但因为 comp=7, rel=9 被 override 为 True
- 该 override 门槛（≥4）偏低，导致部分明显有问题的 case 被放过

### 3.4 rule_hit=empty_reply 的 2 条

| case | 现象 |
|------|------|
| eval_wang_12 | 回复为空/极短 |
| eval_li_09 | 回复为空/极短 |

这两种情况 judge 直接给 0 分，**正确**。

---

## 四、标注正确性辨析

### 4.1 "快手"相关标注

用户提到"jd_9明明是快手jd，可是检索却说没有检索到快手相关"。

**实际情况：**
- 知识库确有快手JD（ID=9：AI产品经理实习生-【电商】）
- 用户查询的是"快手搜索推荐AI PM"，这个**具体岗位**确实不存在
- 标注 notes 说"知识库中无快手搜索推荐AI PM"——**表述正确**
- 但系统的检索问题更严重：连"快手"这个公司都没召回，而是召回美团

**结论**：标注不是错误，但系统检索策略有重大缺陷。

### 4.2 "没有简历"相关

测试时确实传了对应人物简历（resume_text 非空），但：
- 意图识别层**系统性**将 resume_available 设为 false/null
- 导致 match_analyze 等工具无法获取简历
- 系统回复出现"缺少您的简历信息"

**结论**：这是系统 bug，不是评测集问题。

### 4.3 "相关性为0却说相关性很高"

test_dataset.jsonl 中所有 case 的 `relevance_score` 都标注为 0，这不是检索相关性的分数，而是测试集设计时的一个固定值（未实际使用）。因此不存在"标注说相关性为0但实际很高"的问题。

---

## 五、核心问题优先级排序

| 优先级 | 问题 | 影响case数 | 修复难度 |
|--------|------|-----------|----------|
| P0 | 意图识别 resume_available 系统性错误 | ~50 | 中 |
| P1 | 检索策略缺陷（公司名未召回） | 快手/阿里等 | 中 |
| P1 | eval_wang_10 意图识别错误触发澄清 | 1 | 低 |
| P2 | judge rule_override 门槛过低 | 6 | 低 |
| P2 | judge rule 对 needs_clarification 无条件通过 | 3 | 低 |
| P3 | 个别JD匹配错误（chen_03用错JD） | 1 | 低 |

---

## 附录：54条case judge结果速查表

| case | batch | judge_resolved | judge_source | acc | comp | cit | rel | code_override | 主要问题 |
|------|-------|----------------|--------------|-----|------|-----|-----|---------------|----------|
| eval_gen_01 | gen | True | llm | 10 | 10 | 10 | 10 | - | 无 |
| eval_gen_02 | gen | True | llm | 10 | 10 | 10 | 10 | - | 无 |
| eval_gen_03 | gen | True | llm | 7 | 4 | 5 | 5 | ✅ | 缺少prepare |
| eval_gen_04 | gen | True | llm | 8 | 8 | 7 | 8 | - | 未找到简历 |
| eval_gen_05 | gen | False | llm | 4 | 3 | 6 | 5 | - | 意图识别错误 |
| eval_sup_01 | sup | True | llm | 7 | 8 | 8 | 9 | - | 无 |
| eval_sup_02 | sup | False | llm | 4 | 2 | 1 | 1 | - | 编造JD |
| eval_sup_03 | sup | True | llm | 8 | 4 | 4 | 8 | ✅ | 覆盖不全 |
| eval_sup_04 | sup | True | llm | 8 | 4 | 8 | 9 | - | 未指出不匹配 |
| eval_wang_01 | wang | True | llm | 8 | 7 | 7 | 9 | - | 无设计岗（预期） |
| eval_wang_02 | wang | True | llm | 8 | 7 | 7 | 9 | - | 无UI设计岗（预期） |
| eval_wang_03 | wang | True | llm | 7 | 7 | 6 | 9 | - | 无精确匹配 |
| eval_wang_04 | wang | True | llm | 9 | 8 | 7 | 10 | - | 无 |
| eval_wang_05 | wang | True | llm | 6 | 5 | 4 | 7 | ✅ | 未做匹配分析 |
| eval_wang_06 | wang | True | llm | 8 | 7 | 7 | 9 | - | 无精确匹配 |
| eval_wang_07 | wang | True | llm | 8 | 7 | 6 | 9 | - | 无精确匹配 |
| eval_wang_08 | wang | False | llm | 5 | 4 | 5 | 2 | - | 回复偏题（产品vs设计） |
| eval_wang_09 | wang | True | llm | 9 | 7 | 8 | 9 | - | 无精确匹配 |
| eval_wang_10 | wang | True | **rule** | 8 | 8 | 8 | 8 | - | ❌ 错误触发澄清 |
| eval_wang_11 | wang | True | llm | 8 | 8 | 7 | 9 | - | 无精确匹配 |
| eval_wang_12 | wang | False | llm | 0 | 0 | 0 | 0 | - | 空回复 |
| eval_li_01 | li | True | llm | 8 | 5 | 6 | 9 | - | 只返回1个岗位 |
| eval_li_02 | li | False | llm | 6 | 3 | 8 | 6 | - | ❌ 声称缺少简历 |
| eval_li_03 | li | False | llm | 5 | 2 | 2 | 3 | - | ❌ 检索到错误公司 |
| eval_li_04 | li | True | llm | 8 | 6 | 9 | 9 | - | 未生成具体面试题 |
| eval_li_05 | li | True | llm | 8 | 7 | 7 | 9 | - | 无精确匹配 |
| eval_li_06 | li | True | llm | 8 | 8 | 8 | 9 | - | 无 |
| eval_li_07 | li | True | llm | 7 | 5 | 8 | 7 | - | 未做assess |
| eval_li_08 | li | True | llm | 6 | 4 | 4 | 8 | - | 引用JD信息不准确 |
| eval_li_09 | li | False | llm | 0 | 0 | 0 | 0 | - | 空回复 |
| eval_li_10 | li | True | llm | 5 | 4 | 3 | 5 | ✅ | 面试题不相关 |
| eval_li_11 | li | True | llm | 5 | 6 | 3 | 9 | ✅ | 编造要求 |
| eval_li_12 | li | True | llm | 7 | 6 | 6 | 8 | - | 部分岗位缺少薪资 |
| eval_li_14 | li | False | llm | 4 | 3 | 5 | 4 | - | 未触发澄清 |
| eval_li_15 | li | True | **rule** | 8 | 8 | 8 | 8 | - | 正确触发澄清 |
| eval_li_16 | li | True | llm | 8 | 7 | 8 | 9 | - | 无精确匹配（预期） |
| eval_chen_01 | chen | True | llm | 8 | 8 | 7 | 9 | - | 无精确匹配 |
| eval_chen_02 | chen | False | llm | 6 | 2 | 7 | 6 | - | 未做匹配分析 |
| eval_chen_03 | chen | False | llm | 2 | 3 | 1 | 5 | - | ❌ 使用错误JD |
| eval_chen_04 | chen | True | llm | 9 | 7 | 8 | 10 | - | 未生成具体面试题 |
| eval_chen_05 | chen | False | llm | 6 | 3 | 5 | 5 | - | 未做匹配分析 |
| eval_chen_06 | chen | True | llm | 8 | 7 | 6 | 9 | - | 无薪资信息（预期） |
| eval_chen_07 | chen | False | llm | 6 | 3 | 6 | 6 | - | 遗漏explore |
| eval_chen_08 | chen | True | llm | 7 | 4 | 5 | 7 | - | 未做匹配分析 |
| eval_chen_09 | chen | True | llm | 8 | 8 | 7 | 9 | - | 无精确匹配 |
| eval_chen_10 | chen | True | llm | 8 | 8 | 7 | 9 | - | 无 |
| eval_chen_11 | chen | True | llm | 8 | 7 | 6 | 9 | - | 缺少面试题 |
| eval_chen_12 | chen | True | llm | 7 | 7 | 6 | 8 | - | ❌ 检索到美团 |
| eval_chen_13 | chen | True | **rule** | 8 | 8 | 8 | 8 | - | 正确触发澄清 |
| eval_chen_14 | chen | True | llm | 9 | 7 | 8 | 10 | - | 无 |
| eval_chen_15 | chen | True | llm | 6 | 7 | 3 | 8 | - | 编造引用 |
| eval_chen_16 | chen | True | llm | 8 | 6 | 7 | 9 | - | 未列具体岗位 |
| eval_chen_17 | chen | True | llm | 5 | 7 | 2 | 9 | ✅ | 编造岗位信息 |
| eval_chen_18 | chen | True | llm | 6 | 4 | 8 | 7 | - | ❌ 快手JD未命中 |
