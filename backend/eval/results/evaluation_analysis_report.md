# 求职雷达 Agent — 三轮评测结果深度分析报告

> **评测时间**: 2026-04-30  
> **评测规模**: 55 条测试用例 × 3 轮 = 165 次独立调用  
> **模型配置**: CHAT=qwen3.5-plus, CORE=deepseek-v4-flash, PLANNER=qwen2.5:14b, MEMORY=qwen2.5:7b, JUDGE=qwen2.5:7b  
> **评测脚本**: `batch_eval_runner.py` + `judge_postprocess.py`  

---

## 一、执行摘要

| 维度 | 结论 |
|------|------|
| **系统稳定性** | ✅ 优秀。Run1 成功率 94.5% → Run3 100%，说明系统鲁棒性在提升 |
| **意图识别** | ⚠️ 中等。整体约 60%，但存在严重两极分化：prepare 100% vs verify 36% |
| **工具调用** | ⚠️ 中等。平均匹配率 ~79%，完全匹配率 ~73%，主要问题在多意图场景和 verify 场景 |
| **回复质量** | ⚠️ 中等。Judge 任务完成率 81.8%，有 5~8 例空回复 |
| **延迟性能** | ⚠️ 一般。平均 ~74s，中位 ~65s，最大 301s，部分用例明显超时 |

**核心问题排序**:
1. 🔴 **verify 意图识别极差** (36.4%)，大量被误识别为 assess / chat / explore
2. 🔴 **clarification 意图完全失败** (0%)，系统未进入澄清流程
3. 🟡 **空回复问题** (每轮 5~8 例)，Judge 判定失败的主要原因
4. 🟡 **高延迟/超时** (每轮 2~5 例 >120s)
5. 🟢 **prepare 完美** (100%)、**assess 较好** (84.6%)

---

## 二、三轮核心指标稳定性对比

| 指标 | Run1 | Run2 | Run3 | 均值(μ) | 标准差(σ) | 变异系数(CV) | 稳定性 |
|------|------|------|------|---------|-----------|-------------|--------|
| 请求成功率 | 94.5% | 98.2% | **100%** | 97.6% | 0.028 | 2.8% | ✅ 优秀 |
| 意图匹配率 | 60.0% | 60.0% | 60.0% | 60.0% | 0.000 | 0.0% | ✅ 完全一致 |
| 工具平均匹配率 | 79.1% | 78.6% | 79.0% | 78.9% | 0.003 | 0.3% | ✅ 高度稳定 |
| 工具完全匹配率 | 72.3% | 72.3% | 74.5% | 73.0% | 0.012 | 1.7% | ✅ 高度稳定 |
| 回复完成率 | 85.5% | 89.1% | 90.9% | 88.5% | 0.028 | 3.1% | ✅ 稳定 |
| 平均 TTFB | 4.48s | 0.01s | 0.01s | 1.50s | 2.58 | 172% | ⚠️ Run1 异常 |
| 平均总延迟 | 82.3s | 69.4s | 70.4s | 74.0s | 7.15 | 9.7% | ✅ 稳定 |

**关键发现**:
- 除 TTFB 外，所有指标三轮变异系数均 <10%，说明系统行为高度可复现
- TTFB 的异常是因为 Run1 部分用例出现 SSE 连接延迟，后续轮次已优化
- 成功率逐轮提升：Run1 的 3 个 TimeoutError 在 Run2/3 中消失

---

## 三、请求成功率与失败分析

### 3.1 失败用例详情

| 轮次 | Case ID | Gold Intent | 错误类型 | 用户输入 (摘要) |
|------|---------|-------------|----------|-----------------|
| Run1 | eval_wang_03 | explore | **TimeoutError** | "我的Figma组件化经验能投哪个岗..." |
| Run1 | eval_wang_04 | verify | **TimeoutError** | "产品岗对设计能力有要求吗..." |
| Run1 | eval_wang_05 | assess | **TimeoutError** | "我从品牌设计转UI，能投AI产品岗吗..." |
| Run2 | eval_gen_03 | chat | **TimeoutError** | "..." (无意义输入) |

**根因分析**:
- 4 例失败全部为 **TimeoutError (300s 超时)**，非逻辑错误
- eval_wang_03/04/05 连续 3 例超时，推测是 Run1 初期 Ollama 模型加载/预热问题
- Run2 仅 1 例超时，Run3 零失败，说明系统稳定性在持续优化

---

## 四、意图识别准确率详细分析

### 4.1 修正映射后的意图识别准确率（含别名归一化）

> 注：系统将 `match_assess` 映射为 `assess`，`attribute_verify` 映射为 `verify`，`position_explore` 映射为 `explore`。

| Intent | 用例数 | 至少一轮匹配 | 全轮匹配 | 三轮平均匹配率 | 评级 |
|--------|--------|-------------|---------|---------------|------|
| **prepare** | 4 | 4 (100%) | 4 (100%) | **100.0%** | 🟢 优秀 |
| **assess** | 13 | 12 (92%) | 10 (77%) | **84.6%** | 🟢 良好 |
| **chat** | 6 | 5 (83%) | 4 (67%) | **77.8%** | 🟡 尚可 |
| **explore** | 19 | 12 (63%) | 11 (58%) | **61.4%** | 🟡 一般 |
| **verify** | 11 | 5 (45%) | 3 (27%) | **36.4%** | 🔴 极差 |
| **clarification** | 2 | 0 (0%) | 0 (0%) | **0.0%** | 🔴 完全失败 |

### 4.2 误识别模式统计（三轮合并，去重 case）

| 错误模式 | 次数 | 根因分析 |
|----------|------|----------|
| `verify` → `match_assess` | **9 次** | 🔴 最严重。属性查询（如"产品岗对设计能力有要求吗"）被识别为匹配评估 |
| `verify` → `chat` | **6 次** | 属性查询被识别为闲聊，可能是问题过于泛化 |
| `assess` → `position_explore` | **5 次** | 匹配评估被识别为岗位探索，用户query可能偏向"推荐"语义 |
| `verify` → `position_explore` | **3 次** | 属性查询被识别为探索，如"Java岗对学历有什么要求" |
| `verify` → `interview_prepare` | **2 次** | 属性查询被识别为面试准备 |
| `timeout/None` | **2 次** | 超时导致无意图输出 |

### 4.3 verify 意图深度分析

**verify（属性查询/要求验证）是系统的最大短板。**

典型误识别案例：

| Case | 用户输入 | Gold | Pred | 问题 |
|------|----------|------|------|------|
| eval_chen_10 | "算法岗对学历有什么硬性要求吗" | verify | match_assess | 学历要求被识别为匹配评估 |
| eval_chen_15 | "产品岗对设计能力有要求吗" | verify | match_assess | 技能要求被识别为匹配评估 |
| eval_li_11 | "这个岗位对数据分析要求高吗" | verify | interview_prepare | 技能要求被识别为面试准备 |
| eval_wang_07 | "Java岗对学历有什么要求" | verify | chat | 被识别为闲聊 |
| eval_chen_06 | "算法岗一般要什么学历" | verify | chat (Run1/2), match_assess (Run3) | 完全被误识别 |

**根因判断**:
1. **训练数据/示例不足**：verify 类型的 query 在 Prompt 示例中占比过低
2. **语义边界模糊**："对XX有什么要求" 与 "我的XX经验能匹配吗" 在语义上接近，模型难以区分
3. **缺少上下文利用**：部分 verify query 在对话中带有上下文引用（如"这个岗位"），但系统未正确解析

### 4.4 clarification 意图完全失败分析

| Case | 用户输入 | Gold | Pred (三轮) | 系统实际行为 |
|------|----------|------|------------|-------------|
| eval_chen_13 | "分析一下这个岗" | clarification | match_assess | 系统直接猜测用户意图为匹配评估，给出完整分析 |
| eval_li_14 | "分析这个Java岗" | clarification | chat / match_assess | 系统直接尝试分析，未要求澄清 |

**根因**: 系统缺少**主动澄清机制**。当用户输入意图模糊时（如"分析一下这个岗"缺少上下文），系统应反问"您是指哪个岗位？"，但当前实现直接猜测意图并执行。这是一个架构层面的缺失。

---

## 五、工具调用分析

### 5.1 工具匹配率分布

| 匹配率 | Run1 | Run2 | Run3 |
|--------|------|------|------|
| 0.0 | 5 | 5 | 5 |
| 0.3 | 1 | 0 | 0 |
| 0.5 | 7 | 9 | 9 |
| 0.7 | 2 | 3 | 3 |
| 1.0 | 32 | 32 | 33 |
| **总计** | 47 | 49 | 50 |

### 5.2 典型工具不匹配案例

| Case | Gold Intent | 匹配率 | 期望工具 | 实际调用 | 问题描述 |
|------|-------------|--------|----------|----------|----------|
| eval_chen_04 | prepare | 0.5 | kb_retrieve, interview_questions | kb_retrieve, **match_analyze**, interview_gen | 面试准备场景调用了 match_analyze 而非 interview_questions |
| eval_chen_06 | verify | 0.0 | qa_synthesize | **(空)** | verify 被误识别为 chat，未调用任何工具 |
| eval_chen_10 | verify | 0.5 | kb_retrieve, qa_synthesize | kb_retrieve, **match_analyze** | verify 被误识别为 assess，调用了 match_analyze |
| eval_chen_11 | explore+prepare | 0.7 | kb_retrieve, global_rank, interview_questions | kb_retrieve, global_rank, match_analyze, interview_gen | 多意图场景工具冗余，interview_questions 被替换为 interview_gen |
| eval_chen_15 | verify | 0.5 | kb_retrieve, qa_synthesize | kb_retrieve, **match_analyze** | 同 eval_chen_10 |

**根因总结**:
1. **意图误识别直接导致工具错配**：verify → match_assess 是最主要的问题模式
2. **面试准备场景工具定义不一致**：`interview_questions` vs `interview_gen`，期望与实现存在偏差
3. **多意图场景工具冗余**：explore+prepare 同时调用了 match_analyze 和 interview_gen，超出了期望范围
4. **空工具调用**：当意图被识别为 chat 时，系统不调用任何工具，导致工具匹配率为 0

---

## 六、Judge 评估分析（仅 Run1）

### 6.1 Judge 任务完成率

| Intent | 用例数 | Resolved | 完成率 |
|--------|--------|----------|--------|
| prepare | 4 | 4 | **100%** |
| assess | 13 | 12 | **92%** |
| explore | 19 | 17 | **89%** |
| verify | 11 | 8 | **73%** |
| chat | 6 | 3 | **50%** |
| clarification | 2 | 1 | **50%** |
| **总体** | **55** | **45** | **81.8%** |

### 6.2 Judge 判定失败的原因分类

| 失败原因 | 案例数 | 典型 Case |
|----------|--------|-----------|
| **回复为空或过短** | 8 | eval_chen_06, eval_gen_04, eval_li_13/14, eval_wang_03/04/05/07 |
| **意图未明确/未针对性回答** | 2 | eval_gen_01, eval_gen_03 |

**核心发现**: Judge 失败的 10 例中，**8 例（80%）是因为空回复**。这与用户感知一致——即使请求成功（status=success），如果最终 reply 为空，用户体验为"系统无响应"。

---

## 七、回复质量与空回复分析

### 7.1 空回复统计

| 轮次 | 空回复数 | 占比 | 涉及 Case (Gold Intent) |
|------|----------|------|------------------------|
| Run1 | 8 | 14.5% | eval_chen_06(verify), eval_gen_04(多意图), eval_li_13(chat), eval_li_14(clarification), eval_wang_03/04/05(超时), ... |
| Run2 | 6 | 10.9% | eval_chen_06(verify), eval_gen_03(chat/超时), eval_gen_04(多意图), eval_li_13(chat), eval_li_14(clarification) |
| Run3 | 5 | 9.1% | eval_chen_06(verify), eval_gen_04(多意图), eval_li_13(chat), eval_li_14(clarification), eval_wang_07(verify) |

### 7.2 空回复根因分析

| Case | 三轮表现 | 根因判断 |
|------|----------|----------|
| eval_chen_06 | 全部空回复 | verify 被误识别为 chat，chat 意图不调用工具，且 LLM 未生成回复 |
| eval_gen_04 | 全部空回复 | 多意图(explore+prepare+verify)场景复杂，可能工具链执行失败导致最终无输出 |
| eval_li_13 | 全部空回复 | chat 意图但用户输入为"..."，无意义输入导致系统无法生成有意义回复 |
| eval_li_14 | 全部空回复 | clarification 被误识别，系统行为不稳定 |
| eval_wang_03/04/05 | Run1 空(超时)，Run2/3 正常 | 超时导致 SSE 连接中断，无 reply |

---

## 八、延迟性能分析

### 8.1 延迟指标对比

| 指标 | Run1 | Run2 | Run3 |
|------|------|------|------|
| 平均延迟 | 82.3s | 69.4s | 70.4s |
| 中位延迟 | 67.9s | 65.7s | 61.1s |
| 最大延迟 | 146.7s | 146.9s | 151.5s |
| >120s 用例数 | 5 | 2 | 3 |

### 8.2 高延迟用例 (>120s)

| Case | Run1 | Run2 | Run3 | Gold Intent | 说明 |
|------|------|------|------|-------------|------|
| eval_wang_04 | **301.0s** | 正常 | 正常 | verify | Run1 超时，后续轮次正常 |
| eval_wang_05 | **300.9s** | 正常 | 正常 | assess | Run1 超时，后续轮次正常 |
| eval_wang_03 | **300.2s** | 正常 | 正常 | explore | Run1 超时，后续轮次正常 |
| eval_gen_03 | 146.7s | **301.0s** | 正常 | chat | Run2 超时，Run3 正常 |
| eval_li_11 | 137.6s | 正常 | 正常 | verify | 单次高延迟 |
| eval_sup_04 | — | 127.6s | — | explore | 单次高延迟 |
| eval_li_03 | — | — | 139.6s | verify | 单次高延迟 |
| eval_li_07 | — | — | 127.5s | assess+explore | 多意图高延迟 |
| eval_li_01 | — | — | 124.9s | explore | 单次高延迟 |

**延迟根因判断**:
1. **Ollama 本地模型预热**：Run1 初期连续 3 例 300s 超时，与 Ollama 模型加载/上下文切换相关
2. **多意图场景复杂推理**：explore+assess/verify 组合场景需要更多 LLM 调用步骤
3. **reranker/检索耗时**：部分场景涉及大量 chunks 的 reranking，BGE reranker 推理耗时
4. **工具链串行执行**：当前 ReAct 执行器部分工具串行，导致总延迟累加

---

## 九、根因总结

### 9.1 核心问题树

```
系统整体可用性 ~82% (Judge完成率)
├── 意图识别问题 ~60%
│   ├── verify 识别极差 (36%) ← 最大痛点
│   │   ├── Prompt 示例不足 (verify 类型示例占比低)
│   │   ├── 语义边界模糊 ("要求" vs "匹配" 难以区分)
│   │   └── 上下文引用未利用 ("这个岗位"缺少指代消解)
│   ├── clarification 完全失败 (0%)
│   │   └── 架构缺失：无主动澄清机制
│   └── explore 识别一般 (61%)
│       └── 用户query偏向"推荐"语义时易与 assess 混淆
├── 工具调用问题 ~73%完全匹配
│   ├── 意图误识别传导 (verify→assess 导致工具错配)
│   ├── 面试工具命名不一致 (interview_questions vs interview_gen)
│   └── 多意图场景工具冗余
├── 回复质量问题 ~82%完成率
│   ├── 空回复 (5~8例/轮) ← 80%的Judge失败原因
│   │   ├── 意图误识别后不生成回复 (verify→chat)
│   │   ├── 多意图复杂场景工具链失败
│   │   └── 无意义输入无兜底回复
│   └── 回复过短/不相关 (chat场景)
└── 延迟问题 ~74s平均
    ├── Ollama 本地模型预热/切换
    ├── reranker 推理耗时
    └── 多意图多工具串行执行
```

### 9.2 三轮稳定性结论

- ✅ **稳定性优秀**：核心指标 CV < 10%，系统行为可复现
- ✅ **故障自愈**：Run1 的 3 个超时在后续轮次未复现
- ⚠️ **意图识别固化**：60% 的匹配率三轮完全一致（CV=0%），说明这不是随机波动，而是**系统性缺陷**
- ⚠️ **空回复固化**：eval_chen_06, eval_gen_04, eval_li_13/14 每轮都空回复，是**确定性 bug**

---

## 十、改进建议

### 🔴 P0 — 立即修复（影响可用性）

#### 1. 修复 verify 意图识别
**预期效果**: verify 识别率从 36% → 80%+

| 措施 | 实现方式 | 工作量 |
|------|----------|--------|
| **增强 Prompt 示例** | 在 planner prompt 中增加 5~8 个 verify 类型示例，覆盖"学历要求""技能要求""经验要求""薪资要求" | 小 |
| **增加关键词启发** | 在 query 中出现"要求""条件""门槛""需要什么""学历""经验""技能"时，提高 verify 优先级 | 小 |
| **上下文指代消解** | 当 query 含"这个岗位""该职位"时，结合对话历史确定目标 JD | 中 |

#### 2. 修复空回复问题
**预期效果**: 空回复从 5~8 例/轮 → 0~1 例/轮

| 措施 | 实现方式 | 工作量 |
|------|----------|--------|
| **兜底回复机制** | 当 reply 为空时，fallback 到通用回复模板（如"抱歉，我没有理解您的问题，能否再详细描述一下？"） | 小 |
| **诊断空回复根因** | 针对 eval_chen_06 (verify→chat 空回复)、eval_gen_04 (多意图空回复) 做单步调试，定位工具链断裂点 | 中 |

#### 3. 实现 clarification 主动澄清
**预期效果**: clarification 识别率从 0% → 60%+

| 措施 | 实现方式 | 工作量 |
|------|----------|--------|
| **低置信度触发澄清** | 当 planner 对意图分类的置信度 < 0.7 时，返回 need_clarification | 中 |
| **模糊输入检测** | 当用户输入缺少关键实体（如"分析一下这个岗"缺少岗位名/公司名），主动要求澄清 | 中 |
| **澄清后意图继承** | 用户回复澄清后，系统应正确继承上一轮上下文并重新分类 | 中 |

### 🟡 P1 — 短期优化（提升体验）

#### 4. 统一面试工具命名
- 将 `interview_questions` 与 `interview_gen` 统一命名和调用逻辑
- 更新 test_dataset.jsonl 中的 expected_tools 与实际系统保持一致

#### 5. 延迟优化
| 措施 | 预期效果 |
|------|----------|
| Ollama 模型常驻 + keep_alive | 消除模型预热导致的 300s 超时 |
| 工具并行执行优化 | 减少多工具场景的串行等待 |
| reranker 批处理 | 减少 BGE reranker 的多次推理 |

#### 6. 增强 chat 场景兜底
- 对于 eval_li_13 ("..." 无意义输入)，应返回友好提示而非空回复
- 对于 eval_gen_03 (边界_无意义输入)，应明确告知用户输入无效

### 🟢 P2 — 长期增强

#### 7. 评测数据补充
- 当前 55 条用例中 verify 仅 11 条、clarification 仅 2 条，样本不足
- 建议扩充 verify 用例至 20+ 条，clarification 用例至 8+ 条

#### 8. 引入意图识别微调
- 收集 200+ 条标注数据，对 planner 模型 (qwen2.5:14b) 进行 LoRA 微调
- 预期可将整体意图匹配率从 60% 提升至 85%+

#### 9. 端到端可观测性
- 在 batch_eval_runner 中集成 token 消耗统计（Run2/3 代码已支持但数据未正确收集）
- 增加每个工具的执行耗时追踪，定位延迟瓶颈

---

## 附录：原始数据索引

| 文件 | 说明 |
|------|------|
| `eval/results/run1/_report.json` | Run1 基础指标报告 |
| `eval/results/run1/_report_judge.json` | Run1 Judge 评估报告 |
| `eval/results/run2/_report.json` | Run2 基础指标报告 |
| `eval/results/run3/_report.json` | Run3 基础指标报告 |
| `eval/results/_summary.json` | 三轮汇总（原始格式） |
| `eval/results/run{1-3}/eval_*.json` | 每条用例的详细执行结果 |
| `eval/results/runner.log` | Round 1 执行日志 |
| `eval/results/auto_continue.log` | 自动流水线日志 |
