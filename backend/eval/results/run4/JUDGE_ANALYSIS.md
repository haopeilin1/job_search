# Run4 Judge 通过率极低（20.69%）根因分析报告

> 生成时间：2026-06-04  
> 分析对象：backend/eval/results/run4/_report_judge_v35.json

---

## 一、数据概览

| 指标 | 数值 |
|:---|:---:|
| 总 case 数 | 58 |
| Judge 通过 (resolved) | 12 (20.69%) |
| 规则否决 (veto) | 6 (10.34%) |
| 非否决但未通过 | 40 (68.97%) |
| RAG case 通过 | 3 / 43 (6.98%) |
| 非 RAG case 通过 | 9 / 15 (60.0%) |

**核心发现**：通过率极低不是单一原因造成，而是 **"评测数据格式断层导致的系统性误判"** 与 **"真实系统质量问题"** 叠加的结果。

---

## 二、根因一：评测 Pipeline 数据格式断层（最主要原因，影响约 34 条 case）

### 2.1 问题描述

`batch_eval_runner.py`（生成 run4 的脚本）与 `judge_postprocess.py`（v3.5 Judge 评估）之间存在 **数据格式不兼容**：

| 脚本 | 保存的检索结果格式 | 字段名 |
|:---|:---|:---|
| `batch_eval_runner.py` | 截断字符串（200 字符） | `pred_tools[].result_preview` |
| `judge_postprocess.py` 期望 | 完整字典 | `tool_executions_full[].output.chunks` 或 `kb_chunks` |

### 2.2 实际数据验证

- **43 条调用了 `kb_retrieve` 的 case，全部返回了非空 chunks**（`result_preview` 中均有内容）
- **0 条 case 的 `kb_retrieve` 返回空结果**
- 典型示例：
  - `eval_chen_01`：检索到百度 AI 产品实习生 chunk
  - `eval_chen_02`：检索到字节跳动 AI 产品经理 chunk
  - `eval_chen_08`：检索到小米 AI 培训方向产品实习生 chunk

### 2.3 Judge 的误判逻辑链

```
Case 文件缺少 kb_chunks / tool_executions_full
        ↓
_extract_retrieved_chunks() 返回 []
        ↓
Judge 看到 "检索到的证据 chunks（空）"
        ↓
系统回复包含具体岗位/薪资/匹配分析
        ↓
Judge 判定："检索为空却给出具体信息 → 严重编造"
        ↓
response_accuracy = 0~1 / 5
        ↓
resolved = False（因为关键维度 response_accuracy < 3）
```

### 2.4 受影响 case 统计

- **40 条非否决未通过 case 中，34 条（85%）的 Judge 理由包含"检索为空/编造"**
- **24 条（60%）被指控"编造引用标记"**
- 这些指控的**前提是 Judge 认为检索为空**，但实际检索是有结果的

### 2.5 反证：3 条通过的 RAG case

| Case ID | 场景 | 为什么能通过 |
|:---|:---|:---|
| `eval_li_03` | 属性查询_薪资 | verify 意图，回复可能以"知识库中找到的薪资范围"作答，Judge 认可 |
| `eval_li_06` | 属性查询_经验要求 | 同上，回复更简短直接，没有大量"无法验证"的引用标记 |
| `eval_wang_04` | 属性查询_泛化 | faithfulness 仅 3/5（被扣分），但其他关键维度勉强过线 |

这 3 条全部是 **verify（属性查询）** 意图，回复形式通常是"XX 岗位薪资为 XX"，Judge 对这类简短事实性回答的容错率高于 explore/assess。

---

## 三、根因二：真实系统质量问题（影响约 10~15 条 case）

排除数据格式断层的误判后，以下问题是**真实存在**的：

### 3.1 规则否决：空回复（6 条，10.3%）

| Case ID | 场景 | 可能原因 |
|:---|:---|:---|
| `eval_chen_14` | 澄清后_岗位综合分析 | 传输层错误（ClientPayloadError） |
| `eval_sup_03` | 记忆_偏好引用 | 传输层错误（ClientPayloadError） |
| `eval_chen_11` | 多意图_探索+面试准备 | 系统返回空或极短回复 |
| `eval_gen_04` | 复杂多意图 | 系统返回空或极短回复 |
| `eval_gen_06` | 边界_VERIFY缺少company触发澄清 | 系统返回空或极短回复 |
| `eval_wang_08` | 面试准备_设计岗 | 系统返回空或极短回复 |

**结论**：6 条中有 2 条是已知的 HTTP 传输错误，其余 4 条需要排查系统是否确实返回了空回复。

### 3.2 意图识别错误（5 条）

| Case ID | 期望意图 | 实际识别 | 后果 |
|:---|:---|:---|:---|
| `eval_chen_15` | verify | assess | 未调用 `qa_synthesize` |
| `eval_li_14` | clarification | verify | 应澄清却直接检索 |
| `eval_li_15` | verify | chat | 未调用任何工具 |
| `eval_wang_07` | verify | chat | 未调用 `qa_synthesize` |
| `eval_wang_12` | verify | chat | 未调用 `qa_synthesize` |

**结论**：verify 意图的识别存在系统性问题，5 条中有 3 条被误识别为 chat。这可能是 L1/L2 规则层对短查询（如"阿里巴巴后端开发"）的 intent 分类不够精确。

### 3.3 工具调用错误（4 条）

| Case ID | 问题描述 |
|:---|:---|
| `eval_wang_11` | 多意图 explore+verify，但只识别了 explore，且错误调用了 `parse_resume` |
| `eval_li_04` | prepare 意图但多调用了 `kb_retrieve` 和 `match_analyze` |
| `eval_chen_16` | explore 意图缺少 `global_rank` |
| `eval_chen_04` | prepare 意图多调用了不必要的 `match_analyze` |

### 3.4 引用标记问题

即使检索确实返回了 chunks，系统回复中的引用标记（如 `[来源：小米-AI培训方向产品实习生]`）可能存在以下问题：
- 标记格式不统一，Judge 无法与 chunks 对应
- 标记指向的公司/岗位与 chunks 中的内容不完全一致
- 系统在聚合回复时，LLM 可能自行"总结"了来源信息，导致标记失真

由于数据格式断层，无法精确验证每条引用标记的准确性，但这**大概率是真实存在的质量问题**。

---

## 四、根因三：评测方法论本身的保守性

### 4.1 Judge 通过阈值过高

v3.5 Judge 的通过标准：
- `intent_accuracy >= 3`
- `response_accuracy >= 3`
- `response_completeness >= 3`
- 且未触发任何否决项

这个标准非常严格。即使系统回复 80% 准确，只要 Judge 认为有编造，就会被打到 accuracy=0~1，直接 fail。

### 4.2 非 RAG case 表现验证

| Case 类型 | 数量 | 通过数 | 通过率 |
|:---|:---:|:---:|:---:|
| 非 RAG（问候语、边界、管理、闲聊） | 15 | 9 | **60%** |
| RAG（探索、匹配、查询、准备） | 43 | 3 | **7%** |

非 RAG case 通过率高达 60%，说明 Judge 体系本身在非检索场景下是合理的。RAG 场景的极低通过率主要由"Judge 看不到检索证据"导致。

---

## 五、各意图类型的真实表现估算

如果假设"数据格式断层导致的误判"可以通过补全 kb_chunks 修复（即 response_accuracy 从 0~1 提升到 3~5），各意图的**真实通过率**估算如下：

| 意图 | 总 case | 原始通过 | 估算真实通过 | 剩余问题 |
|:---|:---:|:---:|:---:|:---|
| `verify`（属性查询） | 10 | 6 (60%) | ~7-8 (70-80%) | 意图误识别 2 条 |
| `chat`（闲聊） | 1 | 1 (100%) | 1 (100%) | 无 |
| `boundary/clarify`（边界/澄清） | 10 | 4 (40%) | ~4-5 (40-50%) | 空回复 2 条 |
| `explore`（探索） | 18 | 0 (0%) | ~2-4 (10-20%) | 引用标记问题严重 |
| `assess`（匹配评估） | 10 | 0 (0%) | ~1-2 (10-20%) | 引用标记+编造风险 |
| `prepare`（面试准备） | 4 | 0 (0%) | ~0-1 (0-25%) | 工具调用不当 |
| `manage`（简历管理） | 1 | 0 (0%) | ~0-1 (0-100%) | 样本太少 |

**估算真实通过率：约 25%~40%**（而非当前的 20.69%），但仍不算高。

---

## 六、结论与建议

### 6.1 结论

Run4 Judge 通过率极低（20.69%）是 **三层原因叠加** 的结果：

1. **评测数据格式断层（主因，~60% 影响）**：`batch_eval_runner.py` 未保存完整 `kb_chunks`，导致 Judge 误判 34+ 条 case 为"编造"
2. **真实系统质量问题（次因，~25% 影响）**：空回复 6 条、意图识别错误 5 条、工具调用错误 4 条、引用标记不准确
3. **Judge 阈值严格（次要因素）**：response_accuracy 只要被判定有编造就降至 0~1，直接 fail

### 6.2 建议

| 优先级 | 行动 | 预期效果 |
|:---|:---|:---|
| **P0** | 修复 `batch_eval_runner.py`，在 case 文件中保存完整的 `kb_chunks` 和 `tool_executions_full` | Judge faithfulness 评判准确率大幅提升，通过率从 20% → 30~40% |
| **P0** | 对 run4 重新跑 `batch_http_eval_v3.py`（该脚本会保存完整 debug_info + kb_chunks） | 获得可信的 Judge 指标基线 |
| **P1** | 修复 verify 意图识别：优化 L1/L2 对短查询（如"阿里巴巴后端开发"）的分类逻辑 | 减少 3~5 条意图误识别 |
| **P1** | 排查 4 条非传输错误的空回复 case | 减少 4 条否决 |
| **P2** | 规范引用标记格式，确保 `[来源：公司-岗位]` 与检索 chunks 的 metadata 严格对应 | 减少 Judge 对"编造引用"的误判 |
| **P2** | 在 assess/explore 场景的聚合 Prompt 中增加"如无检索证据，请明确说明"的约束 | 降低幻觉风险，提升 response_accuracy |
