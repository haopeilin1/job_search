# Agent 全链路测试审计报告（修正版）

> 测试日期：2026-05-29
> 测试目标：验证 v3 Agent 全链路测试体系（run_eval_v3.py + judge_postprocess.py）的完整性与健壮性
> 测试用例：新增多轮对话+第二轮澄清+RAG意图场景（chen_m3组）

---

## 一、测试用例设计

### 1.1 用例概述

| 轮次 | Case ID | 用户输入 | Gold Intents | Expected Tools | 场景说明 |
|------|---------|----------|-------------|----------------|---------|
| Turn 1 | eval_chen_19 | "帮我推荐几个适合我的AI产品实习岗" | explore | kb_retrieve, global_rank | 第一轮正常RAG探索，推荐多个岗位 |
| Turn 2 | eval_chen_20 | "这个岗需要出差吗" | clarification | (空) | 第二轮指代不清，应触发澄清 |

- 简历：陈雨桐（cc50dcfb-aeba-41a3-989a-9e485aa2fc3f）
- 多轮通过 `session_group: chen_m3` 共享 SessionMemory

---

## 二、测试链路发现的 Bug

### 🚨 P0-1：kb_retrieve 工具 `settings` 未定义
**位置**：`backend/app/core/tools.py` 第167-176行
- `settings` 在第167/169行被使用，但直到第176行才 `import`
- **影响**：所有RAG用例的 kb_retrieve 全部失败
- **状态**：✅ 已修复（import 提前到函数开头）

### 🚨 P0-2：run_eval_v3.py `UnboundLocalError`
**位置**：`backend/eval/run_eval_v3.py` 第972行
- `debug` 变量先使用再定义，HTTP多轮case直接崩溃
- **状态**：✅ 已修复（定义提前）

### ⚠️ P1：test_dataset.jsonl 编码损坏
- line 31 存在 UTF-8 损坏 + JSON 引号未转义
- **状态**：✅ 已修复（用备份恢复）

---

## 三、关键纠正（基于用户反馈）

### 3.1 Judge 已包含忠实度与答案相关性

`judge_postprocess.py` 的 `JUDGE_SYSTEM_PROMPT` 中已经明确定义了：
- **第11维 `faithfulness`（忠实度）**：评判回复中的事实声明是否在检索证据中有支撑
- **第12维 `answer_relevance`（答案相关性）**：评判回复是否直接回答了用户问题

代码中对非RAG任务自动给满分，RAG任务基于实际检索证据评判。这是**已经实现的标准**。

### 3.2 澄清场景不走 Judge 是正确的

用户明确指出：
> "既然意图是 clarification，那么会在意图识别时就被卡住，不会进行之后的规划等剩余链路，理应是不走 judge 的"

这是完全正确的。`_run_v2_judge` 中对 clarification 的提前 return 是**正确行为**，不是 bug。因为：
- clarification 场景没有 task_graph（未进入 Planner）
- 没有工具执行（无 tool_execution 可评判）
- 没有最终回复（只有澄清问题，无 answer_relevance / faithfulness 可评判）

**之前的审计报告将此误判为"重大缺陷"，现已纠正。**

---

## 四、已完成的指标计算改进

### 4.1 compute_metrics 现在分离 clarification 用例

`run_eval_v3.py` 的指标计算逻辑已修改：

| 指标 | 修改前 | 修改后 |
|------|--------|--------|
| task_success_rate | 分母=全部用例 | **分母=仅常规用例** |
| tool_primary_hit_rate | 分母=全部用例 | **分母=仅常规用例** |
| process_quality_rate | 分母=全部用例 | **分母=仅常规用例** |
| KB异常率 | 分母=全部用例 | **分母=仅常规用例** |
| clarification 统计 | 只有 count | **新增：正确触发/漏触发/误触发** |

### 4.2 报告中新增【澄清边界用例统计】板块

```
【澄清边界用例统计（单独说明）】
  总数:              X
  正确触发:          Y (gold=clarification 且系统触发)
  漏触发:            Z (gold=clarification 但系统未触发)
  误触发:            W (gold≠clarification 但系统触发)
  识别准确率:        Y/X %
```

### 4.3 批次统计也同步分离

`_compute_batch_metrics` 同样排除 clarification 用例后再计算 success_rate / tool_primary_rate 等常规指标。

---

## 五、产品/算法层面的问题

### 🔍 第二轮"这个岗"未触发澄清

**实际行为**：
- Turn 1 explore 执行后，系统将 `position=AI产品实习岗` 写入了 `global_slots`
- Turn 2 "这个岗需要出差吗" 被 QueryRewrite 消解为具体岗位
- 意图识别为 verify → 被校准器改为 chat
- **needs_clarification=False**，未触发澄清

**Judge 评分**（eval_chen_20 作为常规用例被 judge 评估）：
- `intent_accuracy=0`（应为 clarification，识别为 chat）
- `tool_correctness=0`（expected_tools 为空，实际执行了 kb_retrieve/qa_synthesize/external_search）
- **整体判定：不通过**

**根因**：
- `global_slots` 中的 position 槽位在 explore 后没有被正确管理（应记录推荐列表而非单一岗位）
- QueryRewrite 对"这个岗"的指代消解过于激进

---

## 六、修复清单

| # | 文件 | 修改内容 | 状态 |
|---|------|---------|------|
| 1 | `backend/app/core/tools.py` | settings import 提前，修复 kb_retrieve 全局失败 | ✅ |
| 2 | `backend/eval/run_eval_v3.py` | debug 变量定义提前，修复 HTTP 调用崩溃 | ✅ |
| 3 | `backend/eval/test_dataset.jsonl` | 恢复编码 + 追加 chen_m3 组 2 条用例 | ✅ |
| 4 | `backend/eval/run_eval_v3.py` | compute_metrics 分离 clarification 用例，单独统计 | ✅ |
| 5 | `backend/eval/run_eval_v3.py` | print_report 新增【澄清边界用例统计】板块 | ✅ |
| 6 | `backend/eval/run_eval_v3.py` | _compute_batch_metrics 排除 clarification 后计算 | ✅ |

---

## 七、新增测试用例

已添加到 `backend/eval/test_dataset.jsonl`：

```json
{"session_id": "eval_chen_19", "resume_id": "cc50dcfb-aeba-41a3-989a-9e485aa2fc3f", "message": "帮我推荐几个适合我的AI产品实习岗", "eval_context": {"gold_intents": ["explore"], "gold_slots": {}, "relevance_score": 0, "expected_tools": ["kb_retrieve", "global_rank"], "scenario": "多轮_岗位推荐_第一轮", "notes": "推荐多个AI产品实习岗，为第二轮澄清做铺垫"}, "session_group": "chen_m3"}

{"session_id": "eval_chen_20", "resume_id": "cc50dcfb-aeba-41a3-989a-9e485aa2fc3f", "message": "这个岗需要出差吗", "eval_context": {"gold_intents": ["clarification"], "gold_slots": {}, "relevance_score": 0, "expected_tools": [], "scenario": "多轮_第二轮澄清_指代不清", "notes": "第一轮推荐了多个岗位，\"这个岗\"指代不清，应触发needs_clarification=true"}, "session_group": "chen_m3"}
```
