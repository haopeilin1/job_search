# v2 ReAct Agent 评估体系文档

> 本文档描述 job_search 项目 v2 版本的完整评估体系，包括评测架构、指标体系、数据集、脚本和使用方式。

---

## 一、评估架构（三层）

```
┌─────────────────────────────────────────────────────────────────────┐
│  L1 组件级白盒评测（Component Evaluation）                          │
│  直接调用 QueryRewrite / IntentRecognition / Planner / Executor     │
│  收集完整中间状态，定位问题根因                                      │
├─────────────────────────────────────────────────────────────────────┤
│  L2 端到端链路评测（End-to-End Evaluation）                         │
│  复用核心 test cases，运行完整 v2 链路                               │
│  评估整体成功率、意图匹配、工具链匹配、延迟、成本                    │
├─────────────────────────────────────────────────────────────────────┤
│  L3 稳定性与成本评测（Stability & Cost Evaluation）                 │
│  同一任务重复运行 N 次                                               │
│  统计成功率、结果一致性（Jaccard）、延迟变异系数（CV）               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 二、评测基础设施

```
backend/eval/
├── README.md                    # 测试集说明
├── test_dataset.jsonl           # 核心评测数据集
├── test_resumes.json            # 3份测试简历 + session映射
├── run_eval.py                  # HTTP API 黑盒评测脚本
├── metrics_report.py            # Telemetry 指标报告
├── run_v2_eval.py               # v2 白盒组件级评测脚本（主脚本）
├── v2_eval_report_*.json        # 评测结果JSON报告
└── EVAL_SYSTEM.md               # 本文档
```

### 2.1 核心评测脚本：`run_v2_eval.py`

**评测方式**：直接导入 v2 各组件内部函数，不走 HTTP API  
**核心能力**：
- 收集 QueryRewrite / IntentRecognition / Planner / Executor 完整输入输出
- Monkey-patch LLMClient，精确统计每次 LLM 调用的 token / 延迟 / 成本
- 支持稳定性测试（`--stability N`，同一 case 跑 N 次）
- 输出结构化 JSON 报告 + 控制台摘要

```bash
cd backend && python eval/run_v2_eval.py                    # 跑全部
cd backend && python eval/run_v2_eval.py --batch A          # 只跑 A 批次
cd backend && python eval/run_v2_eval.py --case eval_a_t04  # 单条调试
cd backend && python eval/run_v2_eval.py --stability 3      # 稳定性测试（每条3次）
cd backend && python eval/run_v2_eval.py --output eval/v2_report.json
```

---

## 三、指标体系

### 3.1 结果指标（Outcome Metrics）

| 指标名 | 定义 | 计算方式 | 用途 |
|--------|------|---------|------|
| `task_success_rate` | 任务成功率 | 无异常且关键任务成功 / 总 case 数 | 衡量系统整体可用性 |
| `exception_rate` | 异常率 | 抛未捕获异常 / 总 case 数 | 衡量系统稳定性 |
| `clarification_rate` | 澄清触发率 | `needs_clarification=True` / 总 case 数 | 衡量意图识别过度保守程度 |
| `avg_latency_ms` | 平均端到端延迟 | 所有 case 延迟的算术平均 | 衡量响应速度 |
| `p50_latency_ms` | P50 延迟 | 排序后第 50 百分位 | 衡量典型响应速度 |
| `p99_latency_ms` | P99 延迟 | 排序后第 99 百分位 | 衡量最差情况响应速度 |
| `total_tool_calls` | 总工具调用次数 | 所有成功执行的工具数之和 | 衡量工具使用频率 |
| `avg_tool_calls_per_case` | 平均每例工具调用 | 总工具调用 / 总 case 数 | 衡量任务复杂度 |
| `total_llm_api_calls` | 总 LLM API 调用次数 | monkey-patch 统计的所有 LLM 调用 | 衡量 LLM 依赖程度 |
| `total_tokens` | 总 Token 消耗 | prompt_tokens + completion_tokens 之和 | 衡量资源消耗 |
| `estimated_total_cost_usd` | 预估总成本 | 按 DashScope 单价（$ / 1K tokens）估算 | 衡量运行成本 |
| `avg_cost_per_case_usd` | 平均每例成本 | 总成本 / 总 case 数 | 衡量单任务成本 |

### 3.2 过程指标（Process Metrics）

#### 3.2.1 意图识别指标

| 指标名 | 定义 | 计算方式 |
|--------|------|---------|
| `intent_precision` | 意图识别精确率 | `predicted_intents ∩ gold_intents / predicted_intents` |
| `intent_recall` | 意图识别召回率 | `predicted_intents ∩ gold_intents / gold_intents` |
| `intent_f1` | 意图识别 F1 值 | `2 * precision * recall / (precision + recall)` |

**标签映射**（测试集标签 → 系统标签）：
| 测试集 | 系统 |
|--------|------|
| `explore` | `position_explore` |
| `assess` | `match_assess` |
| `verify` | `attribute_verify` |
| `prepare` | `interview_prepare` |
| `chat` | `general_chat` |
| `clarification` | `clarification` |

> 如果系统触发 `needs_clarification=True`，`predicted_intents` 自动包含 `clarification`。

#### 3.2.2 工具选择指标

| 指标名 | 定义 | 计算方式 |
|--------|------|---------|
| `tool_selection_precision` | 工具选择精确率 | `executed_tools ∩ expected_tools / executed_tools` |
| `tool_selection_recall` | 工具选择召回率 | `executed_tools ∩ expected_tools / expected_tools` |
| `tool_selection_f1` | 工具选择 F1 值 | `2 * precision * recall / (precision + recall)` |

> `executed_tools`：Executor 执行后 `status="success"` 的工具列表  
> `expected_tools`：eval_context 中标注的期望工具列表

#### 3.2.3 工具执行指标

| 指标名 | 定义 | 计算方式 |
|--------|------|---------|
| `tool_execution_success_rate` | 工具执行成功率 | `status="success" 的工具数 / 总工具数` |

#### 3.2.4 规划指标

| 指标名 | 定义 | 计算方式 |
|--------|------|---------|
| `plan_validity_rate` | 规划合法性 | `DAG 无环且依赖可达 / 总 case 数` |
| `plan_optimality_rate` | 规划最优性 | `expected_tools ⊆ executed_tools / 总 case 数` |
| `replan_trigger_rate` | Replan 触发率 | `global_status 为 needs_replan / 总 case 数` |

#### 3.2.5 Fallback 指标

| 指标名 | 定义 | 计算方式 |
|--------|------|---------|
| `query_rewrite_fallback_rate` | QueryRewrite fallback 率 | LLM rewrite 失败 fallback 到规则 / 总 case 数 |
| `intent_l2_fallback_rate` | Intent L2 fallback 率 | L2 LLM 调用失败 fallback 到 L1 / 总 case 数 |
| `intent_l3_trigger_rate` | Intent L3 触发率 | 进入 L3 仲裁 / 总 case 数 |
| `planner_fallback_rate` | Planner fallback 率 | LLM 规划失败 fallback 到规则图 / 总 case 数 |

### 3.3 组件延迟指标（Latency Metrics）

| 指标名 | 定义 | 用途 |
|--------|------|------|
| `query_rewrite_avg_ms` | QueryRewrite 平均耗时 | 衡量改写模块速度 |
| `intent_recognition_avg_ms` | IntentRecognition（L1+L2+L3）平均耗时 | 衡量意图识别速度 |
| `planner_avg_ms` | Planner 动态规划平均耗时 | 衡量规划模块速度 |
| `executor_avg_ms` | Executor 执行平均耗时 | 衡量执行模块速度 |

### 3.4 稳定性指标（Stability Metrics）

通过 `--stability N` 启用：

| 指标名 | 定义 | 用途 |
|--------|------|------|
| `stability_score` | N 次运行中成功次数 / N | 衡量任务可靠性 |
| `result_consistency` | N 次运行 `executed_tools` 集合的 Jaccard 相似度均值 | 衡量结果可复现性 |
| `latency_cv` | 延迟变异系数 = std / mean | 衡量延迟稳定性 |
| `min_latency_ms` | N 次中最小延迟 | 衡量最佳情况 |
| `max_latency_ms` | N 次中最大延迟 | 衡量最差情况 |

### 3.5 质量指标（Quality Metrics）— 预留待实现

| 指标名 | 定义 | 实现方式 | 状态 |
|--------|------|---------|------|
| `response_relevance` | 最终回复与用户问题的相关度 | LLM-as-judge | 待实现 |
| `factual_accuracy` | 回复事实与检索 chunks 的一致性 | 引用对齐检查 | 待实现 |
| `human_preference_score` | 人类偏好评分 | 人工标注 | 待实现 |

---

## 四、评测数据集

### 4.1 数据集文件

| 文件 | 说明 |
|------|------|
| `test_dataset.jsonl` | 核心评测用例，每行一个 JSON 对象 |
| `test_resumes.json` | 3 份测试简历（AI算法/产品、Java后端、空简历）+ session→resume 映射 |

### 4.2 用例字段规范

```json
{
  "session_id": "eval_a_t01",
  "message": "用户消息",
  "eval_context": {
    "gold_intents": ["explore"],
    "gold_slots": {},
    "expected_tools": ["kb_retrieve", "global_rank"],
    "relevance_score": 0,
    "scenario": "场景描述",
    "notes": "备注"
  }
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `session_id` | string | 是 | 唯一标识，格式 `eval_{批次}_{序号}` |
| `message` | string | 是 | 用户输入消息 |
| `eval_context.gold_intents` | string[] | 是 | 期望意图列表（explore/assess/verify/prepare/chat/clarification） |
| `eval_context.gold_slots` | object | 否 | 期望提取的槽位 |
| `eval_context.expected_tools` | string[] | 是 | 期望执行的工具列表 |
| `eval_context.relevance_score` | number | 否 | 相关性评分（0-1） |
| `eval_context.scenario` | string | 是 | 场景描述，便于定位问题 |
| `eval_context.notes` | string | 否 | 特殊说明 |

### 4.3 简历映射规则

`test_resumes.json` 中 `session_resume_map` 定义了 case→resume 的映射：
- `eval_resume_ai`：AI算法/产品背景（Python/LangChain/RAG/Agent，3年本科）
- `eval_resume_java`：Java后端背景（Spring Boot/MySQL/Redis，2年大专）
- `eval_resume_empty`：空简历

---

## 五、评测报告结构

运行 `run_v2_eval.py` 后生成的 JSON 报告：

```json
{
  "metrics": {
    "outcome": { /* 结果指标 */ },
    "process": { /* 过程指标 */ },
    "latency": { /* 组件延迟 */ },
    "batch": { /* 批次统计 */ }
  },
  "cases": [
    {
      "case_id": "eval_a_t01",
      "batch": "a",
      "message": "...",
      "gold_intents": ["explore"],
      "expected_tools": ["kb_retrieve", "global_rank"],
      "task_success": true,
      "has_exception": false,
      "e2e_latency_ms": 11330,
      "rewrite": {
        "component": "query_rewrite",
        "success": true,
        "latency_ms": 0,
        "output": { "rewritten_query": "...", "follow_up_type": "none" }
      },
      "intent": {
        "component": "intent_recognition",
        "success": true,
        "latency_ms": 6902,
        "output": { "demands": [...], "needs_clarification": false }
      },
      "planner": {
        "component": "planner",
        "success": true,
        "latency_ms": 12614,
        "output": { "task_count": 4, "parallel_groups": 3 }
      },
      "executor": {
        "component": "executor",
        "success": true,
        "latency_ms": 4559,
        "output": { "global_status": "success", "success_count": 3 }
      },
      "intent_result": { /* 意图识别详细结果 */ },
      "task_graph": { /* TaskGraph 完整快照 */ },
      "executed_tools": ["kb_retrieve", "global_rank"],
      "failed_tools": [],
      "llm_summary": {
        "total_calls": 4,
        "total_tokens": 2700,
        "estimated_cost_usd": 0.003378
      }
    }
  ],
  "stability": [ /* 仅 --stability N 时存在 */ ]
}
```

---

## 六、评测流程

```bash
# Step 1: 确认配置
cat backend/.env | grep API_KEY

# Step 2: 跑 A 批次快速验证（约 3-5 分钟）
cd backend && python eval/run_v2_eval.py --batch A

# Step 3: 查看控制台报告
# 报告包含：结果指标、过程指标、组件延迟、批次统计

# Step 4: 查看 JSON 详细结果
cat backend/eval/v2_eval_report_*.json | python -m json.tool

# Step 5: 单条调试（定位特定 case 的问题）
cd backend && python eval/run_v2_eval.py --case eval_a_t04

# Step 6: 稳定性测试（检测偶发失败和结果波动）
cd backend && python eval/run_v2_eval.py --batch A --stability 3
```

---

## 七、当前状态与已知问题

### 7.1 A 批次实测基线（12 条，修复前）

| 指标 | 数值 | 说明 |
|------|------|------|
| 任务成功率 | 75.0% (9/12) | match_analyze / interview_gen 超时导致 |
| 澄清触发率 | 41.67% | L1 规则层过度保守 |
| 意图 F1 | 50.0% | 部分 case 意图不匹配 |
| 工具选择 F1 | 36.67% | 部分 case 澄清未进 Planner |
| 工具执行成功率 | 73.68% | match_analyze 超时失败 |
| 平均延迟 | 16,356 ms | 含 Planner LLM 规划 10-14s |
| 每例成本 | $0.006 | 约 4 次 LLM 调用/例 |

### 7.2 已知问题

| 问题 | 严重程度 | 状态 |
|------|---------|------|
| 澄清触发率过高 | P0 | 已修复（L2 可覆盖 L1 澄清决策） |
| match_analyze / interview_gen 超时 | P0 | 已修复（截断 2000 字 + 45s 超时 + 连接池） |
| QueryRewrite 全部 fallback 到规则 | P1 | 已修复（20s 超时 + 连接池复用） |
| 缺少回复质量评估 | P2 | 待实现（LLM-as-judge） |
| 缺少回归测试流水线 | P2 | 待实现 |

---

*文档版本：v1.0*  
*最后更新：2026-04-28*
