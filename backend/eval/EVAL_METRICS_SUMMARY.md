# 求职雷达 Agent 评测系统 — 完整指标体系

> 本文档汇总 job_search 项目所有评测脚本、指标定义与计算方式。  
> 最后更新：2026-06-04

---

## 一、评测架构总览（三层）

```
┌─────────────────────────────────────────────────────────────────────┐
│  L1 组件级白盒评测（Component Evaluation）                          │
│  直接导入 QueryRewrite / IntentRecognition / Planner / Executor     │
│  收集完整中间状态，定位问题根因                                      │
│  脚本：run_v2_eval.py                                               │
├─────────────────────────────────────────────────────────────────────┤
│  L2 端到端黑盒评测（End-to-End Evaluation）                         │
│  通过 HTTP / SSE 调用完整链路，模拟真实用户体验                      │
│  评估整体成功率、意图匹配、工具链匹配、延迟、成本、Judge 评分        │
│  脚本：batch_eval_runner.py（主流量产）                             │
│       batch_http_eval_v3.py（全链路+v3.5 Judge）                    │
│       run_full_eval_v3.py（三轮稳定性+动态 Judge）                  │
│       run_eval_v3.py（严格对齐白盒评测）                            │
├─────────────────────────────────────────────────────────────────────┤
│  L3 稳定性与成本评测（Stability & Cost Evaluation）                 │
│  同一任务重复运行 N 次                                               │
│  统计成功率、结果一致性（Jaccard）、延迟变异系数（CV）               │
│  各脚本均支持多轮运行或 --stability N 参数                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 二、评测脚本对照表

| 脚本 | 类型 | 调用方式 | Judge 体系 | 输出位置 | 主要用途 |
|------|------|---------|-----------|---------|---------|
| `batch_eval_runner.py` | L2 黑盒 | SSE 流式 HTTP | ❌ 无（基于规则+字段匹配） | `eval/results/run{N}/` | **主流量产评测**（跑 run4 的脚本） |
| `batch_http_eval_v3.py` | L2 黑盒 | HTTP `/api/v1/chat` | ✅ v3.5 Judge | `eval/batch_results/` | 批量全链路 + 12 维评分 |
| `run_full_eval_v3.py` | L2 黑盒 | SSE 流式 HTTP | ✅ 动态 LLM Judge | `eval/results/round_{N}.json` | 三轮稳定性 + Judge 完成度 |
| `run_eval_v3.py` | L1+L2 混合 | 白盒组件导入 + HTTP | ✅ v3.5 Judge | `eval/v3_report_*.json` | 严格对齐，monkey-patch 统计 |
| `full_chain_test_v2.py` | L2 黑盒 | HTTP `/api/v1/chat` | ✅ v3.5 Judge | `eval/full_chain_v3_{case_id}_*.json` | **单条 case 调试** |
| `run_v2_eval.py` | L1 白盒 | 组件内部函数 | ❌ 无 | `eval/v2_eval_report_*.json` | v2 组件级评测 |
| `metrics_report.py` | L2 黑盒 | Telemetry 日志 | ❌ 无 | 控制台 + `.report.json` | 基于 `logs/events.jsonl` 的离线报告 |

---

## 三、v3.5 Judge 多维度评分体系（0-5 分制）

> 用于：`batch_http_eval_v3.py`、`run_eval_v3.py`、`full_chain_test_v2.py`、`judge_postprocess.py`

### 3.1 关键维度（决定 resolved 通过与否）

| # | 维度名 | 定义 | 通过阈值 |
|---|--------|------|---------|
| 1 | `intent_accuracy` | 意图识别准确性。pred_intents 与 gold_intents 是否一致；多意图是否全部命中；是否错误触发/遗漏澄清 | ≥ 3 |
| 2 | `slot_accuracy` | 槽位提取准确性。company/position/attributes 等关键槽位是否正确；是否从上下文正确补全 | ≥ 3 |
| 3 | `tool_correctness` | 工具调用正确性。是否调用 expected_tools 中的必要工具；是否遗漏关键工具；是否调用多余工具 | ≥ 3 |
| 4 | `tool_execution` | 工具执行效果。检索结果是否相关；match_analyze 是否基于简历+JD 分析；interview_gen 是否针对具体岗位；失败后是否有重试/恢复 | ≥ 3 |
| 5 | `response_accuracy` | 回复内容准确性。是否基于正确证据；有无编造/张冠李戴；有无声称"缺少简历"但实际已传入 | ≥ 3 |
| 6 | `response_completeness` | 回复完整性。是否覆盖用户核心需求：VERIFY 答属性、ASSESS 给分析、EXPLORE 给岗位、PREPARE 给面试题 | ≥ 3 |

**resolved 通过标准**：以上 6 个关键维度必须同时 ≥ 3 分，且未触发否决项。

### 3.2 辅助维度（参考，不直接否决）

| # | 维度名 | 定义 |
|---|--------|------|
| 7 | `citation_quality` | 引用标注质量。检索类任务是否有引用标注；标注是否准确对应来源；是否编造引用标记 |
| 8 | `coherence` | 连贯性。多轮对话是否保持上下文连贯；是否正确处理指代消解 |
| 9 | `tone` | 语气。是否专业、友好、自然；是否过于机械/模板化 |
| 10 | `efficiency` | 效率。延迟是否合理；Token 消耗是否合理；是否有多余 LLM 调用或重试 |

### 3.3 RAG 专属维度（仅检索类任务评判，非检索类可给满分）

| # | 维度名 | 定义 |
|---|--------|------|
| 11 | `faithfulness` | 忠实度。回复中每个事实声明是否都能在检索到的 chunks 中找到支撑；只看"是否说假话" |
| 12 | `answer_relevance` | 答案相关性。回复是否直接回答了用户 query；是否答非所问 |

### 3.4 规则否决项（任一触发，resolved 强制 false）

| 规则 ID | 触发条件 | 否决强度 |
|---------|---------|---------|
| `empty_reply` | 回复为空或 < 30 字 | **强制否决** (veto=True) |
| `fake_no_resume` | 系统声称"缺少简历"但实际已提供简历 | **强制否决** (veto=True) |
| `wrong_company_jd` | 检索到完全错误的公司/岗位并作为回答依据 | **强制否决** (veto=True) |
| `fabricated_citation` | 编造引用标记 | **强制否决** (veto=True) |
| `tool_error_empty` | 工具执行异常导致空回复或极短回复 | **强制否决** (veto=True) |
| `response_accuracy ≤ 1` | 严重编造或事实错误 | **强制否决** (veto=True) |
| `verify_no_value` | VERIFY 意图但回复无具体属性值且过短 | 规则兜底 |
| `assess_no_analysis` | ASSESS 意图但回复无匹配分析且过短 | 规则兜底 |
| `explore_no_jobs` | EXPLORE 意图但回复无具体岗位推荐且过短 | 规则兜底 |
| `prepare_no_content` | PREPARE 意图但回复无面试题/建议且过短 | 规则兜底 |
| `chat_too_short` | CHAT 意图但回复过短 | 规则兜底 |
| `chat_has_error` | CHAT 意图但回复包含错误信息 | 规则兜底 |

---

## 四、batch_eval_runner.py 指标体系（主流量产评测）

> 该脚本通过 SSE 流式端点 `/api/v1/chat/stream` 跑全部 case，是 run4 的实际执行脚本。

### 4.1 结果指标

| 指标名 | 计算方式 | 说明 |
|--------|---------|------|
| `total_cases` | 参与统计的 case 总数 | 默认排除 clarification + boundary case（共 6 条） |
| `success_cases` | status == "success" 的数量 | HTTP 请求成功完成 |
| `error_cases` | status == "error" 的数量 | HTTP 请求抛异常 |
| `success_rate` | success_cases / total_cases | 端到端成功率 |

### 4.2 意图指标

| 指标名 | 计算方式 | 说明 |
|--------|---------|------|
| `intent_match_rate` | intent_match 为 true 的 success case / 参与统计的 success case | 排除 clarification case；单意图匹配即算命中 |
| `intent_breakdown` | 按 gold_intent 分组统计 total / match / success | explore/assess/verify/prepare/chat/manage 六类 |

### 4.3 工具指标

| 指标名 | 计算方式 | 说明 |
|--------|---------|------|
| `avg_tool_match_rate` | 所有 success case 的 tool_match_rate 算术平均 | 期望工具中被调用的比例（模糊匹配） |
| `full_tool_match_rate` | tool_match_rate >= 1.0 的 case / success_cases | 所有期望工具都被调用的比例 |
| `tool_execution_success_rate` | 所有 case 的 tool_execution_success_rate 算术平均 | 实际调用的工具中 status==✅ 的比例 |
| `tool_correct_rate` | 所有 case 的 tool_correct_rate 算术平均 | 期望工具中被调用且执行成功的比例 |

### 4.4 回复指标

| 指标名 | 计算方式 | 说明 |
|--------|---------|------|
| `reply_completion_rate` | has_reply 为 true 的 success case / success_cases | reply 长度 > 50 字即算有回复 |

### 4.5 延迟指标

| 指标名 | 计算方式 | 说明 |
|--------|---------|------|
| `avg_ttfb_sec` | 所有 success case 的 ttfb 算术平均 | Time To First Byte（首字节时间） |
| `avg_total_latency_sec` | 所有 success case 的 total_latency 算术平均 | 端到端总延迟 |
| `median_total_latency_sec` | total_latency 的中位数 | |
| `max_total_latency_sec` | total_latency 的最大值 | |
| `min_total_latency_sec` | ttfb 的最小值 | |

### 4.6 Token 消耗指标（真实 API Usage）

| 指标名 | 计算方式 | 说明 |
|--------|---------|------|
| `total_prompt_tokens` | 所有 success case 的 prompt_tokens 之和 | SSE done 事件中 usage.prompt_tokens |
| `total_completion_tokens` | 所有 success case 的 completion_tokens 之和 | SSE done 事件中 usage.completion_tokens |
| `total_tokens` | 所有 success case 的 total_tokens 之和 | |
| `avg_prompt_tokens` | total_prompt_tokens / success_cases | |
| `avg_completion_tokens` | total_completion_tokens / success_cases | |
| `avg_total_tokens` | total_tokens / success_cases | |

### 4.7 场景指标

| 指标名 | 计算方式 | 说明 |
|--------|---------|------|
| `scenario_breakdown` | 按 scenario 分组统计 total / success / error | 如"有简历_岗位探索_AI产品"等 40+ 场景 |

### 4.8 特殊 Case 定义

| 类型 | Case ID | 处理规则 |
|------|---------|---------|
| Clarification | `eval_chen_13`, `eval_li_14`, `eval_gen_03`, `eval_gen_06` | 不计入 intent_match_rate；但统计是否触发 clarification |
| Boundary | `eval_gen_02`, `eval_gen_03` | 不计入所有指标；额外生成边界报告 |

---

## 五、run_full_eval_v3.py 指标体系（三轮稳定性）

| 指标名 | 计算方式 | 说明 |
|--------|---------|------|
| `intent_exact` | pred_intent 集合 == gold_intents 集合 | 多意图需全部匹配 |
| `intent_f1` | 多意图 F1 = 2·P·R/(P+R) | precision = TP/(TP+FP), recall = TP/(TP+FN) |
| `tool_f1` | 工具选择 F1 = 2·P·R/(P+R) | 同上 |
| `judge_resolved` | LLM Judge 判定 resolved=true | 动态标准（见 §3） |
| `latency` | 请求开始到结束的总耗时 | |
| `ttfb` | 请求开始到首字节返回的时间 | |

**稳定性分析**：
- 三轮中 intent_exact 或 judge_resolved 不一致的 case 被标记为"不稳定 case"
- 生成 `stability_report.md` 展示每轮指标和不稳定 case 明细

---

## 六、batch_http_eval_v3.py 指标体系（全链路+v3.5 Judge）

| 指标名 | 计算方式 | 说明 |
|--------|---------|------|
| `resolved_rate` | judge_resolved=true 的 case / total_cases | 任务完成率（核心指标） |
| `dimension_averages` | 各 Judge 维度分数的算术平均 | 12 维平均分 |
| `intent_accuracy.rate` | intent_accuracy ≥ 3 的 case / 参与统计 case | 排除 clarification + boundary |
| `clarification_cases` | 4 条 clarification case 的触发情况 | 统计是否触发 clarification |
| `boundary_cases` | 2 条 boundary case 的处理情况 | 额外观察 |
| `failed_cases` | HTTP 错误或超时的 case 列表 | |

---

## 七、run_eval_v3.py 指标体系（严格对齐白盒评测）

### 7.1 追踪能力（Monkey-patch）

| 追踪项 | 说明 |
|--------|------|
| LLM 调用追踪 | 每次 chat/generate 的 model、layer、prompt_tokens、completion_tokens、latency_ms |
| Embedding 调用追踪 | texts_count、total_chars、latency_ms |
| Reranker 调用追踪 | 同上 |

### 7.2 核心指标

| 指标名 | 计算方式 |
|--------|---------|
| 任务完成率 | Judge resolved=true / total |
| 意图精确匹配率 | pred_intents 集合 == gold_intents 集合 |
| 工具链精确匹配率 | executed_tools 集合 == expected_tools 集合 |
| 端到端平均延迟 | 所有 case latency 算术平均 |
| 平均 LLM 调用次数 | total_llm_calls / total_cases |
| 平均 Token 消耗 | total_tokens / total_cases |
| 平均成本 | total_cost_usd / total_cases |

---

## 八、run_v2_eval.py / EVAL_SYSTEM.md 指标体系（v2 白盒组件级）

### 8.1 结果指标（Outcome Metrics）

| 指标名 | 定义 | 用途 |
|--------|------|------|
| `task_success_rate` | 无异常且关键任务成功 / 总 case 数 | 整体可用性 |
| `exception_rate` | 抛未捕获异常 / 总 case 数 | 稳定性 |
| `clarification_rate` | needs_clarification=True / 总 case 数 | 过度保守程度 |
| `avg_latency_ms` | 所有 case 延迟算术平均 | 响应速度 |
| `p50_latency_ms` | 延迟 P50 | 典型速度 |
| `p99_latency_ms` | 延迟 P99 | 最差情况 |
| `total_tool_calls` | 成功执行的工具数之和 | 工具使用频率 |
| `total_llm_api_calls` | monkey-patch 统计的所有 LLM 调用 | LLM 依赖程度 |
| `total_tokens` | prompt + completion tokens 之和 | 资源消耗 |
| `estimated_total_cost_usd` | 按 DashScope 单价估算 | 运行成本 |

### 8.2 过程指标（Process Metrics）

| 指标名 | 计算方式 |
|--------|---------|
| `intent_precision` | pred ∩ gold / pred |
| `intent_recall` | pred ∩ gold / gold |
| `intent_f1` | 2·P·R/(P+R) |
| `tool_selection_precision` | executed ∩ expected / executed |
| `tool_selection_recall` | executed ∩ expected / expected |
| `tool_selection_f1` | 2·P·R/(P+R) |
| `tool_execution_success_rate` | status="success" 的工具数 / 总工具数 |
| `plan_validity_rate` | DAG 无环且依赖可达 / 总 case 数 |
| `plan_optimality_rate` | expected_tools ⊆ executed_tools / 总 case 数 |
| `replan_trigger_rate` | global_status 为 needs_replan / 总 case 数 |

### 8.3 Fallback 指标

| 指标名 | 定义 |
|--------|------|
| `query_rewrite_fallback_rate` | LLM rewrite 失败 fallback 到规则 / 总 case 数 |
| `intent_l2_fallback_rate` | L2 LLM 调用失败 fallback 到 L1 / 总 case 数 |
| `intent_l3_trigger_rate` | 进入 L3 仲裁 / 总 case 数 |
| `planner_fallback_rate` | LLM 规划失败 fallback 到规则图 / 总 case 数 |

### 8.4 组件延迟指标

| 指标名 | 定义 |
|--------|------|
| `query_rewrite_avg_ms` | QueryRewrite 平均耗时 |
| `intent_recognition_avg_ms` | IntentRecognition（L1+L2+L3）平均耗时 |
| `planner_avg_ms` | Planner 动态规划平均耗时 |
| `executor_avg_ms` | Executor 执行平均耗时 |

### 8.5 稳定性指标（`--stability N`）

| 指标名 | 定义 |
|--------|------|
| `stability_score` | N 次运行中成功次数 / N |
| `result_consistency` | N 次 executed_tools 集合的 Jaccard 相似度均值 |
| `latency_cv` | 延迟变异系数 = std / mean |
| `min_latency_ms` | N 次中最小延迟 |
| `max_latency_ms` | N 次中最大延迟 |

### 8.6 质量指标（预留待实现）

| 指标名 | 定义 | 状态 |
|--------|------|------|
| `response_relevance` | 回复与用户问题的相关度（LLM-as-judge） | 待实现 |
| `factual_accuracy` | 回复事实与检索 chunks 的一致性 | 待实现 |
| `human_preference_score` | 人类偏好评分 | 待实现 |

---

## 九、metrics_report.py 指标体系（Telemetry 离线报告）

> 基于 `backend/logs/events.jsonl` 埋点数据生成。

### 9.1 API 层指标

| 指标名 | 计算方式 |
|--------|---------|
| `total` | 总 case 数 |
| `success` | status_code == 200 的数量 |
| `success_rate` | success / total |
| `avg_latency_ms` | latency_ms 算术平均 |
| `p99_latency_ms` | latency_ms P99 |
| `batch_success` | 按批次（session_id 前缀）统计成功率 |

### 9.2 意图准确率

| 指标名 | 计算方式 |
|--------|---------|
| `exact_accuracy` | gold == pred / total |
| `partial_accuracy` | gold ∩ pred ≠ ∅ / total |
| `missing_predictions` | pred 为空的数量 |

### 9.3 工具准确率

| 指标名 | 计算方式 |
|--------|---------|
| `tool_chain_accuracy` | expected_tools == actual_tools / total |
| `all_tools_success` | 所有工具 execution success / total |

### 9.4 核心运营指标

| 指标名 | 计算方式 |
|--------|---------|
| `retry_rate_approx` | 多轮 session 数 / 总 session 数 |
| `recovery_rate` | recovered 异常 / 总异常数 |
| `avg_cost_usd` | 单次 turn 平均成本 |
| `total_cost_usd` | 总成本 |
| `coarse_avg_filter_ratio` | 粗筛平均过滤比例 |
| `coarse_avg_input_jds` | 粗筛平均输入 JD 数 |
| `coarse_avg_output_jds` | 粗筛平均输出 JD 数 |

---

## 十、测试数据集规范

### 10.1 文件

| 文件 | 说明 |
|------|------|
| `test_dataset.jsonl` | 核心评测用例（58 条），每行一个 JSON |
| `test_resumes.json` | 3 份测试简历 + session→resume 映射 |

### 10.2 用例字段

```json
{
  "session_id": "eval_chen_01",
  "resume_id": "...",
  "message": "用户消息",
  "eval_context": {
    "gold_intents": ["explore"],
    "gold_slots": {},
    "expected_tools": ["kb_retrieve", "global_rank"],
    "relevance_score": 0,
    "scenario": "有简历_岗位探索_AI产品",
    "notes": "备注"
  },
  "session_group": null
}
```

### 10.3 意图标签映射

| 测试集标签 | 系统内部标签 |
|-----------|-------------|
| `explore` | `position_explore` |
| `assess` | `match_assess` |
| `verify` | `attribute_verify` |
| `prepare` | `interview_prepare` |
| `chat` | `general_chat` |
| `manage` | `resume_manage` |
| `clarification` | `clarification` |

---

## 十一、run4 实测基线（batch_eval_runner.py）

> 数据来自 `eval/results/_summary.json`（2026-06-03）

| 指标 | 数值 |
|------|------|
| 总用例数 | 58 |
| 实际执行 | 53（排除 5 条特殊 case） |
| 成功 | 51 |
| 失败 | 2（均为 `ClientPayloadError` 传输层错误） |
| **成功率** | **96.2%** |
| 意图匹配率 | 90.2% |
| 工具平均匹配率 | 75.2% |
| 工具完全匹配率 | 64.7% |
| 工具执行成功率 | 65.7% |
| 工具调用正确率 | 64.8% |
| 回复完成率 | 90.2% |
| 平均 TTFB | 0.01s |
| 平均总延迟 | 93.38s |
| 中位总延迟 | 98.79s |
| 最大延迟 | 188.04s |
| 平均 prompt tokens | 7,937.7 |
| 平均 completion tokens | 6,002.6 |
| 平均 total tokens | 13,940.3 |
| 总 prompt tokens | 404,821 |
| 总 completion tokens | 306,135 |
| 总 tokens | 710,956 |

---

## 十二、使用速查

```bash
# 1. 主流量产评测（SSE 流式，当前最常用）
cd backend && python eval/batch_eval_runner.py --runs 1

# 2. 单条 case 全链路调试 + v3.5 Judge
cd backend && python eval/full_chain_test_v2.py --case eval_chen_03

# 3. 批量 HTTP 全链路 + v3.5 Judge
cd backend && python eval/batch_http_eval_v3.py

# 4. 三轮稳定性测试 + 动态 Judge
cd backend && python eval/run_full_eval_v3.py

# 5. v2 白盒组件级评测
cd backend && python eval/run_v2_eval.py --batch A

# 6. Telemetry 离线报告
cd backend && python eval/metrics_report.py

# 7. 对已有 run 结果调用 v3.5 Judge 后处理
cd backend && python eval/judge_postprocess.py --run 4
```
