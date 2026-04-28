# 评估测试集

## 文件说明

| 文件 | 说明 |
|------|------|
| `test_dataset.jsonl` | 46条测试用例（40条标准 + 6条记忆机制），每行一个JSON对象 |
| `test_resumes.json` | 3份测试简历（AI背景、Java背景、空简历）及session映射 |
| `run_eval.py` | 评估执行脚本 |
| `metrics_report.py` | 指标计算与报告生成 |

## 测试集结构

### 批次划分

| 批次 | 用例数 | 说明 |
|------|--------|------|
| A | 12条 | 单意图基础覆盖（explore/assess/verify/prepare/chat） |
| B | 8条 | 多意图组合（explore+assess/verify/prepare等） |
| C | 8条 | 多轮对话（4个session×2轮，测试槽位继承/意图切换/follow-up/澄清） |
| D | 6条 | 边界与异常（无命中/无意义输入/空消息/大范围请求） |
| E | 6条 | 双层召回专项验证（技能匹配/年限过滤/学历过滤/领域匹配/技能不匹配） |
| F | 6条 | 记忆机制专项（6轮对话，测试3轮前记忆留存） |

### 用例字段

```json
{
  "session_id": "eval_a_t01",
  "message": "用户消息",
  "eval_context": {
    "gold_intents": ["explore"],
    "gold_slots": {},
    "relevance_score": 0,
    "expected_tools": ["kb_retrieve", "global_rank"],
    "scenario": "场景描述",
    "notes": "备注"
  }
}
```

### 简历映射规则

- `eval_a_t01~t03`, `eval_d_t02~t04`: 空简历（无简历场景）
- `eval_c3_*`: Java后端简历（测试补充信息后重新探索）
- `eval_e_t04`: 空简历（测试无简历时粗筛行为）
- 其余：AI算法/产品简历（3年本科，Python/LangChain/RAG/Agent）

## 关键测试假设

1. **知识库命中**：A04/A06/A07/A09 期望命中5条测试JD中的特定条目
2. **上下文引用**：A05/B05/C1_t02/C2_t02 期望正确解析"这个"/"第一个"等指代
3. **粗筛行为**：
   - E01：AI技能简历 → 粗筛优先保留AI相关JD（字节/百度/美团）
   - E02：3年经验 → 百度（5年+）应被降权
   - E03：本科学历 → 百度（硕士）应被降权
   - E05：AI简历搜Java → 技能交集极少，fallback到hybrid_score
4. **记忆机制**：F06期望能回忆起F04中"更看重技术成长空间"的偏好
