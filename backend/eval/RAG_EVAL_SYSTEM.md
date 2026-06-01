# RAG 检索评测体系文档

> 本文档描述 job_search 项目 RAG 检索层的独立评测体系，基于 `rag_test_dataset.jsonl` 黄金数据集，用于测试和优化检索策略（向量检索、BM25、融合排序、CrossEncoder 重排序、k值选择等）。

---

## 一、评测目标与范围

### 1.1 评测范围

仅评测 **检索层**（即 `kb_retrieve` 工具的输出），不包含：
- ❌ 生成质量（Faithfulness / Answer Relevancy）→ 端到端评测中测试
- ❌ 意图识别、Planner 编排 → Agent 链路评测中测试
- ❌ 匹配分析（match_analyze）→ 独立组件评测

### 1.2 评测目标

| 目标 | 说明 |
|------|------|
| 检索策略对比 | 向量 vs BM25 vs 混合召回的效果差异 |
| 权重调优 | 向量/BM25 融合权重的最优取值 |
| k值选择 | top_k 对精确率/召回率的 trade-off |
| Reranker 效果 | CrossEncoder 重排序前后的指标变化 |
| Chunk 策略对比 | semantic / fixed / recursive / section 四种切分策略 |
| 指代消解 & 多轮 | expand/clarify/switch 场景下的检索稳定性 |

---

## 二、黄金数据集（Golden Dataset）

### 2.1 文件位置

```
backend/eval/rag_test_dataset.jsonl
```

### 2.2 数据集规模

- **总条数**：43 条
- **覆盖类型**：
  - `single_jd`（单JD精确匹配）：9 条
  - `explore`（宽泛探索）：11 条
  - `verify`（属性查询）：7 条
  - `skill_explore`（技能导向探索）：6 条
  - `verify_expand`（多轮展开查询）：3 条
  - `explore+single_jd`（探索+单JD深度分析）：2 条
  - `single_jd_not_found`（知识库无命中）：2 条
  - `explore+verify`（探索+属性查询）：2 条
  - `multi_jd_compare`（多JD对比）：1 条

### 2.3 单条数据结构

```json
{
  "case_id": "eval_chen_02",
  "original_query": "字节跳动的AI产品经理我够格吗",
  "resume_id": "cc50dcfb-aeba-41a3-989a-9e485aa2fc3f",
  "rewritten_query": "字节跳动的AI产品经理我匹配吗",
  "search_keywords": "字节跳动 AI产品经理 匹配",
  "gold_intents": ["assess"],
  "gold_slots": {"company": "字节跳动", "position": "AI产品经理"},
  "expected_tools": ["kb_retrieve", "match_analyze"],
  "scenario": "有简历_单JD匹配_知识库命中",
  "follow_up_type": "none",
  "session_group": null,
  "golden_jd_ids": [5],
  "golden_chunk_ids": [
    "ac9973c8-01be-4cd6-9a99-097873a59cf0_chunk_0",
    "..."
  ],
  "critical_sections": ["basic_info", "responsibilities", "hard_requirements", "soft_requirements", "keywords"],
  "retrieval_type": "single_jd",
  "relevance_scores": {"5": 3},
  "annotation_notes": "明确查询字节跳动AI产品经理（JD ID=5），期望精确命中该JD全部chunks",
  "eval_context_notes": "指定JD在库中(ID=5)，期望检索后做match_analyze"
}
```

### 2.4 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `case_id` | string | 测试用例唯一标识 |
| `original_query` | string | 用户原始输入 |
| `rewritten_query` | string | **改写后的query**（指代消解+口语降噪后的结构化query） |
| `search_keywords` | string | **检索关键词**（用于向量和BM25检索的query） |
| `golden_jd_ids` | int[] | **期望命中的JD ID列表** |
| `golden_chunk_ids` | string[] | **期望命中的chunk_id列表**（按critical_sections提取） |
| `critical_sections` | string[] | **关键section类型**，用于分层统计精确率/召回率 |
| `retrieval_type` | string | 检索场景类型 |
| `relevance_scores` | dict | **JD级别相关性分数**（0-3分，用于NDCG计算）<br>3=高度相关，2=相关，1=弱相关，0=不相关 |
| `annotation_notes` | string | 标注说明 |

---

## 三、评测指标

### 3.1 检索级指标（Chunk Level）

#### 精确率 & 召回率（基于 Critical Sections）

```
Context Precision = |retrieved_chunks ∩ golden_chunk_ids| / |retrieved_chunks|
Context Recall    = |retrieved_chunks ∩ golden_chunk_ids| / |golden_chunk_ids|
```

> 由于 golden_chunk_ids 按 `critical_sections` 提取（而非全部chunks），这两个指标反映的是**关键信息是否被准确召回**。

#### F1 Score

```
F1 = 2 * Precision * Recall / (Precision + Recall)
```

### 3.2 传统检索指标（JD Level）

#### Hit Rate（命中率）

```
Hit Rate = 1  if 至少一个 golden_jd_id 出现在 retrieved JDs 中
         = 0  otherwise
```

#### MRR（Mean Reciprocal Rank）

```
MRR = (1/|cases|) * Σ(1 / rank_of_first_relevant_jd)
```

> 取检索结果中第一个命中 golden_jd_ids 的 JD 的排名倒数。

#### NDCG（Normalized Discounted Cumulative Gain）

```
DCG@k = Σ(relevance_score_i / log2(i + 1))   for i = 1 to k
NDCG@k = DCG@k / IDCG@k
```

> 使用 `relevance_scores` 作为 graded relevance，IDCG 为理想排序下的 DCG。

### 3.3 组件级指标（供 Chunk 策略 A/B 测试用）

| 指标 | 说明 | 适用场景 |
|------|------|---------|
| `section_precision` | 关键 section 的 chunk 精确率 | 对比不同 chunk 切分策略 |
| `section_recall` | 关键 section 的 chunk 召回率 | 同上 |
| `vec_recall` | 向量路独立召回的 golden chunk 占比 | 评估 Embedding 模型质量 |
| `bm25_recall` | BM25 路独立召回的 golden chunk 占比 | 评估关键词匹配质量 |
| `fusion_coverage` | 两路融合后覆盖的 golden chunk 占比 | 评估融合策略有效性 |
| `rerank_improvement` | 重排序后 MRR/NDCG 的提升幅度 | 评估 Reranker 效果 |

---

## 四、评测流程

### 4.1 准备环境

```bash
cd backend
# 确保 ChromaDB 中有数据
python -c "from app.core.vector_store import VectorStore; print(VectorStore()._collection.count())"
# 输出应为 30 条 JD 对应的 chunk 总数（约 300+ chunks）
```

### 4.2 运行 RAG 评测

```bash
cd backend && python eval/run_rag_eval.py
```

> `run_rag_eval.py` 脚本需自行实现，核心逻辑：
> 1. 读取 `rag_test_dataset.jsonl`
> 2. 对每条 case 调用 `kb_retrieve`（或直接调用 `VectorStore.query` + BM25 + 融合 + Reranker）
> 3. 收集 retrieved_chunks 和 retrieved_jd_ids
> 4. 计算所有指标
> 5. 输出 JSON 报告

### 4.3 对比实验示例

```bash
# 实验1：仅向量检索（top_k=10）
cd backend && python eval/run_rag_eval.py --strategy vector --top_k 10

# 实验2：仅 BM25（top_k=10）
cd backend && python eval/run_rag_eval.py --strategy bm25 --top_k 10

# 实验3：混合召回（默认 70%vec + 30%bm25）
cd backend && python eval/run_rag_eval.py --strategy hybrid --top_k 10

# 实验4：混合 + CrossEncoder 重排序
cd backend && python eval/run_rag_eval.py --strategy hybrid --rerank --top_k 10

# 实验5：不同 chunk 切分策略对比
# 需先重新构建知识库（semantic vs fixed vs recursive vs section）
cd backend && python scripts/rebuild_kb.py --strategy semantic
```

---

## 五、指标解读与问题定位

参考视频中提到的 RAG 评测三层拆解思想：

| 指标组合 | 可能的问题定位 |
|---------|--------------|
| Context Precision 低，Context Recall 高 | 召回了很多无关 chunks，可能 BM25 权重过高或 chunk 切分过细 |
| Context Precision 高，Context Recall 低 | 召回的 chunks 都对但不够全，可能 top_k 太小或向量检索漏召 |
| Precision & Recall 都低 | 检索方向完全错误，检查 Embedding 模型、Query Rewrite、Chunk 切分 |
| Hit Rate 低 | 连正确的 JD 都没找到，检查 Embedding 质量或 BM25 索引 |
| MRR 低但 Hit Rate 高 | 找到了正确的 JD 但排名靠后，检查融合权重或 Reranker |
| NDCG 低 | 整体排序质量差，检查 Reranker 模型或相关性分数校准 |

---

## 六、与端到端评测的关系

```
┌─────────────────────────────────────────────┐
│  L1 组件级评测（本文档）                      │
│  kb_retrieve 独立评测                         │
│  指标：Precision / Recall / Hit Rate / MRR / NDCG │
├─────────────────────────────────────────────┤
│  L2 端到端链路评测（EVAL_SYSTEM.md）          │
│  完整 ReAct Agent 链路                         │
│  指标：task_success_rate / intent_f1 / tool_f1 │
├─────────────────────────────────────────────┤
│  L3 生成质量评测（未来）                       │
│  Faithfulness / Answer Relevancy              │
│  基于 LLM-as-judge 或人工评估                  │
└─────────────────────────────────────────────┘
```

---

## 七、已知问题与注意事项

1. **Golden Set 局限性**：
   - 当前 43 条 case 全部来自现有端到端测试集，覆盖场景有限
   - 未来应补充更多边界 case（如长query、多语言、错别字等）

2. **Relevance Score 主观性**：
   - `relevance_scores` 为人工标注，可能存在偏差
   - 建议定期抽检并由多人标注取平均

3. **Chunk 标注粒度**：
   - 当前按 `critical_sections` 提取 chunk_ids，同一 section 内的多条要求无法区分优先级
   - 未来可细化到单条 requirement 级别

4. **多轮场景**：
   - `verify_expand` 类型的 case（如 eval_chen_06）依赖历史槽位继承
   - 单独测试 RAG 时需手动注入 history_slots 或 evidence_cache

---

*文档版本：v1.0*  
*生成日期：2026-05-28*  
*黄金数据集：backend/eval/rag_test_dataset.jsonl（43条）*
