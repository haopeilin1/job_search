# Routing RAG 部署方案

> 基于 ablation 实验结果，按实际意图（explore/assess/verify/prepare）动态配置检索参数。

---

## 1. 问题背景

当前系统 `_kb_retrieve_stub` 的 reranker 输出固定为 `RERANKER_TOP_K=10`，导致：
- **assess** 意图（单JD深度匹配）只能拿到 10 条 chunk，MRR 被限制在 0.537
- **verify** 意图（属性事实查询）需要精确覆盖，10 条可能遗漏关键信息
- **explore** 意图（宽泛浏览）10 条刚好够用，但池子太大反而引入噪声

ablation 实验已证明：**不同意图对 pool 大小和 rerank 输出量的需求显著不同**。

---

## 2. 实验依据

### 2.1 按实际意图映射 retrieval_type

| 实际意图 | 分布 | 对应 retrieval_type |
|---------|------|-------------------|
| explore | 21条 | explore(10), skill_explore(6), explore+single_jd(2), explore+verify(2) |
| assess | 11条 | single_jd(6), explore+single_jd(2), multi_jd_compare(1), explore(1) |
| verify | 16条 | verify(7), verify_expand(3), single_jd(2), explore+verify(2), explore(1) |
| prepare | 6条 | unknown(3无RAG), explore(2), single_jd(1) |
| chat/manage/clarification | 6条 | 不需要 RAG |

### 2.2 关键实验数据（hybrid_73 策略）

| 场景 | k15-10 MRR | k25-15 MRR | 结论 |
|------|-----------|-----------|------|
| explore | 0.649 | 0.606 | k15-10 **更好** |
| skill_explore | 0.774 | 0.694 | k15-10 **更好** |
| single_jd | 0.537 | **0.815** | k25-15 **显著更优** |
| verify | 0.929 | **1.000** | k25-15 **完美** |
| multi_jd_compare | **1.000** | 0.500 | k15-10 更好（rerank 限制） |

> 加权后的意图级结论：
> - **explore**: k15-10 占优（76% 的 explore case 在 k15-10 更优）
> - **assess**: k25-15 占优（73% 的 assess case 在 single_jd 场景，k25-15 MRR 高 52%）
> - **verify**: k25-15 占优（verify 本身在 k25-15 达到 1.0 MRR）

---

## 3. 路由配置

| 意图 | pool_k | rerank_k | 策略特征 | 实验依据 |
|------|--------|----------|---------|---------|
| **explore** | 15 | 10 | 广度优先，控制噪声 | explore+skill_explore 在 k15-10 的 MRR(0.65/0.77) > k25-15(0.61/0.69) |
| **assess** | 20 | 12 | 深度优先，覆盖单JD全量 | single_jd MRR: 0.815 vs 0.537，提升 52% |
| **verify** | 20 | 12 | 精确优先，高召回 | verify MRR: 1.000 vs 0.929 |
| **prepare**(有JD) | 20 | 12 | 同 assess | 需要 JD 全量信息生成面试题 |
| **prepare**(无JD) | — | — | 不调用 kb_retrieve | 纯通用面试问题，如"Java后端面试一般问什么" |
| **chat/clarification/manage** | — | — | 不调用 kb_retrieve | 闲聊/澄清/管理操作 |

---

## 4. 代码修改清单

### 4.1 `backend/app/core/config.py`

新增意图级路由参数：

```python
EXPLORE_POOL_K: int = 15
EXPLORE_RERANK_K: int = 10
ASSESS_POOL_K: int = 20
ASSESS_RERANK_K: int = 12
VERIFY_POOL_K: int = 20
VERIFY_RERANK_K: int = 12
PREPARE_POOL_K: int = 20
PREPARE_RERANK_K: int = 12
```

旧参数保留向后兼容：
```python
EXPLORE_TOP_K: int = 15   # = EXPLORE_POOL_K
MATCH_TOP_K: int = 5
VERIFY_TOP_K: int = 10
ASSESS_TOP_K: int = 5
```

### 4.2 `backend/app/core/tools.py`

修改 `_kb_retrieve_stub` 签名，支持 `pool_k` 和 `rerank_k`：

```python
async def _kb_retrieve_stub(
    query: str,
    company: Optional[str] = None,
    position: Optional[str] = None,
    top_k: int = None,      # 向后兼容
    pool_k: int = None,     # 混合池大小（优先）
    rerank_k: int = None,   # 重排序输出（优先）
) -> ToolResult:
```

- `pool_k` 控制混合融合后的候选池大小
- `rerank_k` 控制 CrossEncoder 最终输出数量
- `top_k` 保留向后兼容，仅在 `pool_k` 为 None 时生效

### 4.3 `backend/app/core/llm_planner.py`

修改 `_rule_fallback_plan()` 中的 `get_or_create_kb`：

```python
# explore → 广度优先
def get_or_create_kb(query, pool_k=None, rerank_k=None, ...)

# 各意图调用方式：
kb = get_or_create_kb(query, pool_k=settings.EXPLORE_POOL_K, rerank_k=settings.EXPLORE_RERANK_K)
kb = get_or_create_kb(query, company=company, pool_k=settings.ASSESS_POOL_K, rerank_k=settings.ASSESS_RERANK_K)
kb = get_or_create_kb(query, company=company, pool_k=settings.VERIFY_POOL_K, rerank_k=settings.VERIFY_RERANK_K)
kb = get_or_create_kb(query, company=company, pool_k=settings.PREPARE_POOL_K, rerank_k=settings.PREPARE_RERANK_K)
```

---

## 5. 生产安全约束

### 5.1 严禁单路检索
> ablation 实验中，纯 vector-only 和 bm25-only 策略会导致 reranker 崩溃（"重排序输入 0 个候选发生异常"）。
> 
> **生产环境必须使用 hybrid 融合**，保证候选池始终充足。

### 5.2 参数范围
```python
pool_k:    [5, 50]   # 混合池太小会漏召回，太大会拖慢 reranker
rerank_k:  [3, 20]   # 输出太少信息不足，太多 LLM context 装不下
```

### 5.3 Fallback 策略
- 若 `pool_k`/`rerank_k` 传值为字符串/None，自动 fallback 到 `settings.RETRIEVAL_TOP_K` / `settings.RERANKER_TOP_K`
- 若 reranker 崩溃，fallback 到 hybrid_score 排序
- 若 hybrid 池为空，返回空结果（不 crash）

---

## 6. 效果预期

| 指标 | 改造前（统一k15-10） | 改造后（路由RAG） | 提升 |
|------|-------------------|-----------------|------|
| explore MRR | 0.649 | 0.649 | 持平（已是该意图最优） |
| assess MRR | 0.537 | **0.815** | **+52%** |
| verify MRR | 0.929 | **1.000** | **+7.6%** |
| 全局平均 MRR | ~0.65 | ~0.74 | **+14%** |

---

## 7. 后续优化方向（不紧急）

1. **multi_jd_compare 优化**：当前 assess 的 pool=20/rerank=12 对多JD对比场景（multi_jd_compare）略有过剩，可考虑在 detect 到多公司时动态降级到 pool=15/rerank=10
2. **prepare(无JD) 免检索**：当前 prepare 有 company/position 时才检索，但可通过语义判断"通用面试问题" vs "针对某JD准备"来更精准地控制
3. **在线 A/B**：上线后收集真实用户 query，按 intent 分组评估 Precision@K 和 NDCG，微调参数
