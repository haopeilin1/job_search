# 求职雷达 Agent — 系统架构文档

> 本文档描述当前后端系统的完整架构，基于代码实际实现，非设计愿景。

---

## 一、总体架构概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FastAPI 入口层                                   │
│  POST /api/v1/chat  →  chat_endpoint()  →  _handle_llm_route_v2()            │
│  POST /api/v1/chat/stream  →  chat_stream_endpoint()                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 0: QueryRewriter（问题改写）                                           │
│  输入: 原始用户消息 + SessionMemory（三层记忆）                               │
│  输出: QueryRewriteResult(rewritten_query, follow_up_type, resolved_refs)   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 1: LLMIntentRouter（三级意图识别流水线）                                │
│  ┌─────────────┐    ┌──────────────────┐    ┌──────────────────────────┐   │
│  │ ① 规则层    │ →  │ ② 小模型校准器   │ →  │ ③ 大模型仲裁/兜底        │   │
│  │ RuleRegistry│    │ SmallModel       │    │ LLMFallbackClassifier    │   │
│  │ (关键词匹配)│    │ Calibrator       │    │ (冲突消解+澄清生成)      │   │
│  └─────────────┘    │ (Ollama qwen2.5) │    └──────────────────────────┘   │
│                     └──────────────────┘                                    │
│  输出: MultiIntentResult(candidates[], primary_intent, topology, clarify?)  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 2: TaskGraphPlanner（动态任务规划）                                     │
│  输入: MultiIntentResult + SessionMemory + resume_text                        │
│  输出: TaskGraph（任务节点 + 依赖关系 + 并行组 + Fallback策略）               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 3: ReActExecutor（任务执行引擎）                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ while 还有未完成任务:                                               │   │
│  │   1. 获取所有依赖已就绪的任务（ready tasks）                        │   │
│  │   2. 并行执行本批任务（asyncio.gather）                             │   │
│  │   3. 每任务执行后 → Replanner.should_replan() 判断                  │   │
│  │   4. 若触发 Replan(T1~T5) → 动态扩展任务图                         │   │
│  │   5. 检查全局状态 / 死锁 / 全部完成                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  输出: 执行后的 TaskGraph（各任务 success/failed/skipped 状态）              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 4: LLM 聚合回复（最终生成）                                             │
│  输入: 所有成功任务的输出结果                                                 │
│  模型: chat → core → planner → memory 四级降级                               │
│  输出: 结构化回复文本                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 5: 记忆管理（MemoryManager）                                            │
│  工作记忆轮转(WorkingMemory → CompressedMemory) + 长期记忆更新                │
│  保存 DialogueTurn + PendingClarification 到 SessionMemory                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、分层详细架构

### 2.1 入口层 (`app/routers/chat.py`)

**职责**：统一对话入口，路由分发，Session 管理，响应组装。

**关键函数**：

| 函数 | 职责 |
|------|------|
| `chat_endpoint()` | 主入口。读取 `AGENT_MODE` 配置，决定走规则路线还是 LLM 路线 |
| `_handle_llm_route_v2()` | LLM 路线 v2 完整链路（Query改写 → 意图 → Plan → 执行 → 聚合） |
| `_handle_rule_route()` | 规则路线（旧版，保留兼容） |
| `_get_or_create_session()` | Session 获取/创建，支持从 SQLite 恢复 |

**Session 存储**：
- 内存缓存：`_session_store: dict[str, SessionMemory]`（进程内）
- 持久化：`memory.db` SQLite（长期记忆、会话元数据）
- 恢复策略：重启后按 session_id 从数据库恢复元数据，长期记忆始终从 DB 加载

**路由注册**（`app/main.py`）：
```python
app.include_router(chat_router)          # 旧版 /api/v1/chat/* 兼容
app.include_router(chat_api_router, prefix="/api/v1")  # 新版 /api/v1/chat
```

---

### 2.2 QueryRewrite 层 (`app/core/query_rewrite.py`)

**职责**：问题改写、指代消解、口语降噪、追问检测。

**输出数据结构**：
```python
@dataclass
class QueryRewriteResult:
    rewritten_query: str       # 改写后的结构化 query
    search_keywords: str       # 检索关键词
    resolved_references: Dict  # 指代映射 {company: "字节跳动", position: "AI产品经理"}
    is_follow_up: bool         # 是否追问
    follow_up_type: str        # none / expand / switch / clarify
```

**追问检测规则**（基于关键词，非 LLM）：

| 类型 | 检测条件 | 示例 |
|------|---------|------|
| `switch` | 含切换词（"换", "不说", "另外", "对比"） | "换一家看看" |
| `clarify` | 含澄清词（"不对", "我说的是", "纠正"） | "不对，我说的是阿里巴巴" |
| `expand` | 极短 + 历史槽位 / 含指代词（"那个", "这个", "刚才"） | "那个岗薪资多少" |
| `expand` | 无实体 + 历史槽位 | "要求是什么"（上轮提过公司/岗位） |
| `none` | 以上均不满足 | 首轮对话 / 完整输入 |

**历史槽位获取**（`_get_history_slots`）：
- 优先从 `session.global_slots` 读取
- 次选从 `session.working_memory.turns[-1].global_slots` 读取
- 若都为空，返回 `{}`

**⚠️ 当前限制**：
- 追问检测完全基于规则关键词，**不使用 LLM**
- 指代消解能力较弱（"上面那个岗" 无法正确解析）
- `follow_up_type` 的准确性依赖 `global_slots` 是否正确写入

---

### 2.3 意图识别层 (`app/core/llm_intent.py`)

**核心类**：`LLMIntentRouter`

**三级流水线**：

#### ① 规则层（RuleRegistry）

**注册表结构**：按优先级 `priority=10~34` 注册，执行时按 `strength(STRONG>WEAK)` 和 `priority` 排序。

| 优先级 | 规则名 | 触发条件 | 输出 strength |
|--------|--------|---------|--------------|
| 10 | L1-附件+单实体评估 | 有附件 + ASSESS 关键词 | STRONG |
| 11 | L1-附件+核实词 | 有附件 + VERIFY 关键词 | STRONG |
| 12 | L1-附件+准备词 | 有附件 + PREPARE 关键词 | STRONG |
| 13 | L1-附件+管理词 | 有附件 + MANAGE 关键词 | STRONG |
| 14 | L1-JD长文本 | >200字 + JD markers | STRONG |
| 20 | L2-全局探索 | RANGE 关键词 | STRONG |
| 21 | L2-问候 | 纯问候语 | STRONG |
| 22 | L2-面试准备 | PREPARE 关键词 | STRONG |
| 23 | L2-属性核实 | ATTR 关键词 | STRONG/WEAK |
| 23 | L2-岗位介绍 | "分析/介绍" + 实体 | WEAK |
| 24 | L2-管理操作 | MANAGE 关键词 | STRONG |
| 25 | L2-通用咨询 | 职业规划等词 | WEAK |
| 30 | L3-引用式探索 | 公司名 + EXPLORE 词 | WEAK |
| 31 | L3-引用式评估 | 公司名/岗位名 + 匹配度词 | WEAK |
| 32 | L3-引用式核实 | 问题词 + 属性词 + 实体 | WEAK |
| 33 | L3-引用式准备 | 实体 + 准备词 | WEAK |
| 34 | L3-存在性核实 | 公司 + "有"/"有没有" + 疑问词 | WEAK |

**兜底规则**：若所有规则均未命中，返回一个 `intent=None, strength=MISS` 的兜底结果。

**意图类型枚举**：
```python
class LLMIntentType(Enum):
    ASSESS = "assess"      # 匹配度评估（需简历）
    VERIFY = "verify"      # 属性核实（薪资/要求/学历等）
    EXPLORE = "explore"    # 全局探索/推荐/排序
    PREPARE = "prepare"    # 面试准备
    MANAGE = "manage"      # 简历/JD 管理
    CHAT = "chat"          # 通用对话（兜底）
```

#### ② 小模型校准器（SmallModelCalibrator）

**模型**：`qwen2.5:14b` @ Ollama（本地，keep_alive=1h）

**触发条件**：
- `STRONG` 规则命中 → **跳过校准**（fast path，节省 25-40s）
- `WEAK` 规则命中 → 触发校准
- 规则全 MISS → 若 clarify 场景，创建虚拟 CHAT 候选后触发校准

**校准流程**：
1. 构建三层记忆上下文（工作记忆 + 压缩记忆 + 长期记忆）
2. 拼接 `INTENT_CALIBRATION_SYSTEM` + `INTENT_CALIBRATION_EXAMPLES` prompt
3. 调用 Ollama 生成 JSON：意图判断 + 槽位抽取 + 置信度
4. 解析结果生成 `IntentCandidate`

**⚠️ 当前问题**：Ollama 本地模型在冷启动/大 prompt 时容易 ReadTimeout（默认 timeout=300s，但实际受模型加载影响）

#### ③ 大模型仲裁/兜底（LLMFallbackClassifier）

**模型**：与 chat 层相同（默认 `qwen3.5-plus` @ DashScope）

**触发条件**：
- 校准器返回空列表（规则全 MISS 且校准失败）
- 多意图冲突需消解
- 置信度 < 0.5 的候选需过滤

**仲裁流程**（`arbitrate`）：
1. 过滤低置信度候选（< 0.5）
2. 若过滤后为空 → 调用 `_generate_clarification_question()` 生成友好澄清问题
3. 冲突消解（硬规则）：CHAT+其他→CHAT 降级；MANAGE 优先；ASSESS+PREPARE→添加依赖
4. 若候选 > 1 且存在复杂依赖 → 调用大模型 `_llm_arbitrate()` 最终仲裁
5. 构建执行拓扑（`execution_topology`）
6. 全局槽位合并（`_merge_global_slots`）
7. 澄清判断（`_check_clarification_need`）

**澄清判断规则**：
- ASSESS：必须有 `resume_available=True`，否则触发澄清
- VERIFY：有 company 或 position 之一即可通过
- 其他意图：按具体规则判断

#### Clarify 场景特殊处理

**Clarify 状态机**：

```
Round 1: 用户输入意图模糊
    → 规则全 MISS / 校准器返回 CHAT / 仲裁判断 needs_clarification=True
    → 保存 PendingClarification(session.pending_clarification)
    → 返回澄清问题给用户

Round 2: 用户补充实体（短输入 < 20 字）
    → QueryRewrite.follow_up_type 可能为 "none"（若 global_slots 未同步）
    → 但检测到 pc 存在且未过期 + 输入短 → is_clarify_follow_up=True
    → 进入 Clarify 恢复分支：
        a. 合并 pc.resolved_slots + rewrite_result.resolved_references
        b. 从用户原始输入中按长度降序提取 company/position
        c. 若 pending_intent=chat（上轮意图模糊）：
           - 重新调用 route_multi()，让校准器利用工作记忆推断
           - 若校准器仍返回 CHAT，根据上轮用户输入关键词硬编码修正：
             * "分析/介绍/了解一下" → VERIFY
             * "匹配/适合/差距" → ASSESS
        d. 构建 restored_candidate，检查仍有缺失槽位则再次澄清
    → 若恢复成功，进入正常 Plan→Execute 流程
```

**PendingClarification 数据结构**：
```python
@dataclass
class PendingClarification:
    pending_intent: str           # 上轮推断的意图（如 "chat"）
    missing_slots: List[str]      # 缺失的槽位（如 ["company", "position"]）
    resolved_slots: Dict          # 已解析的槽位
    clarification_question: str   # 系统提出的澄清问题
    expected_slot_types: List[str]
    created_turn_id: int
    timestamp: float

    def is_expired(self, current_turn_id, max_gap=2) -> bool:
        return current_turn_id - self.created_turn_id > max_gap
```

---

### 2.4 规划层 (`app/core/llm_planner.py`)

**核心类**：`TaskGraphPlanner`

**职责**：将多意图拆解为原子任务图，定义依赖、失败处理、执行策略。

**输入**：
- `MultiIntentResult`（意图候选 + 全局槽位 + 执行拓扑）
- `SessionMemory`（历史缓存、证据池）
- `resume_text`（简历文本）
- `rewrite_result`（改写结果）

**输出**：`TaskGraph`（新体系）→ 经 `new_arch_adapter.convert_task_graph()` 转换为旧体系 `TaskGraph`

**Planner System Prompt 核心内容**：
1. **任务拆分器**：一个意图拆多个原子任务，多意图合并公共前置
2. **任务依赖设定器**：数据依赖 / 控制依赖 / 共享依赖
3. **状态跟踪器**：考虑全局槽位中哪些数据已就绪
4. **失败处理器**：retry / skip / ask_user / abort 四级策略

**Planner KB Schema**：明确告知 Planner 知识库覆盖范围（13 公司/24 岗位）和检索决策原则（先 kb_retrieve，不足再 external_search）

**TaskGraph 数据结构**：
```python
@dataclass
class TaskNode:
    task_id: str                # "T0", "T1"...
    task_type: str              # "tool_call" / "llm_reasoning" / "aggregate"
    tool_name: Optional[str]    # "kb_retrieve", "match_analyze", "qa_synthesize"...
    description: str
    parameters: Dict            # 原始参数（可能含 {{占位符}}）
    resolved_params: Dict       # 解析后的实际参数
    dependencies: List[str]     # 依赖的 task_id 列表
    status: str                 # pending / running / success / failed / skipped
    result: Any                 # 执行结果
    observation: str            # 观察/日志
    is_critical: bool           # 是否关键任务

@dataclass
class TaskGraph:
    tasks: Dict[str, TaskNode]
    execution_strategy: ExecutionStrategy  # 并行组划分
    global_status: str          # running / success / failed / needs_clarification
```

**常用任务模式**：
| 意图 | 典型任务图 |
|------|-----------|
| VERIFY | T0:kb_retrieve → T1:qa_synthesize → T2:aggregate |
| ASSESS | T0:kb_retrieve → T1:match_analyze → T2:aggregate |
| EXPLORE | T0:kb_retrieve → T1:global_rank → T2:aggregate |
| PREPARE | T0:kb_retrieve → T1:match_analyze → T2:interview_gen → T3:aggregate |
| 多意图 | 合并公共 kb_retrieve，下游按依赖分叉 |

---

### 2.5 执行层 (`app/core/react_executor.py`)

**核心类**：`ReActExecutor` + `Replanner`

**执行循环**（`execute` 方法）：

```python
iteration = 0
while iteration < MAX_STEPS(10):
    iteration += 1

    # 1. 获取可执行任务（所有依赖已到达终态）
    ready = graph.get_ready_tasks()
    if not ready: break  # 死锁或完成

    # 2. 并行执行本批任务
    results = await asyncio.gather(*[
        self._execute_single(t, graph, session) for t in ready
    ])

    # 3. 处理结果 + Replan 判断
    for task, result in zip(ready, results):
        replan_trigger = Replanner.should_replan(task, graph)
        if replan_trigger:
            graph = await Replanner.replan(graph, replan_trigger, task)

    # 4. 检查全局状态
    if graph.global_status in ("needs_clarification", "failed"): break

    # 5. 若 Replan 插入了新任务，继续循环
    if needs_replan: continue

    # 6. 检查是否全部完成
    if all(t.status in ("success", "failed", "skipped", "aborted") for t in graph.tasks.values()):
        break
```

**单任务执行**（`_execute_single`）：
1. 动态槽位填充：解析 `{{T0.result.data.chunks}}` 等跨任务引用
2. match_analyze 特殊处理：若 jd_text 与 company/position 不匹配，自动从 T0 筛选匹配的 chunk
3. 按 task_type 分发：
   - `tool_call` → `ToolRegistry.execute()`
   - `llm_reasoning` → 直接调用 LLM
   - `aggregate` → 合并上游结果

**Replan 触发条件**（`Replanner.should_replan`）：

| 触发码 | 条件 | 处理动作 |
|--------|------|---------|
| T1_tool_failed | 工具执行异常/失败 | 替代方案（如禁用 BM25）或 retry |
| T2_insufficient_results | kb_retrieve chunks < 2 | 插入 `external_search` 任务 |
| T2_low_relevance | kb_retrieve 最高分 < 0.3 | 同上 |
| T4_low_match_score | match_analyze score < 50 | 插入建议生成任务 |
| T5 | 预留扩展点 | — |

**⚠️ 当前 Replan 限制**：
- T3（检索结果不相关）未实现（需额外 LLM 调用，成本较高）
- L3 全局重构（重新生成整个 TaskGraph）未实现

---

### 2.6 工具层 (`app/core/tools.py` + `app/core/tool_registry.py`)

**工具注册表**（`ToolRegistry`）：
- 延迟初始化（`create_tool_registry()` 工厂函数）
- 每个工具继承 `BaseTool`，实现 `execute(params)` 方法

**当前可用工具**：

| 工具名 | 职责 | 依赖 | 成本 |
|--------|------|------|------|
| `kb_retrieve` | 混合检索（70%向量+30%BM25）→ top-3 chunks | 无 | medium |
| `match_analyze` | 简历与JD匹配分析 → score/gaps/suggestions | kb_retrieve.chunks | high |
| `qa_synthesize` | 基于证据回答用户问题 → answer/citations | kb_retrieve.chunks | medium |
| `interview_gen` | 生成面试题 → questions/rationale | match_analyze 结果 | medium |
| `global_rank` | 多JD排序 → ranked_list/explanation | kb_retrieve.chunks | high |
| `external_search` | 外部搜索补充（Serper/Google） | 无 | medium |
| `file_ops` | 简历上传/解析/管理 | 无 | low |
| `general_chat` | 通用对话兜底 | 无 | low |
| `evidence_relevance_check` | 证据相关性检查 | 无 | low |

**kb_retrieve 召回策略**：
1. 向量检索（ChromaDB + `text-embedding-v4` @ DashScope）
2. BM25 检索（jieba 分词构建倒排索引）
3. 混合合并：70% 向量 + 30% BM25，pool=top-15
4. 重排序：`BAAI/bge-reranker-base`（本地 CUDA）→ top-3
5. 结果返回：chunks + metadata(company, position, section, jd_id)

**⚠️ 当前问题**：rerank_score 始终为 0.0000，原因待排查。

---

### 2.7 记忆层 (`app/core/memory.py`)

**三层记忆架构**：

```
┌─────────────────────────────────────────────────────────────┐
│                    SessionMemory（会话级）                     │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Working Mem  │  │ Compressed   │  │ Long-Term Mem    │  │
│  │ (最近3轮)     │  │ Mem (4-10轮) │  │ (实体/偏好/简历)  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**WorkingMemory**：
- 容量：`max_turns=3`
- 内容：`List[DialogueTurn]`（完整对话记录，含 intent/tool_calls/tool_results/retrieved_chunks）
- 用途：直接参与推理（意图识别校准器的 prompt 中注入）

**CompressedMemory**：
- 容量：`max_blocks=5`
- 内容：由 `MemoryManager.rotate_memory()` 将溢出的 WorkingMemory turns 压缩为摘要块
- 压缩方式：调用 memory 层 LLM 生成摘要
- 用途：校准器 prompt 中作为 "compressed_history" 注入

**LongTermMemory**：
- 持久化到 SQLite（`memory.db`）
- 内容：
  - `profile`：用户画像（经验、技能、求职意向）
  - `preferences`：用户偏好（公司规模、薪资范围、工作城市）
  - `resume_summary`：简历摘要
  - `interaction_patterns`：交互模式（常用查询类型）
- 加载时机：`_get_or_create_session()` 时从 DB 恢复

**记忆轮转**（`MemoryManager.rotate_memory`）：
- WorkingMemory 超出容量 → 最老的 turn 被压缩 → 加入 CompressedMemory
- CompressedMemory 超出容量 → 最老的 block 被丢弃或合并

**全局槽位**（`global_slots`）：
- 类型：`Dict[str, Any]`
- 内容：跨轮次共享的槽位（company, position, resume_available, search_keywords, attributes...）
- 写入时机：每轮执行完成后，`_merge_global_slots()` 合并所有候选意图的 slots
- 读取时机：QueryRewrite `_get_history_slots()`、Planner 状态跟踪

---

### 2.8 模型层 (`app/core/llm_client.py`)

**LLMClient**：轻量 OpenAI 兼容客户端（httpx + 连接池复用）

**分层超时策略**：
```python
TIMEOUT_LIGHT    = 60.0   # 轻量调用：token少、快速分类
TIMEOUT_STANDARD = 180.0  # 标准调用：常规生成
TIMEOUT_HEAVY    = 300.0  # 重型调用：最终聚合回复
```

**模型分层配置**（`llm_config_store`）：

| 层级 | 用途 | 默认配置 | Fallback |
|------|------|---------|----------|
| `chat` | 最终聚合回复、通用对话 | `qwen3.5-plus` @ DashScope | 无 |
| `core` | match_analyze / interview_gen / qa_synthesize | 复用 chat | 复用 chat |
| `planner` | 动态任务规划（TaskPlanner） | `qwen2.5:14b` @ Ollama | 复用 chat |
| `memory` | 记忆轮转、摘要生成 | `qwen2.5:7b` @ Ollama | 复用 chat |
| `vision` | 多模态理解（简历图片 OCR） | 独立配置 | 复用 chat |
| `judge` | 意图仲裁/澄清生成 | 独立配置 | 复用 chat |

**Ollama 特殊处理**：
- `payload["keep_alive"] = "1h"`：避免 5 分钟卸载
- 连接池按 `{base_url}:{timeout}` key 复用

**最终聚合回复的模型降级链**（`_handle_llm_route_v2`）：
```python
fallback_configs = [
    ("chat", TIMEOUT_HEAVY),      # 30s
    ("core", TIMEOUT_STANDARD),   # 20s
    ("planner", TIMEOUT_LIGHT),   # 10s
    ("memory", TIMEOUT_LIGHT),    # 10s
]
```
若 chat 模型失败，依次降级到 core/planner/memory，避免单点故障。

---

## 三、完整数据流示例

以 **"阿里巴巴后端开发我匹配吗"** 为例：

```
1. 用户请求 → POST /api/v1/chat
   session_id="test_01", message="阿里巴巴后端开发我匹配吗"

2. Session 管理
   → _get_or_create_session("test_01") → 新建 SessionMemory

3. OCR 处理（无附件，跳过）

4. 路线选择
   → AGENT_MODE="llm" → use_llm_agent=True

5. QueryRewrite
   → rewriter.rewrite("阿里巴巴后端开发我匹配吗", session)
   → 历史槽位为空（首轮）
   → 无切换词/澄清词/指代词
   → rewritten="阿里巴巴后端开发我匹配吗", follow_up_type="none"

6. 意图识别（LLMIntentRouter.route_multi）
   ├─ ① 规则层
   │   → "匹配" 在 MATCH_KWS 中
   │   → "阿里巴巴" 在 kb_companies 中
   │   → "后端开发" 在 kb_positions 中
   │   → rule_l3_referenced_assess 命中: ASSESS(WEAK)
   │   → slots={company:"阿里巴巴", position:"后端开发", attributes:["匹配度"]}
   │
   ├─ ② 校准器
   │   → WEAK 规则 → 触发校准
   │   → 构建 prompt（含系统指令 + 三层记忆 + 规则参考）
   │   → Ollama qwen2.5:14b 生成 JSON
   │   → 解析结果: IntentCandidate(ASSESS, confidence=0.82, slots=...)
   │
   └─ ③ 仲裁
       → calibrated=[ASSESS(0.82)]
       → 无冲突，快速通道通过
       → needs_clarification=True（resume_available=false）

7. 澄清判断
   → ASSESS 需要 resume，但 session 无简历
   → 保存 PendingClarification(pending_intent="assess", missing_slots=["resume"])
   → 返回: "分析匹配度需要简历信息，请先上传简历"

   [用户上传简历后，第二轮继续...]

8. 第二轮: 用户输入 "已上传"
   → QueryRewrite: follow_up_type="clarify"（短输入 + pending_clarification 存在）
   → Clarify 恢复: 合并 slots, pending_intent="assess"
   → route_multi: 重新识别 → ASSESS, resume_available=true
   → needs_clarification=False

9. Plan 模块
   → TaskGraphPlanner.create_graph(
        multi_result={ASSESS, company:"阿里巴巴", position:"后端开发"},
        resume_text=简历文本
     )
   → 生成 TaskGraph:
        T0: kb_retrieve(query="阿里巴巴 后端开发", top_k=3)
        T1: match_analyze(resume_text, jd_text=T0.chunks)
        T2: aggregate(合并 T0/T1 结果)
   → parallel_groups=3（串行依赖）

10. ReAct 执行
    → iteration=1: ready=[T0]
        → T0 执行: kb_retrieve → pool=12 → top=3 chunks
        → Replan: chunks>=2, max_score>=0.3 → 不触发
    → iteration=2: ready=[T1]
        → T1 执行: match_analyze → score=78, gaps=["分布式经验不足"]
        → Replan: score>=50 → 不触发
    → iteration=3: ready=[T2]
        → T2 执行: aggregate → 合并结果
    → 全部完成

11. 更新 evidence_cache
    → session.evidence_cache = T0.chunks

12. LLM 聚合回复
    → chat 模型生成: "根据分析，您与阿里巴巴后端开发岗位的匹配度为 78 分..."

13. 保存对话历史
    → DialogueTurn 加入 WorkingMemory
    → MemoryManager.rotate_memory()（若超出容量则压缩）

14. 构造返回
    → JSON: {session_id, intent:"match_assess", reply:"...", tools:[...]}
```

---

## 四、新旧架构关系

**当前状态**：新旧架构并存，通过适配层桥接。

```
新体系（活跃开发）                          旧体系（兼容保留）
┌──────────────────┐                       ┌──────────────────┐
│ LLMIntentRouter  │                       │ IntentRouter     │
│ (llm_intent.py)  │                       │ (intent.py)      │
└────────┬─────────┘                       └────────┬─────────┘
         │                                          │
         ▼                                          ▼
┌──────────────────┐                       ┌──────────────────┐
│ TaskGraphPlanner │                       │ TaskPlanner      │
│ (llm_planner.py) │                       │ (planner.py)     │
└────────┬─────────┘                       └────────┬─────────┘
         │                                          │
         ▼                                          ▼
┌──────────────────┐                       ┌──────────────────┐
│ ReActExecutor    │◄─────────────────────►│ ReActExecutor    │
│ (react_executor) │   共用执行引擎         │ (react_executor) │
└────────┬─────────┘                       └────────┬─────────┘
         │                                          │
         └──────────────────┬───────────────────────┘
                            ▼
                   ┌──────────────────┐
                   │ ToolRegistry     │
                   │ (tool_registry)  │
                   └──────────────────┘
```

**适配层**（`app/core/new_arch_adapter.py`）：
- `multi_intent_result_to_intent_result()`：新体系 MultiIntentResult → 旧体系 IntentResult
- `convert_task_graph()`：新体系 TaskGraph → 旧体系 TaskGraph（字段映射）
- `map_intent_name_to_new()`：旧意图名 → 新意图名（`match_assess`→`assess`, `attribute_verify`→`verify`）

**保留原因**：前端 API 响应格式仍使用旧体系字段名（如 `intent="match_assess"`），适配层负责格式转换。

---

## 五、配置驱动设计

**环境变量**（`.env`）：

```bash
# Agent 模式
AGENT_MODE=llm                    # llm / rule / auto

# 模型分层（LLM 路线使用）
CHAT_MODEL=qwen3.5-plus
CHAT_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
PLANNER_BASE_URL=http://localhost:11434/v1
PLANNER_MODEL=qwen2.5:14b
MEMORY_BASE_URL=http://localhost:11434/v1
MEMORY_MODEL=qwen2.5:7b

# 向量数据库
CHROMA_PERSIST_DIR=./data/chroma_db
EMBEDDING_MODEL=text-embedding-v4

# 功能开关
EVIDENCE_CACHE_ENABLED=true
EVIDENCE_CACHE_MAX_SIZE=10
```

**Settings 配置类**（`app/core/config.py`）：
- `AGENT_MODE`: 全局 Agent 模式开关
- `DEFAULT_AGENT_MODE`: auto 模式下的默认路线
- `*_BASE_URL` / `*_API_KEY` / `*_MODEL`: 各模型层配置

---

## 六、当前实现与理想架构的对齐情况

| 理想设计 | 当前实现 | 差距 |
|---------|---------|------|
| QueryRewrite 用 LLM 做语义级改写 | 纯规则关键词匹配 | ⚠️ 指代消解能力弱，"上面那个岗"无法正确解析 |
| 意图分类含"模糊"类型 | CHAT 意图作为模糊兜底，但没有显式的"模糊"类型 | ⚠️ 规则全 MISS 时返回 CHAT，但 CHAT 也包含真实闲聊 |
| 小模型校准器稳定运行 | Ollama 频繁 ReadTimeout | ❌ 需增大超时或降级为纯规则 |
| Replan T3（结果不相关） | 未实现 | ❌ 仅实现 T1/T2/T4 |
| Replan L3（全局重构） | 未实现 | ❌ 仅实现 L1/L2 |
| 证据缓存自动复用 | 实现了缓存存储，但复用判断逻辑较简单 | ⚠️ 追问场景未充分利用历史证据 |
| 记忆压缩全自动 | 实现了轮转，但压缩质量依赖模型 | ⚠️ 小模型压缩可能丢失关键信息 |
| QueryRewrite follow_up_type=clarify 准确识别 | 依赖 global_slots 同步，经常识别为 none | ❌ clarify 恢复靠备用条件（短输入+pending）触发 |
