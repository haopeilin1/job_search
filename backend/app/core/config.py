from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "求职雷达 Agent API"
    DEBUG: bool = True
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ]

    # ── Agent 路线配置 ──
    # "rule"  = 固定走规则路线（PlanGenerator + PlanExecutor）
    # "llm"   = 固定走 LLM Agent 路线（LLMPlanner + ReActExecutor）
    # "auto"  = 读取 DEFAULT_AGENT_MODE 配置直接决定，不再做启发式判断
    AGENT_MODE: str = "llm"
    # 当 AGENT_MODE="auto" 时，直接走哪条路（不再根据消息长度/关键词启发式判断）
    DEFAULT_AGENT_MODE: str = "rule"   # "rule" | "llm"

    # ═══════════════════════════════════════════════════════════
    # LLM 双池 + 层级映射配置（国内模型推荐版）
    # ═══════════════════════════════════════════════════════════
    # 配置思路：
    #   1. 先定义"大模型池"和"小模型池"（各一套 URL/KEY/MODEL）
    #   2. 再通过 *_LAYER 决定各系统层级使用哪个池
    #   3. 某层如需独立配置，设为 "custom" 并填写该层独立参数
    #
    # 国内模型推荐：
    #   大模型（强推理）：qwen-plus / deepseek-v3 / moonshot-v1-32k
    #   小模型（轻量快）：qwen-turbo / glm-4-flash / 本地 qwen2.5:7b
    #   多模态 Vision：qwen-vl-plus
    #   Embedding：text-embedding-v4（DashScope）
    #
    # 【本地模型兼容】Ollama / vLLM：
    #   BASE_URL = "http://localhost:11434/v1"
    #   API_KEY 留空
    #   MODEL   = "qwen2.5:14b" / "deepseek-r1" 等

    # ── 大模型池（强推理任务）──
    # 承担：最终对话回复、简历匹配分析、面试题生成、任务规划、意图兜底仲裁
    LARGE_BASE_URL: str = "https://api.openai.com/v1"
    LARGE_API_KEY: str = ""
    LARGE_MODEL: str = "gpt-4o"

    # ── 小模型池（轻量快速任务）──
    # 承担：query改写、意图识别L2审核、记忆压缩、评测打分
    SMALL_BASE_URL: str = "https://api.openai.com/v1"
    SMALL_API_KEY: str = ""
    SMALL_MODEL: str = "gpt-4o-mini"

    # ── 层级映射（值填 large | small | custom）──
    #   large  → 复用大模型池（LARGE_*）
    #   small  → 复用小模型池（SMALL_*）
    #   custom → 使用下方各层独立配置（*_BASE_URL / *_API_KEY / *_MODEL）
    # 默认推荐：chat=large, core=large, planner=large, rewrite=small, memory=small, judge=small
    CHAT_LAYER: str = "large"
    CORE_LAYER: str = "large"
    PLANNER_LAYER: str = "large"
    REWRITE_LAYER: str = "small"
    MEMORY_LAYER: str = "small"
    JUDGE_LAYER: str = "small"

    # ── 各层独立配置（仅当 *_LAYER="custom" 时生效）──
    # 留空则 fallback 到 chat 的旧行为，保持向后兼容
    CHAT_BASE_URL: str = ""
    CHAT_API_KEY: str = ""
    CHAT_MODEL: str = ""
    CORE_BASE_URL: str = ""
    CORE_API_KEY: str = ""
    CORE_MODEL: str = ""
    PLANNER_BASE_URL: str = ""
    PLANNER_API_KEY: str = ""
    PLANNER_MODEL: str = ""
    REWRITE_BASE_URL: str = ""
    REWRITE_API_KEY: str = ""
    REWRITE_MODEL: str = ""
    MEMORY_BASE_URL: str = ""
    MEMORY_API_KEY: str = ""
    MEMORY_MODEL: str = ""
    JUDGE_BASE_URL: str = ""
    JUDGE_API_KEY: str = ""
    JUDGE_MODEL: str = ""

    # ── 特殊独立配置（不跟随大/小模型池）──
    # Vision 多模态（图片OCR、JD截图解析、简历图片解析）
    VISION_BASE_URL: str = "https://api.openai.com/v1"
    VISION_API_KEY: str = ""
    VISION_MODEL: str = "gpt-4o"

    # Embedding 向量模型（默认复用大模型的 base_url/api_key）
    EMBEDDING_BASE_URL: str = ""
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # ── 检索/召回配置 ──
    RETRIEVAL_TOP_K: int = 20          # 混合召回最终返回数量（重排序前的候选池）
    RETRIEVAL_VEC_TOP_K: int = 20      # 向量路独立召回数量
    RETRIEVAL_BM25_TOP_K: int = 20     # BM25 路独立召回数量
    RETRIEVAL_VEC_WEIGHT: float = 0.70 # 向量分数权重
    RETRIEVAL_BM25_WEIGHT: float = 0.30 # BM25 分数权重

    # ── CrossEncoder 重排序配置 ──
    RERANKER_ENABLED: bool = True
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    RERANKER_TOP_K: int = 10           # 重排序后最终输出数量
    RERANKER_BATCH_SIZE: int = 8
    RERANKER_MAX_LENGTH: int = 512

    # ── 双层召回 / 粗筛配置 ──
    COARSE_FILTER_TOP_K: int = 15              # 粗筛层保留的JD数量
    COARSE_FILTER_MIN_SCORE: float = 0.0       # 粗筛最低得分阈值（负分直接过滤）
    COARSE_FILTER_MULTIPLIER: int = 3          # 粗筛保留倍数（相对于最终top_k）
    COARSE_FILTER_MIN_POOL: int = 8            # 粗筛最小保留池
    GLOBAL_RANK_LLM_THRESHOLD: int = 8         # 聚合后JD数≤此值时跳过LLM精排，使用模板输出

    # ── 各意图默认检索数量 ──
    EXPLORE_TOP_K: int = 15
    MATCH_TOP_K: int = 5
    VERIFY_TOP_K: int = 10
    ASSESS_TOP_K: int = 5

    # ── 简历技能提取兜底关键词（粗筛层技能匹配用） ──
    RESUME_TECH_KEYWORDS: list[str] = [
        "Python", "Java", "Go", "C++", "JavaScript", "TypeScript",
        "React", "Vue", "Node.js", "PyTorch", "TensorFlow", "Keras",
        "MySQL", "PostgreSQL", "Redis", "MongoDB", "Elasticsearch",
        "Docker", "Kubernetes", "Kafka", "RabbitMQ",
        "LLM", "大模型", "NLP", "机器学习", "深度学习", "推荐系统",
        "LangChain", "RAG", "Agent", "Spring Boot", "微服务", "电商", "ERP",
    ]

    # ── Evidence Cache 配置（多轮对话证据复用） ──
    EVIDENCE_CACHE_ENABLED: bool = True           # 是否启用 evidence_cache 复用
    EVIDENCE_CACHE_MAX_SIZE: int = 5              # 每轮保留的 chunk 数量（用于复用判断）
    EVIDENCE_CACHE_RELEVANCE_THRESHOLD: float = 0.6  # 相关性判定阈值（0-1）

    # ── MCP / 外部搜索配置 ──
    # Brave Search（已弃用，国内不可访问）
    BRAVE_SEARCH_MCP_ENABLED: bool = False
    BRAVE_SEARCH_API_KEY: str = ""
    BRAVE_SEARCH_MAX_QUERY_LENGTH: int = 100
    BRAVE_SEARCH_RESULT_COUNT: int = 5
    BRAVE_SEARCH_TIMEOUT: int = 10
    BRAVE_SEARCH_FALLBACK_THRESHOLD: float = 0.3

    # Tavily Search（推荐，国内可直接访问，为AI RAG设计）
    # 注册: https://tavily.com/  免费额度: 1000次/月
    TAVILY_SEARCH_ENABLED: bool = False
    TAVILY_API_KEY: str = ""
    TAVILY_SEARCH_DEPTH: str = "basic"      # "basic" 或 "advanced"
    TAVILY_MAX_RESULTS: int = 5
    TAVILY_TIMEOUT: int = 15
    TAVILY_INCLUDE_ANSWER: bool = False     # 不需要LLM摘要，只要原始结果
    TAVILY_INCLUDE_DOMAINS: list[str] = []  # 限定搜索域名，如 ["zhipu.ai", "juejin.cn"]
    TAVILY_EXCLUDE_DOMAINS: list[str] = []

    class Config:
        env_file = ".env"


settings = Settings()
