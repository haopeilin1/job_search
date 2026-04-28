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

    # ── LLM 分层配置（按功能难度/成本敏感度分模型） ──
    # .env 作为启动默认值，用户界面可覆盖
    # 通用参数：未单独配置时，各层默认复用 chat 的 base_url/api_key
    #
    # 【本地模型兼容】支持 Ollama / vLLM / LM Studio 等本地部署：
    #   BASE_URL = "http://localhost:11434/v1"  (Ollama)
    #   API_KEY 留空（本地模型通常不需要）
    #   MODEL    = "llama3.1" / "qwen2.5" / "deepseek-r1" 等
    #
    # 注意：本地模型能力可能不如云端大模型，建议：
    #   - chat/core 层用 7B+ 模型
    #   - planner/memory 层可用 3B-7B 小模型降低成本
    #   - vision 层本地支持有限（如 llava），效果可能不如 GPT-4V

    # Chat 模型 — 对话最终回复生成（Handler 回复、chat.py 最终回复）
    CHAT_BASE_URL: str = "https://api.openai.com/v1"
    CHAT_API_KEY: str = ""
    CHAT_MODEL: str = "gpt-4o"

    # Core 模型 — 核心业务分析（匹配分析、面试题生成、简历解析）
    # 未配置时默认复用 CHAT_*，但建议配置为比 chat 更强的模型
    CORE_BASE_URL: str = ""
    CORE_API_KEY: str = ""
    CORE_MODEL: str = ""

    # Planner 模型 — 规划与推理（意图识别、query改写、plan生成、replan、澄清、检索决策）
    # 可用中等能力模型，如 gpt-4o-mini，显著降低成本
    PLANNER_BASE_URL: str = ""
    PLANNER_API_KEY: str = ""
    PLANNER_MODEL: str = ""

    # Memory 模型 — 记忆管理（压缩、长期记忆提取、话题切换检测）
    # 可用 cheapest 模型，对质量要求最低
    MEMORY_BASE_URL: str = ""
    MEMORY_API_KEY: str = ""
    MEMORY_MODEL: str = ""

    # Judge 模型 — 评测用 LLM-as-judge（评估回复质量、任务完成度）
    # 可用 cheapest 小模型，如 qwen-turbo / gpt-4o-mini / 本地 3B-7B
    # 未配置时 fallback 到 MEMORY_MODEL → CHAT_MODEL
    JUDGE_BASE_URL: str = ""
    JUDGE_API_KEY: str = ""
    JUDGE_MODEL: str = ""

    # Vision 模型 — 多模态（图片 OCR、JD 截图解析、简历图片解析）
    VISION_BASE_URL: str = "https://api.openai.com/v1"
    VISION_API_KEY: str = ""
    VISION_MODEL: str = "gpt-4o"

    # Embedding 模型（向量库向量化，默认复用 chat 的 base_url/api_key）
    # 本地兼容：Ollama 支持 embedding，如 "nomic-embed-text" / "mxbai-embed-large"
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
    COARSE_FILTER_MIN_SCORE: float = -5.0      # 粗筛最低得分阈值
    COARSE_FILTER_MULTIPLIER: int = 3          # 粗筛保留倍数（相对于最终top_k）
    COARSE_FILTER_MIN_POOL: int = 8            # 粗筛最小保留池

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
