import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings

# 配置全局日志级别为 INFO，方便调试时观察 LLM 解析结果
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ── Telemetry 专用日志：输出结构化 JSON 到 events.jsonl ──
import os
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
telemetry_handler = logging.FileHandler(
    os.path.join(LOG_DIR, "events.jsonl"),
    encoding="utf-8",
)
telemetry_handler.setFormatter(logging.Formatter("%(message)s"))  # 纯 JSON，无前缀
telemetry_logger = logging.getLogger("telemetry")
telemetry_logger.setLevel(logging.INFO)
telemetry_logger.addHandler(telemetry_handler)
telemetry_logger.propagate = False  # 不向上传播，避免重复输出到控制台

from app.routers import chat_router, chat_api_router, knowledge_base_router, resumes_router, settings_router, memory_admin_router

app = FastAPI(
    title=settings.APP_NAME,
    description="求职雷达 Agent 后端 API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat_router)          # 旧版 /api/v1/chat/* 接口（保留兼容）
app.include_router(chat_api_router, prefix="/api/v1")  # 新版统一入口 /api/v1/chat
app.include_router(knowledge_base_router)
app.include_router(resumes_router)
app.include_router(settings_router)
app.include_router(memory_admin_router)   # 长期记忆管理接口


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": settings.APP_NAME}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
