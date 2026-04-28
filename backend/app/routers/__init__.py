from .chat import router as chat_router
from .chat import api_router as chat_api_router
from .knowledge_base import router as knowledge_base_router
from .resumes import router as resumes_router
from .settings import router as settings_router
from .memory_admin import router as memory_admin_router

__all__ = [
    "chat_router",
    "chat_api_router",
    "knowledge_base_router",
    "resumes_router",
    "settings_router",
    "memory_admin_router",
]
