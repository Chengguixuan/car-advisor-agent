"""API 路由包。

导出 chat 路由模块，供 FastAPI 应用注册。
"""

from .chat import router as chat_router

__all__ = ["chat_router"]
