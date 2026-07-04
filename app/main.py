"""买车智能体 — FastAPI 服务入口。

启动后提供 REST API 和 SSE 流式接口。

启动方式:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

接口:
    POST /chat          — 同步对话
    POST /chat/stream   — SSE 流式对话
    GET  /docs          — Swagger API 文档
    GET  /health        — 健康检查
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routes import chat_router, conversation_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时预加载 Agent。"""
    logger.info("starting Car Advisor API server...")
    try:
        from app.services.agent_service import AgentService
        svc = AgentService()
        svc.get_agent()  # 预热 Agent
        logger.info("agent preloaded")
    except Exception as e:
        logger.warning("preload failed (will lazy-load on first request): %s", e)
    yield
    logger.info("shutting down Car Advisor API server")


app = FastAPI(
    title="买车智能体 API",
    description="基于 LangGraph + RAG 的智能购车顾问服务",
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=JSONResponse,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 路由注册
app.include_router(chat_router)
app.include_router(conversation_router)


# ---------------------------------------------------------------------------
# 根路由 + 健康检查
# ---------------------------------------------------------------------------


@app.get("/")
async def root():
    """API 说明。"""
    return {
        "service": "买车智能体 API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "POST /chat": "同步对话",
            "POST /chat/stream": "SSE 流式对话",
            "GET /health": "健康检查",
        },
    }


@app.get("/health")
async def health():
    """健康检查。"""
    return {"status": "ok", "service": "car-advisor"}


# ---------------------------------------------------------------------------
# 全局异常处理
# ---------------------------------------------------------------------------


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录请求耗时 + 确保 UTF-8 编码。"""
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info("%s %s → %d (%.0fms)", request.method, request.url.path, response.status_code, elapsed_ms)
    ct = response.headers.get("content-type", "")
    if ("text/" in ct or "application/json" in ct) and "charset" not in ct:
        response.headers["content-type"] = ct + "; charset=utf-8"
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理 — 返回统一 JSON 格式。"""
    logger.exception("unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": str(exc),
            "path": request.url.path,
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
