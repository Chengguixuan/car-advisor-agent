"""共享 HTTP 客户端（连接池复用）。

提供进程级单例的同步/异步 httpx 客户端，供 llm_client 和 graph
模块共用。生产环境（FastAPI + uvicorn）应使用 AsyncClient 以
获得更好的并发性能。

用法:
    from .http_client import get_http_client, get_async_http_client

    # 同步（CLI / 脚本）
    client = OpenAI(http_client=get_http_client())

    # 异步（FastAPI）
    client = AsyncOpenAI(http_client=get_async_http_client())
"""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# 连接池配置
_MAX_KEEPALIVE = 20
_MAX_CONNECTIONS = 100
_KEEPALIVE_EXPIRY = 30.0  # 空闲连接秒数
_CONNECT_TIMEOUT = 10.0
_TOTAL_TIMEOUT = 120.0

# 单例
_sync_client: Optional[httpx.Client] = None
_async_client: Optional[httpx.AsyncClient] = None


def get_http_client() -> httpx.Client:
    """返回共享的同步 httpx 客户端。

    启用 HTTP/1.1 Keep-Alive 连接池，同一进程内所有 LLM API 调用
    共用此客户端，避免每次请求重新建立 TCP 连接。
    """
    global _sync_client
    if _sync_client is None:
        _sync_client = httpx.Client(
            limits=httpx.Limits(
                max_keepalive_connections=_MAX_KEEPALIVE,
                max_connections=_MAX_CONNECTIONS,
                keepalive_expiry=_KEEPALIVE_EXPIRY,
            ),
            timeout=httpx.Timeout(_TOTAL_TIMEOUT, connect=_CONNECT_TIMEOUT),
        )
        logger.info(
            "created shared sync httpx.Client (keepalive=%d, max_conn=%d)",
            _MAX_KEEPALIVE, _MAX_CONNECTIONS,
        )
    return _sync_client


def get_async_http_client() -> httpx.AsyncClient:
    """返回共享的异步 httpx 客户端（生产环境推荐）。

    用于 FastAPI + uvicorn 等 ASGI 场景，配合 AsyncOpenAI 使用。
    """
    global _async_client
    if _async_client is None:
        _async_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=_MAX_KEEPALIVE,
                max_connections=_MAX_CONNECTIONS,
                keepalive_expiry=_KEEPALIVE_EXPIRY,
            ),
            timeout=httpx.Timeout(_TOTAL_TIMEOUT, connect=_CONNECT_TIMEOUT),
        )
        logger.info(
            "created shared async httpx.AsyncClient (keepalive=%d, max_conn=%d)",
            _MAX_KEEPALIVE, _MAX_CONNECTIONS,
        )
    return _async_client
