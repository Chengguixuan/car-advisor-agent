"""对话 API 路由。

提供 /chat（同步）和 /chat/stream（SSE 流式）两个接口。
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.services.agent_service import AgentService
from car_advisor.src.session_store import get_messages, save_session, save_message
from car_advisor.src.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

# 全局 AgentService 单例
agent_service = AgentService()


# ==========================================================================
# 请求 / 响应模型
# ==========================================================================


class ChatRequest(BaseModel):
    """对话请求。"""

    user_input: str = Field(
        ...,
        description="用户消息文本",
        min_length=1,
        examples=["推荐一款15万的SUV"],
    )
    thread_id: Optional[str] = Field(
        default=None,
        description="会话 ID（不传则自动生成）。同 ID 共享对话历史。",
        examples=["user_001"],
    )


class ChatResponse(BaseModel):
    """对话响应（同步）。"""

    response: str = Field(..., description="助手回复文本")
    final_recommendation: Optional[dict] = Field(
        default=None,
        description="结构化推荐结果（含 understanding / recommended_models / follow_up_question）",
    )
    thread_id: str = Field(..., description="本次对话的会话 ID")
    elapsed_ms: float = Field(..., description="处理耗时（毫秒）")


class StreamEvent(BaseModel):
    """SSE 流事件。"""

    event: str = Field(..., description="事件类型：start / tool_call / tool_result / message / final_recommendation / done / error")
    data: Optional[dict] = Field(default=None, description="事件负载数据")


# ==========================================================================
# 接口
# ==========================================================================


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """同步对话接口。

    接收用户消息，调用 Agent 进行工具搜索和推理，
    返回完整回复和结构化推荐。

    示例请求:
        POST /chat
        {
            "user_input": "推荐一款20万的混动SUV",
            "thread_id": "user_001"
        }
    """
    thread_id = request.thread_id or f"session_{uuid.uuid4().hex[:8]}"
    logger.info("POST /chat: thread=%s input=%.60s", thread_id, request.user_input)

    # 从 DB 加载历史 + 追加新消息 = 完整上下文
    # 过滤掉 tool 消息，只保留 user/assistant（避免 tool_calls 序列损坏）
    history = get_messages(thread_id)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history:
        if h["role"] in ("user", "assistant"):
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": request.user_input})

    try:
        result = await agent_service.ainvoke(messages=messages)
    except Exception as exc:
        logger.exception("POST /chat failed")
        raise HTTPException(status_code=500, detail=f"Agent 调用失败: {exc}")

    # 保存到 DB
    save_session(thread_id, request.user_input, request.user_input[:24])
    save_message(thread_id, "user", request.user_input)
    save_message(thread_id, "assistant", result["response"], result.get("final_recommendation"))

    return ChatResponse(
        response=result["response"],
        final_recommendation=result.get("final_recommendation"),
        thread_id=thread_id,
        elapsed_ms=result["elapsed_ms"],
    )


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """流式对话接口（Server-Sent Events）。

    逐步返回 Agent 的执行过程：
    工具调用 → 搜索结果 → 助手回复 → 最终推荐。

    前端可用 EventSource 接收：
        const es = new EventSource("/chat/stream");
        es.onmessage = (e) => console.log(JSON.parse(e.data));

    或使用 fetch + ReadableStream：
        const res = await fetch("/chat/stream", {
            method: "POST",
            body: JSON.stringify({user_input: "..."}),
        });
    """
    thread_id = request.thread_id or f"session_{uuid.uuid4().hex[:8]}"
    logger.info("POST /chat/stream: thread=%s input=%.60s", thread_id, request.user_input)

    # 从 DB 加载历史 + 追加新消息（过滤 tool 消息）
    history = get_messages(thread_id)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history:
        if h["role"] in ("user", "assistant"):
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": request.user_input})

    async def event_generator():
        try:
            # 保存用户消息
            save_session(thread_id, request.user_input, request.user_input[:24])
            save_message(thread_id, "user", request.user_input)

            assistant_response = ""
            assistant_rec = None

            async for chunk in agent_service.astream(messages=messages):
                event_type = chunk.get("event", "message")
                data = chunk.get("data") or {}
                if event_type == "message":
                    assistant_response = data.get("content", "")
                elif event_type == "final_recommendation":
                    assistant_rec = data
                elif event_type == "done":
                    save_message(thread_id, "assistant", assistant_response, assistant_rec)
                yield {
                    "event": event_type,
                    "data": StreamEvent(event=event_type, data=data).model_dump_json(),
                }
        except Exception as exc:
            logger.exception("POST /chat/stream failed")
            yield {
                "event": "error",
                "data": StreamEvent(
                    event="error",
                    data={"message": str(exc)},
                ).model_dump_json(),
            }

    return EventSourceResponse(event_generator())
