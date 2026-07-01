"""Agent 调用封装。

将 LangGraph Agent 包装为可复用的服务层，提供：
- 单例 agent 实例（共享 MemorySaver）
- 同步/异步调用接口
- 流式输出支持
- 请求日志和耗时统计
"""

import logging
import time
import sys
from pathlib import Path
from typing import Any, AsyncIterator

from langchain_core.messages import HumanMessage

# 确保 car_advisor 包可导入
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from car_advisor.src.graph import build_agent
from car_advisor.src.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class AgentService:
    """购车智能体服务层。

    封装 LangGraph Agent 的创建、调用和流式输出，
    确保每个请求使用独立的 thread_id 实现会话隔离。

    用法:
        service = AgentService()
        result = service.invoke("推荐一款20万的SUV", thread_id="user_001")
        async for chunk in service.astream("推荐...", thread_id="user_001"):
            ...
    """

    def __init__(self):
        self._agent = None

    # ------------------------------------------------------------------
    # Agent 单例
    # ------------------------------------------------------------------

    def get_agent(self):
        """返回缓存的 Agent 实例（延迟初始化）。"""
        if self._agent is None:
            logger.info("initializing LangGraph agent (first request)")
            self._agent = build_agent()
        return self._agent

    # ------------------------------------------------------------------
    # 同步调用
    # ------------------------------------------------------------------

    def invoke(
        self,
        user_input: str,
        thread_id: str = "default",
    ) -> dict[str, Any]:
        """同步调用 Agent，返回完整结果。

        Args:
            user_input: 用户消息文本。
            thread_id:  会话 ID（同一 id 共享上下文）。

        Returns:
            包含以下字段的字典：
            - response:           最终消息文本（JSON 字符串）
            - final_recommendation: 结构化推荐结果（如有）
            - messages:           完整对话历史
            - elapsed_ms:         耗时（毫秒）
        """
        agent = self.get_agent()
        config = {"configurable": {"thread_id": thread_id}}

        t0 = time.perf_counter()
        logger.info("invoke start: thread=%s input=%.60s", thread_id, user_input)

        input_state = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                HumanMessage(content=user_input),
            ],
        }

        result = agent.invoke(input_state, config=config)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        messages = result.get("messages", [])
        rec = result.get("final_recommendation")
        msg_count = len(messages)

        # 提取最后一条助手消息
        response = ""
        for m in reversed(messages):
            role = getattr(m, "type", "") or getattr(m, "role", "")
            if role in ("ai", "assistant"):
                response = getattr(m, "content", "") or ""
                break

        logger.info(
            "invoke done: thread=%s msgs=%d elapsed=%.0fms has_rec=%s",
            thread_id, msg_count, elapsed_ms, rec is not None,
        )

        return {
            "response": response,
            "final_recommendation": rec,
            "messages": messages,
            "elapsed_ms": round(elapsed_ms, 1),
        }

    # ------------------------------------------------------------------
    # 异步调用
    # ------------------------------------------------------------------

    async def ainvoke(
        self,
        user_input: str,
        thread_id: str = "default",
    ) -> dict[str, Any]:
        """异步调用 Agent，返回完整结果。

        与 invoke() 参数和返回值相同，适用于 FastAPI async 路由。
        """
        agent = self.get_agent()
        config = {"configurable": {"thread_id": thread_id}}

        t0 = time.perf_counter()
        logger.info("ainvoke start: thread=%s input=%.60s", thread_id, user_input)

        input_state = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                HumanMessage(content=user_input),
            ],
        }

        result = await agent.ainvoke(input_state, config=config)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        messages = result.get("messages", [])
        rec = result.get("final_recommendation")

        response = ""
        for m in reversed(messages):
            role = getattr(m, "type", "") or getattr(m, "role", "")
            if role in ("ai", "assistant"):
                response = getattr(m, "content", "") or ""
                break

        logger.info(
            "ainvoke done: thread=%s msgs=%d elapsed=%.0fms",
            thread_id, len(messages), elapsed_ms,
        )

        return {
            "response": response,
            "final_recommendation": rec,
            "messages": messages,
            "elapsed_ms": round(elapsed_ms, 1),
        }

    # ------------------------------------------------------------------
    # 流式输出
    # ------------------------------------------------------------------

    async def astream(
        self,
        user_input: str,
        thread_id: str = "default",
    ) -> AsyncIterator[dict[str, Any]]:
        """异步流式调用 Agent，逐步返回状态变化。

        每次 yield 一个状态块，前端可据此展示实时进度。

        Yields:
            {"event": str, "data": Any} 格式的字典：
            - event="start":     流开始
            - event="tool_call": LLM 请求调用工具（含 tools 列表）
            - event="tool_result": 工具执行结果（含 results）
            - event="message":   助手文本消息
            - event="final_recommendation": 最终推荐
            - event="done":      流结束（含 elapsed_ms）
            - event="error":     错误
        """
        agent = self.get_agent()
        config = {"configurable": {"thread_id": thread_id}}

        t0 = time.perf_counter()
        logger.info("astream start: thread=%s input=%.60s", thread_id, user_input)

        input_state = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                HumanMessage(content=user_input),
            ],
        }

        try:
            yield {"event": "start", "data": None}

            async for step in agent.astream(input_state, config=config, stream_mode="updates"):
                for node_name, node_output in step.items():
                    messages = node_output.get("messages", [])

                    if node_name == "tools":
                        # 工具执行完成
                        yield {
                            "event": "tool_result",
                            "data": {
                                "node": node_name,
                                "messages": _serialize_messages(messages),
                            },
                        }

                    elif node_name == "chatbot":
                        last_msg = messages[-1] if messages else None
                        tool_calls = getattr(last_msg, "tool_calls", None) if last_msg else None

                        if tool_calls:
                            yield {
                                "event": "tool_call",
                                "data": {
                                    "tools": [
                                        {
                                            "name": getattr(tc, "name", ""),
                                            "args": getattr(tc, "args", {}),
                                        }
                                        for tc in tool_calls
                                    ],
                                },
                            }
                        else:
                            content = getattr(last_msg, "content", "") if last_msg else ""
                            if content:
                                yield {
                                    "event": "message",
                                    "data": {"content": content},
                                }

                    elif node_name == "finalize":
                        rec = node_output.get("final_recommendation")
                        if rec:
                            yield {
                                "event": "final_recommendation",
                                "data": rec,
                            }

            elapsed_ms = (time.perf_counter() - t0) * 1000
            yield {"event": "done", "data": {"elapsed_ms": round(elapsed_ms, 1)}}
            logger.info("astream done: thread=%s elapsed=%.0fms", thread_id, elapsed_ms)

        except Exception as exc:
            logger.exception("astream error: thread=%s", thread_id)
            yield {"event": "error", "data": {"message": str(exc)}}


# ------------------------------------------------------------------
# 辅助函数
# ------------------------------------------------------------------


def _serialize_messages(messages: list) -> list[dict]:
    """将 LangChain Message 对象序列化为可 JSON 化的 dict。"""
    result = []
    for msg in messages:
        result.append({
            "role": getattr(msg, "type", "unknown"),
            "content": getattr(msg, "content", ""),
        })
    return result
