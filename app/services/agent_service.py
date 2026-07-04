"""Agent 调用封装。

将 LangGraph Agent 包装为可复用的服务层，提供：
- 单例 agent 实例
- 同步/异步调用接口
- 流式输出支持
- 请求日志和耗时统计
"""

import logging
import time
import sys
from pathlib import Path
from typing import Any, AsyncIterator

# 确保 car_advisor 包可导入
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from car_advisor.src.graph import build_agent
from car_advisor.src.utils import serialize_messages

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

    def invoke(self, messages: list[dict]) -> dict[str, Any]:
        """同步调用 Agent，传入完整消息历史。

        Args:
            messages: 完整消息列表 [{"role":..., "content":...}, ...]。

        Returns:
            {"response": ..., "final_recommendation": ..., "messages": ..., "elapsed_ms": ...}
        """
        agent = self.get_agent()
        t0 = time.perf_counter()
        logger.info("invoke start: msgs=%d", len(messages))

        result = agent.invoke({"messages": messages})
        elapsed_ms = (time.perf_counter() - t0) * 1000
        all_msgs = result.get("messages", [])
        rec = result.get("final_recommendation")
        response = _extract_response(all_msgs)
        logger.info("invoke done: msgs=%d elapsed=%.0fms", len(all_msgs), elapsed_ms)
        return {"response": response, "final_recommendation": rec, "messages": all_msgs, "elapsed_ms": round(elapsed_ms, 1)}

    async def ainvoke(self, messages: list[dict]) -> dict[str, Any]:
        """异步调用 Agent。"""
        agent = self.get_agent()
        t0 = time.perf_counter()
        result = await agent.ainvoke({"messages": messages})
        elapsed_ms = (time.perf_counter() - t0) * 1000
        all_msgs = result.get("messages", [])
        rec = result.get("final_recommendation")
        response = _extract_response(all_msgs)
        return {"response": response, "final_recommendation": rec, "messages": all_msgs, "elapsed_ms": round(elapsed_ms, 1)}

    async def astream(self, messages: list[dict]) -> AsyncIterator[dict[str, Any]]:
        """异步流式调用，传入完整消息历史。

        Yields: start → tool_call → tool_result → message → final_recommendation → done
        """
        agent = self.get_agent()
        t0 = time.perf_counter()
        try:
            yield {"event": "start", "data": None}
            async for step in agent.astream({"messages": messages}, stream_mode="updates"):
                for node_name, node_output in step.items():
                    messages = node_output.get("messages", [])

                    if node_name == "tools":
                        # 工具执行完成
                        yield {
                            "event": "tool_result",
                            "data": {
                                "node": node_name,
                                "messages": serialize_messages(messages),
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
            logger.info("astream done: elapsed=%.0fms", elapsed_ms)

        except Exception as exc:
            logger.exception("astream error")
            yield {"event": "error", "data": {"message": str(exc)}}


def _extract_response(messages: list) -> str:
    """从消息列表中提取最后一条助手回复文本。"""
    for m in reversed(messages):
        role = getattr(m, "type", "") or getattr(m, "role", "")
        if role in ("ai", "assistant"):
            return getattr(m, "content", "") or ""
    return ""
