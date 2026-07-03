"""购车智能体 LangGraph 状态图。

构建完整的 Agent 工作流：
    START → chatbot ⇄ tools → finalize → END

chatbot 节点负责对话和工具调用决策，tools 节点执行搜索/对比，
finalize 节点生成结构化的最终推荐。
"""

import json
import logging
from typing import Any, Literal

from langchain_openai import ChatOpenAI
import sqlite3
from pathlib import Path
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .config import load_config
from .http_client import get_http_client
from .prompts import SYSTEM_PROMPT
from .state import CarAdvisorState
from .tools import tools

logger = logging.getLogger(__name__)

# 最大搜索次数，防止 Agent 无限循环
_MAX_SEARCHES = 3


# ==========================================================================
# 内部辅助
# ==========================================================================


def _build_llm() -> ChatOpenAI:
    """根据项目配置创建 ChatOpenAI 实例（兼容 DeepSeek API）。"""
    cfg = load_config().llm
    return ChatOpenAI(
        model=cfg.model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        http_client=get_http_client(),
    )


def _parse_candidates_from_messages(messages: list) -> list[dict[str, Any]]:
    """从对话消息中提取工具返回的候选车型。

    遍历消息列表，找到所有 tool 角色的消息（ToolMessage），
    尝试解析其中的 JSON 数组并合并车型数据。

    Args:
        messages: 对话消息列表。

    Returns:
        候选车型字典列表。
    """
    candidates: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for msg in messages:
        role = getattr(msg, "role", "") or getattr(msg, "type", "")
        if role not in ("tool", "function"):
            continue

        content = getattr(msg, "content", "") or ""
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            continue

        # 可能是 {"matched": [...], "unmatched": [...]}（compare_cars 格式）
        # 或直接是车型列表（search_local_cars 格式）
        items: list[dict] = []
        if isinstance(parsed, dict):
            items = parsed.get("matched", []) or parsed.get("results", [])
        elif isinstance(parsed, list):
            items = parsed

        for item in items:
            name = item.get("name", "")
            if name and name not in seen_names:
                seen_names.add(name)
                candidates.append(item)

    return candidates


# ==========================================================================
# Agent 构建
# ==========================================================================


def build_agent() -> StateGraph:
    """构建并返回编译后的购车智能体 LangGraph。

    工作流：
        1. chatbot — LLM 接收消息，决定是调用工具还是结束
        2. tools   — 执行工具调用（搜索/对比），增量 search_count
        3. finalize — 综合所有信息，生成结构化推荐 JSON

    防无限循环：
        search_count 达到 _MAX_SEARCHES 后，强制路由到 finalize。

    Returns:
        编译后的 StateGraph，含 SqliteSaver checkpointer。
    """
    llm = _build_llm()
    model_with_tools = llm.bind_tools(tools)
    tool_executor = ToolNode(tools)

    # ------------------------------------------------------------------
    # 节点定义
    # ------------------------------------------------------------------

    def chatbot(state: CarAdvisorState) -> dict[str, Any]:
        """对话节点：调用 LLM，根据 intent 分流处理。"""
        messages = state["messages"]

        if not messages or _get_role(messages[0]) != "system":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(messages)

        logger.info("chatbot: invoking with %d messages", len(messages))
        try:
            response = model_with_tools.invoke(messages)
        except Exception as exc:
            logger.exception("chatbot: LLM invocation failed")
            from langchain_core.messages import AIMessage
            return {"messages": [AIMessage(content=f"抱歉，调用模型时出现错误：{exc}")]}

        # 解析 LLM 响应中的 intent
        content = getattr(response, "content", "") or ""
        result: dict[str, Any] = {"messages": [response]}

        try:
            parsed = json.loads(content)
            intent = parsed.get("intent", "search")
            logger.info("chatbot: intent=%s", intent)

            if intent == "preference":
                # 更新 exclusions / preferences，不调工具
                exclusions = state.get("exclusions", [])
                preferences = state.get("preferences", [])
                understanding = parsed.get("understanding", "")

                # 从用户消息中提取排除项和偏好
                if "日系" in content or "日本" in content:
                    exclusions.append("日系")
                if "纯电" in content:
                    exclusions = list(set(exclusions + ["纯电"])) if "不" not in content else exclusions

                result["exclusions"] = exclusions
                result["preferences"] = preferences
                logger.info("chatbot: updated exclusions=%s, preferences=%s", exclusions, preferences)

            elif intent == "opinion":
                # 记录用户对具体车型的评价
                opinions = dict(state.get("car_opinions", {}))
                understanding = parsed.get("understanding", "")
                recommendations = parsed.get("recommended_models", [])
                for m in recommendations:
                    name = m.get("name", "")
                    if name and name not in opinions:
                        opinions[name] = understanding
                result["car_opinions"] = opinions
                logger.info("chatbot: recorded opinion about %d cars", len(opinions))

            # search / question → 正常流程，由 tools_condition 路由

        except (json.JSONDecodeError, KeyError, TypeError):
            pass  # 非 JSON 响应，不影响正常流程

        tool_calls = getattr(response, "tool_calls", []) or []
        logger.info(
            "chatbot: role=%s, tool_calls=%d",
            getattr(response, "role", "?"), len(tool_calls),
        )

        return result

    def tool_node(state: CarAdvisorState) -> dict[str, Any]:
        """工具执行节点：运行 ToolNode 并更新计数器和候选车型。"""
        current_count = state.get("search_count", 0)
        logger.info("tool_node: executing tools (search #%d)", current_count + 1)

        try:
            result = tool_executor.invoke(state)
        except Exception as exc:
            logger.exception("tool_node: tool execution failed")
            # 工具失败时返回错误信息并跳过此轮
            from langchain_core.messages import ToolMessage, AIMessage
            last_msg = state["messages"][-1] if state.get("messages") else None
            tool_call_id = ""
            if last_msg and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                tool_call_id = getattr(last_msg.tool_calls[0], "id", "")
            return {
                "messages": [ToolMessage(content=f"工具执行失败：{exc}", tool_call_id=tool_call_id)],
                "search_count": current_count + 1,
            }

        # 提取候选车型
        all_messages = list(state.get("messages", [])) + result.get("messages", [])
        candidates = _parse_candidates_from_messages(all_messages)

        result["candidates"] = candidates if candidates else state.get("candidates")
        result["search_count"] = current_count + 1

        logger.info(
            "tool_node: done — %d candidates found, search_count=%d",
            len(candidates), result["search_count"],
        )

        return result

    def finalize(state: CarAdvisorState) -> dict[str, Any]:
        """推荐生成节点：综合所有信息输出最终购车建议。"""
        logger.info("finalize: generating recommendation")

        candidates = state.get("candidates") or []
        messages = list(state.get("messages", []))

        # 构建候选车型摘要
        if candidates:
            summary_lines = ["以下是通过搜索找到的候选车型："]
            for i, car in enumerate(candidates, 1):
                pros = ", ".join(car.get("pros", [])[:3])
                cons = ", ".join(car.get("cons", [])[:2])
                summary_lines.append(
                    f"{i}. {car.get('name', '?')} — "
                    f"{car.get('price_range', '?')} | "
                    f"{car.get('fuel', '?')} | "
                    f"优点: {pros} | 缺点: {cons}"
                )
            summary = "\n".join(summary_lines)
        else:
            summary = "未找到匹配的候选车型，请根据对话内容给出建议。"

        finalize_prompt = (
            f"{summary}\n\n"
            "请根据以上候选车型和之前的对话内容，生成最终的购车推荐。"
            "以 JSON 格式输出，包含 understanding（需求理解）、"
            "recommended_models（推荐车型列表，每款含 name/price_range/pros/cons/reason）、"
            "follow_up_question（信息不足时追问，否则 null）。"
        )

        # 过滤消息：仅保留 system / user / assistant-text 供 LLM 生成推荐
        # LangChain Message 类型：human / ai / system / tool
        # 去掉 tool 消息和含 tool_calls 的 ai 消息
        filtered: list[dict] = []
        for m in messages:
            if isinstance(m, dict):
                role = m.get("role", "")
                has_tc = bool(m.get("tool_calls"))
            else:
                role = getattr(m, "type", "") or getattr(m, "role", "")
                has_tc = bool(getattr(m, "tool_calls", None) or getattr(m, "tool_calls", None))

            if role in ("tool", "function"):
                continue
            if role in ("ai", "assistant") and has_tc:
                continue

            filtered.append(m)

        final_messages = filtered + [{"role": "user", "content": finalize_prompt}]

        try:
            response = llm.invoke(final_messages)
            content = getattr(response, "content", "") or ""
        except Exception as exc:
            logger.exception("finalize: LLM invocation failed")
            return {"final_recommendation": {"raw": str(exc), "parse_error": True}}

        # 尝试解析为 JSON
        import re
        try:
            recommendation = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
            if match:
                try:
                    recommendation = json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    recommendation = {"raw": content, "parse_error": True}
            else:
                recommendation = {"raw": content, "parse_error": True}

        logger.info("finalize: done")
        return {"final_recommendation": recommendation}

    # ------------------------------------------------------------------
    # 路由函数
    # ------------------------------------------------------------------

    def route_after_chatbot(
        state: CarAdvisorState,
    ) -> Literal["tools", "finalize"]:
        """决定 chatbot 之后的流向。

        - 有未执行的 tool_calls 且 search_count 未超限 → tools
        - 无 tool_calls 或 search_count 超限 → finalize
        """
        messages = state.get("messages", [])
        search_count = state.get("search_count", 0)

        # 使用 LangGraph 内置的 tools_condition 判断
        condition = tools_condition(state)

        if condition == "tools":
            if search_count >= _MAX_SEARCHES:
                logger.warning(
                    "route: tool_calls pending but search_count=%d >= %d — "
                    "forcing finalize anyway (unresolved tool_calls will be stripped)",
                    search_count, _MAX_SEARCHES,
                )
                return "finalize"
            logger.info("route: tools_condition=%s → tools", condition)
            return "tools"

        logger.info("route: no tool_calls → finalize")
        return "finalize"

    # ------------------------------------------------------------------
    # 图构建
    # ------------------------------------------------------------------

    graph = StateGraph(CarAdvisorState)

    graph.add_node("chatbot", chatbot)
    graph.add_node("tools", tool_node)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "chatbot")
    graph.add_conditional_edges(
        "chatbot",
        route_after_chatbot,
        {"tools": "tools", "finalize": "finalize"},
    )
    graph.add_edge("tools", "chatbot")
    graph.add_edge("finalize", END)

    # SqliteSaver 持久化到本地文件，重启后保留对话历史
    db_path = Path(__file__).resolve().parent.parent.parent / "checkpoints.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return graph.compile(checkpointer=checkpointer)


def _get_role(msg) -> str:
    """兼容 dict 和 Message 对象，获取消息的 role。"""
    if isinstance(msg, dict):
        return msg.get("role", "")
    return getattr(msg, "role", "") or ""
