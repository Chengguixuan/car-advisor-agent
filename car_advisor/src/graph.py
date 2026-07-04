"""购车智能体 LangGraph 状态图。

构建完整的 Agent 工作流：
    START → chatbot ⇄ tools → finalize → END

chatbot 节点负责对话和工具调用决策，tools 节点执行搜索/对比，
finalize 节点生成结构化的最终推荐。
"""

import json
import logging
import re
from typing import Any, Literal

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .config import load_config
from .http_client import get_http_client
from .prompts import SYSTEM_PROMPT
from .state import CarAdvisorState
from .tools import tools
from .utils import get_role

logger = logging.getLogger(__name__)

# 最大搜索次数，防止 Agent 无限循环
_MAX_SEARCHES = 6

# 上下文压缩阈值（消息条数超过此值触发压缩）
_COMPRESS_THRESHOLD = 10          # 5 轮对话
_COMPRESS_KEEP_LAST = 10          # 最近 10 条消息不压缩
# 强制调用 search_online 的关键词
_SEARCH_ONLINE_KEYWORDS = ["优惠", "降价", "行情", "多少钱", "口碑", "评价", "车主", "销量", "新闻", "促销"]


def _should_force_search_online(user_msg: str) -> bool:
    return any(kw in user_msg for kw in _SEARCH_ONLINE_KEYWORDS)


_SUMMARY_PROMPT = (
    "请用2-3句话总结以下对话的核心信息，包括用户需求、排除项和已推荐的车型。"
    "只输出摘要文本，不要包含任何其他内容。"
)


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


# 参数筛选关键词（用户条件 → 从 candidates 中语义匹配）
_PARAM_PATTERNS = [
    (r"(百公里|零百|0-?100|零到一百)\s*(加速|提速).*?(\d+[\.]?\d*)\s*秒", "加速"),
    (r"加速.*?(\d+[\.]?\d*)\s*秒", "加速"),
    (r"(\d+[\.]?\d*)\s*秒.*?(破百|加速)", "加速"),
    (r"续航.*?(\d+)\s*(公里|km)", "续航"),
    (r"(\d+)\s*(公里|km).*?续航", "续航"),
    (r"油耗.*?(\d+[\.]?\d*)\s*[Ll升]", "油耗"),
    (r"电耗.*?(\d+[\.]?\d*)", "电耗"),
    (r"空间.*?[大宽长]", "空间"),
    (r"后备箱.*?(\d+)", "后备箱"),
    (r"轴距.*?(\d+)", "轴距"),
]


def _detect_param_keywords(text: str) -> list:
    """检测用户消息中的参数条件，返回 [(类型, 原文), ...] 列表。"""
    found = []
    for pattern, ptype in _PARAM_PATTERNS:
        match = re.search(pattern, text)
        if match:
            found.append((ptype, match.group(0)))
    return found


def _filter_candidates_by_params(
    candidates: list[dict],
    param_conditions: list,
    user_query: str,
    llm,
) -> list[dict]:
    """用 LLM 从 candidates 中筛选符合参数条件的车型。

    筛选失败时降级返回原始列表。
    """
    if not candidates or not param_conditions:
        return candidates

    car_lines = []
    for i, car in enumerate(candidates):
        name = car.get("name", "?")
        specs = car.get("specs", {})
        spec_str = " | ".join(f"{k}: {v}" for k, v in specs.items())
        fuel_econ = car.get("fuel_economy", car.get("fuel", ""))
        car_lines.append(f"{i+1}. {name} — {car.get('price_range','?')} | {fuel_econ} | {spec_str}")

    conditions_str = "; ".join(f"{kw}: {raw}" for kw, raw in param_conditions)
    filter_prompt = (
        f"用户条件：{conditions_str}\n"
        f"原始查询：{user_query}\n\n"
        f"候选车型：\n" + "\n".join(car_lines) + "\n\n"
        "请从以上候选车型中选出最符合用户参数条件的车型。\n"
        "只返回车型编号，按匹配程度排序，用逗号分隔。例如：3, 1, 5"
    )

    try:
        resp = llm.invoke([{"role": "user", "content": filter_prompt}])
        content = getattr(resp, "content", "") or ""
        logger.info("param_filter: LLM response — %s", content[:100])

        ids = [int(s) for s in re.findall(r"\d+", content)]
        filtered = [candidates[i - 1] for i in ids if 1 <= i <= len(candidates)]

        if filtered:
            logger.info("param_filter: %d → %d candidates", len(candidates), len(filtered))
            return filtered
    except Exception as e:
        logger.warning("param_filter: failed — %s, keeping all", e)

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
        编译后的 StateGraph（无 checkpointer，调用方传入完整历史）。
    """
    llm = _build_llm()
    model_with_tools = llm.bind_tools(tools)
    tool_executor = ToolNode(tools)

    # ------------------------------------------------------------------
    # 内部：上下文压缩
    # ------------------------------------------------------------------

    def _compress_messages(all_messages: list) -> list:
        """压缩消息历史：保留 system + 最近 N 条，中间生成摘要。

        原始: [sys, m1, m2, m3, ..., m12]
        压缩: [sys, summary_msg, m11, m12]   (保留最近 2 条示例)

        Args:
            all_messages: 完整消息列表（至少含 system prompt）。

        Returns:
            压缩后的消息列表。如果不需要压缩，返回原列表。
        """
        total = len(all_messages)
        if total <= _COMPRESS_THRESHOLD:
            return all_messages

        # system prompt 始终保留
        system_msg = all_messages[0]
        # 中间部分：需要压缩的消息
        middle = all_messages[1:total - _COMPRESS_KEEP_LAST]
        # 最近的消息：保留不压缩
        recent = all_messages[total - _COMPRESS_KEEP_LAST:]

        if not middle:
            return all_messages

        logger.info(
            "_compress: total=%d → keep=%d(system)+%d(recent), summarize=%d middle msgs",
            total, 1, len(recent), len(middle),
        )

        # 构建摘要请求
        summary_input = [{"role": "user", "content": _SUMMARY_PROMPT}] + list(middle)
        try:
            summary_resp = llm.invoke(summary_input)
            summary_text = getattr(summary_resp, "content", "") or ""
        except Exception as e:
            logger.warning("_compress: summary generation failed: %s, truncating instead", e)
            summary_text = "（对话历史过长，已截断早期内容）"

        if not summary_text.strip():
            summary_text = "（对话历史过长，已截断早期内容）"

        summary_msg = {
            "role": "system",
            "content": f"[历史摘要] {summary_text.strip()}",
        }

        compressed = [system_msg, summary_msg] + recent
        logger.info(
            "_compress: done — %d msgs → %d msgs (saved %d tokens)",
            total, len(compressed), total - len(compressed),
        )
        return compressed

    # ------------------------------------------------------------------
    # 节点定义
    # ------------------------------------------------------------------

    def chatbot(state: CarAdvisorState) -> dict[str, Any]:
        """对话节点：调用 LLM，根据 intent 分流处理。"""
        messages = state["messages"]

        if not messages or get_role(messages[0]) != "system":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(messages)

        # 上下文压缩：超过阈值时自动压缩早期消息
        original_count = len(messages)
        messages = _compress_messages(messages)
        compression_applied = len(messages) < original_count

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

        # 保存压缩摘要
        if compression_applied and len(messages) >= 2:
            second = messages[1]
            if isinstance(second, dict) and "[历史摘要]" in second.get("content", ""):
                result["history_summary"] = second["content"]

        tool_calls = getattr(response, "tool_calls", []) or []

        # 强制路由：用户问实时信息但 LLM 没调 search_online（仅首次）
        already_called = state.get("called_tools", [])
        if not tool_calls and "search_online" not in already_called:
            last_user_msg = ""
            for m in reversed(messages):
                r = get_role(m)
                if r in ("user", "human"):
                    last_user_msg = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
                    break
            if _should_force_search_online(last_user_msg):
                logger.info("chatbot: forcing search_online for: %.60s", last_user_msg)
                from langchain_core.messages import AIMessage
                forced_msg = AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "search_online",
                        "args": {"query": last_user_msg},
                        "id": "force_search_online",
                        "type": "tool_call",
                    }]
                )
                return {"messages": [forced_msg]}

        logger.info(
            "chatbot: role=%s, tool_calls=%d, compressed=%s",
            getattr(response, "role", "?"), len(tool_calls), compression_applied,
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

        # 提取候选车型 + 记录调用的工具名
        all_messages = list(state.get("messages", [])) + result.get("messages", [])
        candidates = _parse_candidates_from_messages(all_messages)

        called = list(state.get("called_tools", []))
        for msg in result.get("messages", []):
            tname = getattr(msg, "name", None) or (msg.get("name", "") if isinstance(msg, dict) else "")
            if tname and tname not in called:
                called.append(tname)

        result["candidates"] = candidates if candidates else state.get("candidates")
        result["search_count"] = current_count + 1
        result["called_tools"] = called

        logger.info(
            "tool_node: done — %d candidates found, search_count=%d",
            len(candidates), result["search_count"],
        )

        return result

    def finalize(state: CarAdvisorState) -> dict[str, Any]:
        """推荐生成节点：综合所有信息输出最终购车建议。

        如果用户的最后一条消息中包含参数条件（加速/空间/续航/油耗等），
        会先用 LLM 从 candidates 中筛选匹配的车型，再生成推荐。
        """
        logger.info("finalize: generating recommendation")

        candidates = state.get("candidates") or []
        messages = list(state.get("messages", []))

        # ---- 参数筛选 ----
        if candidates:
            last_user = ""
            for m in reversed(messages):
                role = get_role(m)
                if role in ("user", "human"):
                    last_user = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
                    break

            param_keywords = _detect_param_keywords(last_user)
            if param_keywords:
                logger.info("finalize: param filter detected — %s", param_keywords)
                candidates = _filter_candidates_by_params(candidates, param_keywords, last_user, llm)

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

        candidate_count = len(candidates)
        table_hint = ""
        if candidate_count >= 2:
            table_hint = (
                "同时必须输出 tradeoff_summary（2-4句权衡说明）和 comparison_table（参数对比表格）。"
            )
        finalize_prompt = (
            f"{summary}\n\n"
            "请根据以上候选车型和之前的对话内容，生成最终的购车推荐。"
            "以 JSON 格式输出，包含 understanding（需求理解）、"
            "recommended_models（推荐车型列表，每款含 name/price_range/pros/cons/reason）、"
            f"follow_up_question（信息不足时追问，否则 null）。{table_hint}"
            "注意：如果用户指定了具体车型进行对比，recommended_models 只能包含"
            "用户指定的车型，不得追加任何新车。"
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

    return graph.compile()
