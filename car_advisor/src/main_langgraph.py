"""LangGraph 版本 — 命令行交互入口（v2）。

使用 LangGraph 状态图替代单轮 LLM 调用，支持：
- 工具调用（搜索 / 对比 / 在线查询）
- 多步推理（chatbot ⇄ tools 循环）
- 流式输出，每步可见 + 信息来源标注
- 结构化最终推荐
- 多轮对话记忆（传入完整历史）
"""

import json
import logging
from typing import Any

from .config import load_config
from .display import (
    BOLD, BLUE, CYAN, DIM, GREEN, RED, RESET, YELLOW,
    display_recommendation,
    is_clear_command,
    is_exit_command,
    is_help_command,
    is_empty_or_whitespace,
    print_error,
    print_warning,
)
from .graph import build_agent
from .prompts import SYSTEM_PROMPT
from .utils import get_role, get_content, get_tool_calls

logger = logging.getLogger(__name__)

WELCOME_MSG = f"""
{BOLD}{BLUE}🚗 买车智能体 v2 (LangGraph){RESET}

我现在的能力更强了：
  • 结构化搜索 — 精确匹配预算和车型
  • 多款车型参数对比
  • 多步推理，综合分析后给出推荐

{DIM}输入 {BOLD}help{DIM} 查看帮助 | 输入 {BOLD}exit{DIM} 退出{RESET}
"""

HELP_MSG = f"""
{BOLD}可用指令：{RESET}
  {GREEN}help{RESET}    显示帮助
  {GREEN}clear{RESET}   清空对话历史
  {GREEN}exit{RESET}    退出程序

{DIM}直接描述购车需求即可，Agent 会自动搜索和对比。{RESET}
"""

EXIT_MSG = f"\n{BOLD}{GREEN}感谢使用买车智能体，祝你选到心仪的爱车！🚗{RESET}\n"


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------


def _extract_source_from_tool_result(data: dict | list) -> str:
    """从工具返回的 JSON 中提取来源标签。"""
    if isinstance(data, dict):
        src = data.get("source", "")
        if src:
            return src
        if "matched" in data:
            return "本地数据库"
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        if "source" in data[0]:
            return "语义检索"
    return ""


def main() -> None:
    """LangGraph v2 入口。"""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config()
    if not config.llm.api_key:
        print_warning("未设置 DEEPSEEK_API_KEY，请检查 .env 文件")
        return

    agent = build_agent()
    history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    print(WELCOME_MSG)

    while True:
        try:
            user_input = input(f"\n{BOLD}{BLUE}你：{RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(EXIT_MSG)
            break

        if is_empty_or_whitespace(user_input):
            print(f"{DIM}请输入你的购车需求，或输入 help 查看帮助。{RESET}")
            continue

        if is_exit_command(user_input):
            print(EXIT_MSG)
            break

        if is_help_command(user_input):
            print(HELP_MSG)
            continue

        if is_clear_command(user_input):
            history = [{"role": "system", "content": SYSTEM_PROMPT}]
            print(f"{DIM}对话历史已清空。{RESET}")
            continue

        print(f"\n{DIM}分析中...{RESET}")

        try:
            history.append({"role": "user", "content": user_input})
            input_state: dict[str, Any] = {"messages": list(history)}

            tool_phase = False

            for step in agent.stream(input_state, stream_mode="values"):
                messages: list = step.get("messages", [])
                if not messages:
                    continue

                last_msg = messages[-1]
                role = get_role(last_msg)
                content = get_content(last_msg)
                tool_calls = get_tool_calls(last_msg)

                # --- 工具调用 ---
                if tool_calls:
                    if not tool_phase:
                        tool_phase = True
                        print(f"\n{DIM}🔧 正在查询车型数据...{RESET}")
                    for tc in tool_calls:
                        t_name = (
                            tc.get("name", "") if isinstance(tc, dict)
                            else getattr(tc, "name", "")
                        )
                        if t_name:
                            print(f"{DIM}  → 调用 {t_name}{RESET}")
                    continue

                # --- 工具返回 ---
                if role in ("tool", "function") and content:
                    tool_phase = False
                    try:
                        data = json.loads(content)
                    except json.JSONDecodeError:
                        continue

                    source_label = _extract_source_from_tool_result(data)

                    if isinstance(data, list):
                        if data:
                            src = f"（{source_label}）" if source_label else ""
                            print(f"{DIM}  ✅ 找到 {len(data)} 款候选车型 {src}{RESET}")
                    elif isinstance(data, dict):
                        results = data.get("results", [])
                        matched = data.get("matched", [])
                        if results or matched:
                            items = results or matched
                            src = f"（{source_label}）" if source_label else ""
                            print(f"{DIM}  ✅ 找到 {len(items)} 条结果 {src}{RESET}")
                    continue

                # --- 助手文本回复 ---
                if role == "assistant" and content:
                    if content.strip().startswith("{"):
                        print(f"\n{BOLD}{GREEN}顾问：{RESET}")
                        try:
                            parsed = json.loads(content)
                            display_recommendation(parsed)
                        except json.JSONDecodeError:
                            print(f"{DIM}{content}{RESET}")
                    else:
                        print(f"\n{BOLD}{GREEN}顾问：{RESET}{content}")

                # --- 最终推荐 ---
                rec = step.get("final_recommendation")
                if rec:
                    print(f"\n{BOLD}🎯 最终推荐：{RESET}")
                    display_recommendation(rec)

            # 将助手回复存入本地历史
            msgs = step.get("messages", []) if step else []
            for m in reversed(msgs):
                r = get_role(m)
                if r in ("ai", "assistant"):
                    c = get_content(m)
                    if c:
                        history.append({"role": "assistant", "content": c})
                        break

        except Exception as e:
            print_error(f"执行失败：{e}")
            logger.exception("Agent execution failed")


if __name__ == "__main__":
    main()
