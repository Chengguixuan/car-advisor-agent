"""命令行交互入口（v1 — 单轮 JSON 模式）。

提供交互式对话界面，管理对话历史，以结构化 JSON 格式展示推荐结果。
"""

import json
import logging
from typing import Any

from .config import AppConfig, load_config
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
from .llm_client import LLMClient, LLMError, LLMResponseError
from .prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

WELCOME_MSG = f"""
{BOLD}{BLUE}🚗 买车智能体 — 你的专业购车顾问{RESET}

我可以帮你分析购车需求、推荐合适的车型。
直接描述你的想法即可开始，例如：
  "预算 15 万，想买 SUV，家用为主，不太懂车"

{DIM}输入 {BOLD}help{DIM} 查看帮助 | 输入 {BOLD}exit{DIM} 或 {BOLD}quit{DIM} 退出{RESET}
"""

HELP_MSG = f"""
{BOLD}可用指令：{RESET}
  {GREEN}help{RESET} / {GREEN}帮助{RESET}    显示此帮助信息
  {GREEN}clear{RESET} / {GREEN}清空{RESET}   清空对话历史
  {GREEN}exit{RESET} / {GREEN}quit{RESET}    退出程序

{DIM}直接输入购车需求即可获得推荐。{RESET}
"""

EXIT_MSG = f"\n{BOLD}{GREEN}感谢使用买车智能体，祝你选到心仪的爱车！🚗{RESET}\n"


# ---------------------------------------------------------------------------
# 对话管理
# ---------------------------------------------------------------------------


def _trim_history(history: list[dict[str, str]], max_turns: int) -> list[dict[str, str]]:
    """按最大轮数裁剪对话历史，始终保留 system prompt。"""
    if max_turns <= 0:
        return history
    conversation = history[1:]
    if len(conversation) > max_turns * 2:
        conversation = conversation[-(max_turns * 2):]
    return [history[0]] + conversation


def run_interactive(config: AppConfig) -> None:
    """运行 v1 交互式对话循环。"""
    if not config.llm.api_key:
        print_warning("未设置 DEEPSEEK_API_KEY 环境变量，请检查 .env 文件")
        print("  可以复制 .env.example 为 .env 并填入你的 API 密钥。\n")
        return

    client = LLMClient(config)
    history: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print(WELCOME_MSG)

    while True:
        try:
            user_input = input(f"\n{BOLD}{BLUE}你：{RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(EXIT_MSG)
            break

        if is_empty_or_whitespace(user_input):
            print(f"{DIM}请输入你的购车需求，例如 \"预算 15 万，想买 SUV\"，或输入 help 查看帮助。{RESET}")
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

        history.append({"role": "user", "content": user_input})
        print(f"\n{DIM}分析中...{RESET}\n")

        try:
            history = _trim_history(history, config.max_history_turns)
            parsed = client.chat_json(messages=list(history))
            display_recommendation(parsed)
            history.append({
                "role": "assistant",
                "content": json.dumps(parsed, ensure_ascii=False),
            })

            if config.verbose:
                char_count = sum(len(m["content"]) for m in history)
                print(f"{DIM}[调试] 历史消息数: {len(history)}, 总字符数: ~{char_count}{RESET}")

        except LLMResponseError as e:
            logger.warning("JSON parse failed, falling back to raw chat: %s", e)
            print(f"{YELLOW}⚠ JSON 解析失败，显示原始回复：{RESET}\n")
            try:
                raw = client.chat(messages=list(history))
                print(f"{DIM}{raw}{RESET}")
                print(f"\n{YELLOW}请重新表述你的需求，或输入 clear 清空对话后重试。{RESET}")
                history.append({"role": "assistant", "content": raw})
            except LLMError as e2:
                print_error(f"获取原始回复也失败：{e2}")
                logger.exception("Raw fallback also failed")
                if history and history[-1]["role"] == "user":
                    history.pop()

        except LLMError as e:
            print_error(str(e))
            logger.exception("LLM call failed")
            if history and history[-1]["role"] == "user":
                history.pop()

        except Exception as e:
            print_error(f"意外错误：{e}")
            logger.exception("Unexpected error")
            if history and history[-1]["role"] == "user":
                history.pop()


def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_interactive(load_config())


if __name__ == "__main__":
    main()
