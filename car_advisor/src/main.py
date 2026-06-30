"""命令行交互入口。

提供交互式对话界面，管理对话历史，以结构化 JSON 格式展示推荐结果。
"""

import json
import logging
from typing import Any

from .config import AppConfig, load_config
from .llm_client import LLMClient, LLMError, LLMResponseError
from .prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 终端颜色（Windows 10+ 支持 ANSI）
# ---------------------------------------------------------------------------

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
RESET = "\033[0m"

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
# 结构化响应展示
# ---------------------------------------------------------------------------


def _print_separator(char: str = "─", width: int = 60) -> None:
    print(f"{DIM}{char * width}{RESET}")


def _print_model_card(model: dict, index: int) -> None:
    """打印单个车型推荐卡片。

    Args:
        model: 车型信息字典，包含 name / price_range / pros / cons / reason。
        index: 车型序号。
    """
    name = model.get("name", "未知车型")
    price = model.get("price_range", "价格未提供")
    pros = model.get("pros", [])
    cons = model.get("cons", [])
    reason = model.get("reason", "")

    print(f"\n  {BOLD}#{index}  {name}{RESET}")
    print(f"  {GREEN}价格：{RESET}{price}")

    if pros:
        print(f"  {GREEN}优点：{RESET}")
        for p in pros:
            print(f"    • {p}")

    if cons:
        print(f"  {YELLOW}缺点：{RESET}")
        for c in cons:
            print(f"    • {c}")

    if reason:
        print(f"  {BLUE}推荐理由：{RESET}{reason}")


def display_response(parsed: dict[str, Any]) -> None:
    """将模型的结构化 JSON 回复渲染到终端。

    Args:
        parsed: 模型返回的解析后 JSON 对象。
    """
    _print_separator()

    # 1) 我的理解
    understanding = parsed.get("understanding", "")
    if understanding:
        print(f"\n{BOLD}我的理解：{RESET}{understanding}")

    # 2) 推荐车型
    models = parsed.get("recommended_models", [])
    if models:
        print(f"\n{BOLD}推荐车型：{RESET}")
        for i, model in enumerate(models, 1):
            _print_model_card(model, i)
    else:
        print(f"\n{DIM}（暂无推荐，等待更多信息）{RESET}")

    # 3) 追问
    follow_up = parsed.get("follow_up_question")
    if follow_up:
        print(f"\n{BOLD}{CYAN}{follow_up}{RESET}")

    _print_separator()


# ---------------------------------------------------------------------------
# 对话管理
# ---------------------------------------------------------------------------


def _is_exit_command(text: str) -> bool:
    """判断用户输入是否为退出指令。"""
    return text.lower() in ("exit", "quit", "q", "退出", "/exit", "/quit", "/q", "/退出")


def _is_help_command(text: str) -> bool:
    """判断用户输入是否为帮助指令。"""
    return text.lower() in ("help", "h", "帮助", "/help", "/h", "/帮助")


def _is_clear_command(text: str) -> bool:
    """判断用户输入是否为清空历史指令。"""
    return text.lower() in ("clear", "清空", "/clear", "/清空")


def _trim_history(history: list[dict[str, str]], max_turns: int) -> list[dict[str, str]]:
    """按最大轮数裁剪对话历史，始终保留 system prompt。

    Args:
        history:   完整历史（第一条为 system prompt）。
        max_turns: 最大保留轮数，<= 0 表示不限制。

    Returns:
        裁剪后的历史列表。
    """
    if max_turns <= 0:
        return history
    conversation = history[1:]  # 去掉 system prompt
    if len(conversation) > max_turns * 2:
        conversation = conversation[-(max_turns * 2):]
    return [history[0]] + conversation


def run_interactive(config: AppConfig) -> None:
    """运行交互式对话循环。

    Args:
        config: 应用配置实例。
    """
    # 检查 API Key
    if not config.llm.api_key:
        print(f"{YELLOW}⚠ 未设置 DEEPSEEK_API_KEY 环境变量，请检查 .env 文件{RESET}")
        print("  可以复制 .env.example 为 .env 并填入你的 API 密钥。\n")
        return

    client = LLMClient(config)
    history: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print(WELCOME_MSG)

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    while True:
        # 读取用户输入
        try:
            user_input = input(f"\n{BOLD}{BLUE}你：{RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(EXIT_MSG)
            break

        if not user_input:
            print(f"{DIM}请输入你的购车需求，例如 \"预算 15 万，想买 SUV\"，或输入 help 查看帮助。{RESET}")
            continue

        # --- 内置指令 ---
        if _is_exit_command(user_input):
            print(EXIT_MSG)
            break

        if _is_help_command(user_input):
            print(HELP_MSG)
            continue

        if _is_clear_command(user_input):
            history = [{"role": "system", "content": SYSTEM_PROMPT}]
            print(f"{DIM}对话历史已清空。{RESET}")
            continue

        # --- 添加用户消息并发送请求 ---
        history.append({"role": "user", "content": user_input})

        print(f"\n{DIM}分析中...{RESET}\n")

        try:
            history = _trim_history(history, config.max_history_turns)

            # 使用 JSON 模式调用模型
            parsed = client.chat_json(messages=list(history))

            # 结构化展示
            display_response(parsed)

            # 将助手回复以 JSON 字符串存入历史，保持上下文连贯
            history.append({
                "role": "assistant",
                "content": json.dumps(parsed, ensure_ascii=False),
            })

            if config.verbose:
                char_count = sum(len(m["content"]) for m in history)
                print(
                    f"{DIM}[调试] 历史消息数: {len(history)}, "
                    f"总字符数: ~{char_count}{RESET}"
                )

        except LLMResponseError as e:
            # JSON 解析失败 — 用普通 chat 获取原始回复
            logger.warning("JSON parse failed, falling back to raw chat: %s", e)
            print(f"{YELLOW}⚠ JSON 解析失败，显示原始回复：{RESET}\n")
            try:
                raw = client.chat(messages=list(history))
                print(f"{DIM}{raw}{RESET}")
                print(f"\n{YELLOW}请重新表述你的需求，或输入 clear 清空对话后重试。{RESET}")
                # 将原始回复也存入历史，避免上下文断裂
                history.append({"role": "assistant", "content": raw})
            except LLMError as e2:
                print(f"{RED}❌ 获取原始回复也失败：{e2}{RESET}")
                logger.exception("Raw fallback also failed")
                if history and history[-1]["role"] == "user":
                    history.pop()

        except LLMError as e:
            # 其他 LLM 异常（网络、认证、限流等）
            print(f"{RED}❌ {e}{RESET}")
            logger.exception("LLM call failed")
            if history and history[-1]["role"] == "user":
                history.pop()
            continue

        except Exception as e:
            # 未预期的异常
            print(f"{RED}❌ 意外错误：{e}{RESET}")
            logger.exception("Unexpected error")
            if history and history[-1]["role"] == "user":
                history.pop()
            continue


def main() -> None:
    """程序入口。"""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config()
    run_interactive(config)


if __name__ == "__main__":
    main()
