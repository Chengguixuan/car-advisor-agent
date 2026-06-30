"""命令行交互入口。

提供交互式对话界面，管理对话历史，以结构化 JSON 格式展示推荐结果。
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

# 确保 src 目录在 sys.path 中，方便直接运行此文件
_src_dir = Path(__file__).resolve().parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from config import AppConfig, load_config
from llm_client import LLMClient, LLMError
from prompts import SYSTEM_PROMPT

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

我会帮你理清需求、推荐合适的车型，并对你关心的问题给出建议。
告诉你的想法，我们慢慢聊——

{DIM}输入 {BOLD}/帮助{DIM} 查看快捷指令 | 输入 {BOLD}/退出{DIM} 结束对话{RESET}
"""

HELP_MSG = f"""
{BOLD}快捷指令：{RESET}
  {GREEN}/清空{RESET}    清空对话历史
  {GREEN}/帮助{RESET}    显示此帮助信息
  {GREEN}/退出{RESET}    结束对话

{DIM}直接输入你的购车需求即可，例如：{RESET}
  "预算 15 万，想买 SUV，家用为主，不太懂车"
"""

EXIT_MSG = f"\n{BOLD}{GREEN}感谢使用买车智能体，祝你选到心仪的爱车！🚗{RESET}\n"


# ---------------------------------------------------------------------------
# 结构化响应展示
# ---------------------------------------------------------------------------


def _print_divider(char: str = "─", width: int = 60) -> None:
    print(f"{DIM}{char * width}{RESET}")


def _print_model_card(model: dict, index: int) -> None:
    """打印单个车型推荐卡片。"""
    name = model.get("name", "未知车型")
    price = model.get("price_range", "价格未提供")
    pros = model.get("pros", [])
    cons = model.get("cons", [])
    reason = model.get("reason", "")

    print(f"\n  {BOLD}{CYAN}#{index}  {name}{RESET}")
    print(f"  {DIM}💰 落地参考价：{RESET}{price}")

    if pros:
        print(f"  {GREEN}✅ 优点：{RESET}")
        for p in pros:
            print(f"     • {p}")

    if cons:
        print(f"  {YELLOW}⚠️  缺点：{RESET}")
        for c in cons:
            print(f"     • {c}")

    if reason:
        print(f"  {BLUE}💡 推荐理由：{RESET}{reason}")


def display_response(parsed: Optional[dict[str, Any]]) -> None:
    """将模型的结构化 JSON 回复渲染到终端。

    Args:
        parsed: 模型返回的解析后 JSON 对象。
    """
    if parsed is None:
        print(f"{RED}未能获取有效的回复，请重试。{RESET}")
        return

    _print_divider()

    # 1) 需求理解
    understanding = parsed.get("understanding", "")
    if understanding:
        print(f"\n{BOLD}📋 需求理解：{RESET}{DIM}{understanding}{RESET}")

    # 2) 推荐车型
    models = parsed.get("recommended_models", [])
    if models:
        print(f"\n{BOLD}🚘 推荐车型：{RESET}")
        for i, model in enumerate(models, 1):
            _print_model_card(model, i)
    else:
        print(f"\n{DIM}暂无推荐，等待更多信息...{RESET}")

    # 3) 追问
    follow_up = parsed.get("follow_up_question")
    if follow_up:
        print(f"\n{BOLD}{CYAN}🤔 {follow_up}{RESET}")

    _print_divider()


# ---------------------------------------------------------------------------
# 对话管理
# ---------------------------------------------------------------------------


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
    # history 存储 OpenAI 格式的消息
    history: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    print(WELCOME_MSG)

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    while True:
        try:
            user_input = input(f"\n{BOLD}{BLUE}你：{RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(EXIT_MSG)
            break

        if not user_input:
            continue

        # --- 内置指令 ---
        if user_input in ("/退出", "/quit", "/exit", "/q"):
            print(EXIT_MSG)
            break

        if user_input in ("/帮助", "/help", "/h"):
            print(HELP_MSG)
            continue

        if user_input in ("/清空", "/clear"):
            # 保留 system prompt，只清空对话
            history = [{"role": "system", "content": SYSTEM_PROMPT}]
            print(f"{DIM}对话历史已清空。{RESET}")
            continue

        # --- 添加用户消息 ---
        history.append({"role": "user", "content": user_input})

        # --- 限制历史长度 ---
        max_turns = config.max_history_turns
        if max_turns > 0:
            # system prompt 不算在轮数内
            conversation = history[1:]
            if len(conversation) > max_turns * 2:
                history = [history[0]] + conversation[-(max_turns * 2):]

        # --- 调用 LLM（JSON 模式）---
        print(f"\n{DIM}分析中...{RESET}\n")
        try:
            parsed = client.chat_json(messages=list(history))

            # 显示结构化响应
            display_response(parsed)

            # 将助手回复保存到历史（以 JSON 字符串形式存储，便于模型理解上下文）
            history.append({
                "role": "assistant",
                "content": json.dumps(parsed, ensure_ascii=False),
            })

            if config.verbose:
                token_count = sum(len(m["content"]) for m in history)
                print(
                    f"{DIM}[调试] 历史消息数: {len(history)}, "
                    f"总字符数: ~{token_count}{RESET}"
                )

        except LLMError as e:
            print(f"{RED}❌ {e}{RESET}")
            logger.exception("LLM call failed")
            # 移除刚才添加的用户消息，避免历史中残留无对应回复的记录
            if history and history[-1]["role"] == "user":
                history.pop()
            continue
        except Exception as e:
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
