"""CLI 展示公共模块。

提供终端颜色、消息渲染、指令判断等 main.py 和 main_langgraph.py
共享的 UI 函数，消除重复代码。
"""

from typing import Any

# ---------------------------------------------------------------------------
# ANSI 终端颜色
# ---------------------------------------------------------------------------

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
RESET = "\033[0m"

# ---------------------------------------------------------------------------
# 基础渲染
# ---------------------------------------------------------------------------


def print_separator(char: str = "─", width: int = 60) -> None:
    """打印分隔线。"""
    print(f"{DIM}{char * width}{RESET}")


def print_model_card(model: dict[str, Any], index: int = 0) -> None:
    """打印单个车型推荐卡片。

    Args:
        model: 车型字典，含 name / price_range / pros / cons / reason。
        index: 序号（0 表示不打印序号）。
    """
    name = model.get("name", "未知车型")
    price = model.get("price_range", "价格未提供")
    pros = model.get("pros", [])
    cons = model.get("cons", [])
    reason = model.get("reason", "")

    prefix = f"#{index}  " if index > 0 else ""
    print(f"\n  {BOLD}{prefix}{name}{RESET}")
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


def display_recommendation(rec: dict[str, Any]) -> None:
    """美化打印结构化推荐结果。

    Args:
        rec: 推荐字典，含 understanding / recommended_models / follow_up_question。
    """
    print_separator()

    understanding = rec.get("understanding", "")
    if understanding:
        print(f"\n{BOLD}我的理解：{RESET}{understanding}")

    models = rec.get("recommended_models", [])
    if models:
        print(f"\n{BOLD}推荐车型：{RESET}")
        for i, m in enumerate(models, 1):
            print_model_card(m, index=i)
    elif not rec.get("raw"):
        print(f"\n{DIM}（暂无推荐，等待更多信息）{RESET}")

    follow_up = rec.get("follow_up_question")
    if follow_up:
        print(f"\n{BOLD}{CYAN}{follow_up}{RESET}")

    if rec.get("parse_error"):
        print(f"\n{YELLOW}⚠ 模型输出的 JSON 格式异常，原始内容：{RESET}")
        raw = rec.get("raw", "")
        print(f"{DIM}{raw[:500]}{RESET}")

    print_separator()


def print_error(message: str) -> None:
    """打印错误信息。"""
    print(f"{RED}❌ {message}{RESET}")


def print_warning(message: str) -> None:
    """打印警告信息。"""
    print(f"{YELLOW}⚠ {message}{RESET}")


# ---------------------------------------------------------------------------
# 指令判断（大小写不敏感，/ 前缀可选）
# ---------------------------------------------------------------------------


def is_exit_command(text: str) -> bool:
    return text.lower() in ("exit", "quit", "q", "退出", "/exit", "/quit", "/q", "/退出")


def is_help_command(text: str) -> bool:
    return text.lower() in ("help", "h", "帮助", "/help", "/h", "/帮助")


def is_clear_command(text: str) -> bool:
    return text.lower() in ("clear", "清空", "/clear", "/清空")


def is_empty_or_whitespace(text: str) -> bool:
    return not text.strip()
