"""命令行交互入口。

提供交互式对话界面，管理对话历史，响应用户指令。
"""

import logging
import sys
from pathlib import Path

# 确保 src 目录在 sys.path 中，方便直接运行此文件
_src_dir = Path(__file__).resolve().parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from config import AppConfig, load_config
from llm_client import LLMClient
from prompts import SYSTEM_PROMPT, SHORTCUT_PROMPTS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 终端颜色（Windows 10+ 支持 ANSI）
# ---------------------------------------------------------------------------

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RESET = "\033[0m"

WELCOME_MSG = f"""
{BOLD}{BLUE}🚗 买车智能体 — 你的专业购车顾问{RESET}

我可以帮你：
  • 根据需求和预算推荐车型
  • 对比多款车型的优缺点
  • 分析燃油车 vs 新能源车的选择
  • 梳理购车流程和贷款方案

{DIM}输入 {BOLD}/帮助{DIM} 查看快捷指令 | 输入 {BOLD}/退出{DIM} 结束对话{RESET}
"""

HELP_MSG = f"""
{BOLD}快捷指令：{RESET}
  {GREEN}/对比{RESET}    对比多款车型
  {GREEN}/新能源{RESET}  分析燃油车 vs 新能源
  {GREEN}/流程{RESET}    了解购车流程
  {GREEN}/贷款{RESET}    分析贷款方案
  {GREEN}/帮助{RESET}    显示此帮助信息
  {GREEN}/退出{RESET}    结束对话
  {GREEN}/清空{RESET}    清空对话历史

{DIM}也可以直接用自然语言描述你的需求。{RESET}
"""

EXIT_MSG = f"\n{BOLD}{GREEN}感谢使用买车智能体，祝你选到心仪的爱车！🚗{RESET}\n"


def build_history_messages(history: list[dict]) -> list[dict]:
    """将内部历史格式转为 LLM API 所需的消息列表。

    Args:
        history: 对话历史，每项为 {"role": "user"/"assistant", "content": "..."}

    Returns:
        API 格式的消息列表（不含 system）。
    """
    return [{"role": h["role"], "content": h["content"]} for h in history]


def run_interactive(config: AppConfig) -> None:
    """运行交互式对话循环。

    Args:
        config: 应用配置实例。
    """
    # 初始化 LLM 客户端
    if not config.llm.api_key:
        print(f"{YELLOW}⚠ 未设置 LLM_API_KEY 环境变量，请检查 .env 文件{RESET}")
        print("  可以复制 .env.example 为 .env 并填入你的 API 密钥。\n")
        return

    client = LLMClient(config)
    history: list[dict] = []

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

        # --- 处理内置指令 ---
        if user_input in ("/退出", "/quit", "/exit", "/q"):
            print(EXIT_MSG)
            break

        if user_input in ("/帮助", "/help", "/h"):
            print(HELP_MSG)
            continue

        if user_input in ("/清空", "/clear"):
            history.clear()
            print(f"{DIM}对话历史已清空。{RESET}")
            continue

        # --- 处理快捷指令 -> 引导用户补充信息 ---
        if user_input in ("/对比",):
            print(f"\n{BOLD}请输入要对比的车型（每行一个），格式如下：{RESET}")
            print(f"{DIM}  车型列表：{RESET}")
            print(f"{DIM}  1. 丰田凯美瑞 2.0G 豪华版{RESET}")
            print(f"{DIM}  2. 本田雅阁 260TURBO 豪华版{RESET}")
            print(f"{DIM}  预算范围：20万{RESET}")
            print(f"{DIM}  主要用途：日常通勤{RESET}")
            user_input = input(f"\n{BOLD}{BLUE}你：{RESET}").strip()
            if not user_input:
                continue
            user_input = (
                "请帮我对比以下车型，从价格、油耗、空间、动力、配置、"
                f"保值率等方面分析：\n\n{user_input}"
            )

        elif user_input in ("/新能源",):
            print(f"\n{BOLD}请告诉我以下信息：{RESET}")
            info = {}
            info["annual_mileage"] = input(f"  年行驶里程约（公里）：").strip()
            info["has_charger"] = input(f"  是否有固定车位/充电桩（是/否）：").strip()
            info["usage_scenario"] = input(f"  主要使用场景：").strip()
            info["budget"] = input(f"  预算（万元）：").strip()
            info["hold_years"] = input(f"  计划持有年限：").strip()
            user_input = (
                f"我在燃油车和新能源车之间犹豫。请根据以下情况帮我分析：\n"
                f"- 年行驶里程约 {info['annual_mileage']} 公里\n"
                f"- 是否有固定车位/充电桩：{info['has_charger']}\n"
                f"- 主要使用场景：{info['usage_scenario']}\n"
                f"- 预算：{info['budget']} 万元\n"
                f"- 计划持有年限：{info['hold_years']} 年"
            )

        elif user_input in ("/流程",):
            situation = input(f"\n{BOLD}请简单描述你的情况（可选，按回车跳过）：{RESET}").strip()
            user_input = "我准备近期买车，请帮我梳理一下完整的购车流程，包括选车、试驾、谈价、贷款、保险、提车、上牌等环节的注意事项。"
            if situation:
                user_input += f" 我的情况是：{situation}"

        elif user_input in ("/贷款",):
            print(f"\n{BOLD}请告诉我以下信息：{RESET}")
            car_price = input(f"  目标车型价格（万元）：").strip()
            down_payment = input(f"  首付预算（万元）：").strip()
            monthly_income = input(f"  月收入（元）：").strip()
            monthly_budget = input(f"  每月可用于养车的金额（元）：").strip()
            user_input = (
                f"请帮我分析一下购车贷款方案：\n"
                f"- 目标车型价格：{car_price} 万元\n"
                f"- 首付预算：{down_payment} 万元\n"
                f"- 月收入：{monthly_income} 元\n"
                f"- 每月可用于养车的金额：{monthly_budget} 元\n"
                f"\n请帮我算一下合理的贷款方案，并提醒隐性成本。"
            )

        # --- 调用 LLM ---
        print(f"\n{DIM}思考中...{RESET}\n")
        try:
            # 限制历史轮数以控制 token 消耗
            max_turns = config.max_history_turns
            recent_history = (
                history[-max_turns * 2 :] if max_turns > 0 else []
            )

            # 构建 messages 列表（OpenAI 格式）
            messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
            if recent_history:
                messages.extend(build_history_messages(recent_history))
            messages.append({"role": "user", "content": user_input})

            response = client.chat(messages=messages)

            print(f"{BOLD}{GREEN}顾问：{RESET}{response}\n")

            # 保存到历史
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})

            if config.verbose:
                print(
                    f"{DIM}[调试] 当前历史轮数: {len(history) // 2}, "
                    f"token 估算: ~{sum(len(m['content']) for m in history) // 2}{RESET}"
                )

        except Exception as e:
            print(f"{YELLOW}❌ 调用模型失败：{e}{RESET}")
            logger.exception("LLM call failed")
            # 不要让一次失败中断整个会话
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
