"""环境变量和配置管理。

从 .env 文件和环境变量中加载 DeepSeek 相关配置。
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# 加载项目根目录下的 .env 文件
# config.py → src/ → car_advisor/ → 项目根目录
_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_ENV_FILE)


@dataclass
class LLMConfig:
    """DeepSeek 大模型连接配置（兼容 OpenAI 格式）。"""

    api_key: str = ""
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 4096


@dataclass
class AppConfig:
    """应用全局配置。"""

    llm: LLMConfig = field(default_factory=LLMConfig)
    max_history_turns: int = 20  # 对话历史最大轮数
    verbose: bool = False        # 是否输出调试信息


def load_config() -> AppConfig:
    """从环境变量加载配置。

    Returns:
        AppConfig: 应用配置实例。

    环境变量说明:
        DEEPSEEK_API_KEY:     API 密钥（必填）
        DEEPSEEK_BASE_URL:    API 地址（默认 https://api.deepseek.com/v1）
        DEEPSEEK_MODEL:       模型名称（默认 deepseek-chat）
        DEEPSEEK_TEMPERATURE: 生成温度（默认 0.7）
        DEEPSEEK_MAX_TOKENS:  最大输出 token（默认 4096）
        MAX_HISTORY:          对话历史最大轮数（默认 20）
        VERBOSE:              调试模式（默认 false）
    """
    try:
        temperature = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.7"))
    except ValueError:
        temperature = 0.7

    try:
        max_tokens = int(os.getenv("DEEPSEEK_MAX_TOKENS", "4096"))
    except ValueError:
        max_tokens = 4096

    llm = LLMConfig(
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        base_url=os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1",
        model=os.getenv("DEEPSEEK_MODEL") or "deepseek-chat",
        temperature=temperature,
        max_tokens=max_tokens,
    )

    try:
        max_history_turns = int(os.getenv("MAX_HISTORY", "20"))
    except ValueError:
        max_history_turns = 20

    return AppConfig(
        llm=llm,
        max_history_turns=max_history_turns,
        verbose=os.getenv("VERBOSE", "false").lower() in ("1", "true", "yes"),
    )
