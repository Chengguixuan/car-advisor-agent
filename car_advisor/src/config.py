"""环境变量和配置管理。

从 .env 文件和环境变量中加载项目配置。
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# 加载项目根目录下的 .env 文件
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_FILE)


@dataclass
class LLMConfig:
    """大模型连接配置。"""

    provider: str = "openai"  # "openai" 或 "anthropic"
    api_key: str = ""
    api_base: Optional[str] = None  # 自定义 API 地址（代理/兼容接口）
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096


@dataclass
class AppConfig:
    """应用全局配置。"""

    llm: LLMConfig = field(default_factory=LLMConfig)
    max_history_turns: int = 20  # 对话历史最大轮数
    verbose: bool = False  # 是否输出调试信息


def load_config() -> AppConfig:
    """从环境变量加载配置。

    Returns:
        AppConfig: 应用配置实例。

    环境变量说明:
        LLM_PROVIDER:  模型提供商，openai 或 anthropic（默认 openai）
        LLM_API_KEY:    API 密钥
        LLM_API_BASE:   自定义 API 地址（可选）
        LLM_MODEL:      模型名称（默认 gpt-4o）
        LLM_TEMPERATURE: 生成温度（默认 0.7）
        LLM_MAX_TOKENS:  最大输出 token（默认 4096）
        MAX_HISTORY:     对话历史最大轮数（默认 20）
        VERBOSE:         调试模式（默认 false）
    """
    llm = LLMConfig(
        provider=os.getenv("LLM_PROVIDER", "openai").lower(),
        api_key=os.getenv("LLM_API_KEY", ""),
        api_base=os.getenv("LLM_API_BASE") or None,
        model=os.getenv("LLM_MODEL", "gpt-4o"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
    )

    return AppConfig(
        llm=llm,
        max_history_turns=int(os.getenv("MAX_HISTORY", "20")),
        verbose=os.getenv("VERBOSE", "false").lower() in ("1", "true", "yes"),
    )
