"""DeepSeek 大模型调用封装。

DeepSeek API 兼容 OpenAI SDK 格式，通过 openai 库调用。
支持文本对话和 JSON 模式输出，含自动重试机制。
"""

import json
import logging
import re
import time
from typing import Any, Callable, Optional, TypeVar

from openai import OpenAI

from .config import AppConfig, LLMConfig
from .http_client import get_http_client

# 重试配置
_MAX_RETRIES = 3          # 最大重试次数
_RETRY_BACKOFF_BASE = 1.0  # 基础退避时间（秒），指数增长：1, 2, 4

_T = TypeVar("_T")



# ---------------------------------------------------------------------------
# 异常定义
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """LLM 调用基础异常。"""


class LLMConnectionError(LLMError):
    """网络连接异常。"""


class LLMAuthenticationError(LLMError):
    """API 密钥认证失败。"""


class LLMRateLimitError(LLMError):
    """API 请求频率超限。"""


class LLMResponseError(LLMError):
    """模型返回内容异常（如空响应、格式错误）。"""


# ---------------------------------------------------------------------------
# 客户端
# ---------------------------------------------------------------------------


class LLMClient:
    """DeepSeek 大模型调用客户端。

    基于 OpenAI 兼容接口，用法示例:

        from config import load_config
        from llm_client import LLMClient

        cfg = load_config()
        client = LLMClient(cfg)

        # 普通对话
        reply = client.chat([
            {"role": "system", "content": "你是一个助手。"},
            {"role": "user",   "content": "你好"},
        ])

        # JSON 模式
        data = client.chat_json([
            {"role": "user", "content": "以JSON列出3种水果及其价格"},
        ])
    """

    def __init__(self, config: AppConfig):
        self._cfg: LLMConfig = config.llm
        self._client = self._build_client()

    # ------------------------------------------------------------------
    # 公共方法
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> str:
        """发送对话请求，返回模型文本回复。

        Args:
            messages:    消息列表，格式为 [{"role": "...", "content": "..."}]
            temperature: 生成温度（None 则使用配置默认值）
            max_tokens:  最大输出 token（None 则使用配置默认值）
            model:       模型名称（None 则使用配置默认值）

        Returns:
            模型回复的文本内容。

        Raises:
            LLMConnectionError:      网络连接失败
            LLMAuthenticationError:  API 密钥无效
            LLMRateLimitError:       请求频率超限
            LLMResponseError:        返回内容为空
            LLMError:                其他调用异常
        """
        logger.info("chat: sending %d messages, model=%s", len(messages), model or self._cfg.model)
        logger.debug("messages: %s", json.dumps(messages, ensure_ascii=False)[:500])

        resp = self._call_with_retry(
            lambda: self._client.chat.completions.create(
                model=model or self._cfg.model,
                messages=messages,
                temperature=temperature if temperature is not None else self._cfg.temperature,
                max_tokens=max_tokens if max_tokens is not None else self._cfg.max_tokens,
            ),
            context="chat",
        )

        # 提取文本内容
        try:
            content = resp.choices[0].message.content
        except (IndexError, AttributeError):
            raise LLMResponseError("模型返回内容为空")

        if not content:
            raise LLMResponseError("模型返回内容为空字符串")

        # 记录 token 用量
        usage = getattr(resp, "usage", None)
        if usage:
            logger.info(
                "chat: done — prompt=%s, completion=%s, total=%s",
                getattr(usage, "prompt_tokens", "?"),
                getattr(usage, "completion_tokens", "?"),
                getattr(usage, "total_tokens", "?"),
            )

        return content

    # JSON 指令标记，用于判断是否已追加
    _JSON_INSTRUCTION = "请严格以合法的 JSON 格式回复，不要包含 markdown 代码块标记。"

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> dict[str, Any]:
        """发送对话请求，返回解析后的 JSON 对象。

        DeepSeek API 通过 response_format={"type": "json_object"} 启用 JSON
        模式，同时在 system prompt 末尾追加 JSON 指令以确保模型输出合法 JSON。
        如果 system prompt 已包含 JSON 指令则不会重复追加，避免多轮对话时
        消息膨胀。

        Args:
            messages:    消息列表
            temperature: 生成温度（None 则使用配置默认值，JSON 模式建议较低温度）
            max_tokens:  最大输出 token
            model:       模型名称

        Returns:
            解析后的 dict / list。

        Raises:
            LLMResponseError: 返回内容不是合法 JSON
            其他同 chat()。
        """
        augmented: list[dict[str, str]] = []
        instruction_appended = False

        for m in messages:
            if m["role"] == "system":
                content = m["content"]
                if self._JSON_INSTRUCTION not in content:
                    content = content + "\n\n" + self._JSON_INSTRUCTION
                    instruction_appended = True
                augmented.append({"role": "system", "content": content})
            else:
                augmented.append(dict(m))

        # 确保至少有一条 system 消息
        if not any(m["role"] == "system" for m in augmented):
            augmented.insert(0, {
                "role": "system",
                "content": self._JSON_INSTRUCTION,
            })
            instruction_appended = True

        logger.info(
            "chat_json: sending %d messages, json_mode=True%s",
            len(augmented),
            " (instruction appended)" if instruction_appended else " (instruction already present)",
        )

        resp = self._call_with_retry(
            lambda: self._client.chat.completions.create(
                model=model or self._cfg.model,
                messages=augmented,
                temperature=temperature if temperature is not None else min(self._cfg.temperature, 0.3),
                max_tokens=max_tokens if max_tokens is not None else self._cfg.max_tokens,
                response_format={"type": "json_object"},
            ),
            context="chat_json",
        )

        try:
            raw = resp.choices[0].message.content or ""
        except (IndexError, AttributeError):
            raise LLMResponseError("JSON 模式下模型返回内容为空")

        # 尝试解析 JSON，兼容 markdown 代码块包裹的情况
        return self._parse_json(raw)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _call_with_retry(self, fn: Callable[[], _T], *, context: str = "") -> _T:
        """带指数退避重试的 API 调用。

        仅在网络连接错误和频率限制（429）时重试，认证错误和响应错误
        直接抛出。

        Args:
            fn:      实际 API 调用的无参函数。
            context: 调用上下文描述，用于日志。

        Returns:
            API 调用的返回值。

        Raises:
            LLMAuthenticationError: 认证失败（不重试）
            LLMConnectionError:     网络错误（重试耗尽）
            LLMRateLimitError:      频率限制（重试耗尽）
            LLMError:               其他错误
        """
        last_exc: Optional[LLMError] = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return fn()
            except Exception as exc:
                translated = self._translate_error(exc)
                last_exc = translated

                # 认证错误不重试
                if isinstance(translated, LLMAuthenticationError):
                    raise translated

                # 连接和限流错误可以重试
                retryable = isinstance(translated, (LLMConnectionError, LLMRateLimitError))
                if not retryable:
                    raise translated

                if attempt < _MAX_RETRIES:
                    delay = _RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
                    logger.warning(
                        "%s: attempt %d/%d failed (%s), retrying in %.1fs...",
                        context or "call", attempt, _MAX_RETRIES, translated, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "%s: all %d attempts failed, last error: %s",
                        context or "call", _MAX_RETRIES, translated,
                    )

        # 所有重试耗尽
        raise last_exc  # type: ignore[misc]

    def _build_client(self) -> OpenAI:
        """创建 OpenAI 兼容客户端，指向 DeepSeek API。"""
        logger.info("initializing DeepSeek client: base_url=%s, model=%s",
                     self._cfg.base_url, self._cfg.model)

        if not self._cfg.api_key:
            logger.warning("DEEPSEEK_API_KEY is empty — API calls will fail")

        return OpenAI(
            api_key=self._cfg.api_key,
            base_url=self._cfg.base_url,
            http_client=get_http_client(),
        )

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        """从模型原始输出中提取合法的 JSON 对象。

        兼容以下情况：
        - 纯 JSON 字符串
        - 被 ```json ... ``` 包裹的内容
        - 字符串首尾包含少量非 JSON 字符

        Raises:
            LLMResponseError: 无法解析为合法 JSON。
        """
        raw = raw.strip()

        # 1) 先尝试直接解析
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # 2) 尝试提取 ```json ... ``` 代码块
        code_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if code_block:
            try:
                return json.loads(code_block.group(1).strip())
            except json.JSONDecodeError:
                pass

        # 3) 尝试找到最外层 JSON 边界
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = raw.find(start_char)
            end = raw.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(raw[start:end + 1])
                except json.JSONDecodeError:
                    continue

        logger.warning("failed to parse JSON from: %.200s", raw)
        raise LLMResponseError(f"无法将模型输出解析为 JSON，原始内容前 200 字符: {raw[:200]}")

    @staticmethod
    def _translate_error(exc: Exception) -> LLMError:
        """将 OpenAI SDK 异常转换为项目自定义异常。"""
        msg = str(exc)

        # 认证失败 (HTTP 401)
        if "401" in msg or "authentication" in msg.lower() or "invalid api key" in msg.lower():
            logger.error("DeepSeek authentication failed — check DEEPSEEK_API_KEY")
            return LLMAuthenticationError(f"API 密钥无效或已过期: {msg}")

        # 频率限制 (HTTP 429)
        if "429" in msg or "rate limit" in msg.lower():
            logger.error("DeepSeek rate limited")
            return LLMRateLimitError(f"请求频率超限，请稍后重试: {msg}")

        # 连接错误
        if "connection" in msg.lower() or "timeout" in msg.lower() or "timed out" in msg.lower() or "connection error" in msg.lower():
            logger.error("DeepSeek connection failed: %s", msg)
            return LLMConnectionError(f"网络连接失败: {msg}")

        # 其他错误
        logger.error("DeepSeek API error: %s", msg)
        return LLMError(f"DeepSeek API 调用失败: {msg}")
