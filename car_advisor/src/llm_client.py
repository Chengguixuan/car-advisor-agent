"""大模型调用封装。

支持 OpenAI 和 Anthropic 两种后端，通过配置切换。
"""

import logging
from typing import Optional

from .config import AppConfig, LLMConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """统一的大模型调用客户端。

    支持:
        - OpenAI API（含兼容接口，如本地模型、中转代理）
        - Anthropic API（Claude 系列模型）
    """

    def __init__(self, config: AppConfig):
        self._cfg: LLMConfig = config.llm
        self._client = self._build_client()

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        *,
        history: Optional[list[dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """发送一轮对话，返回模型回复文本。

        Args:
            system_prompt:  系统提示词。
            user_message:   用户消息。
            history:        可选的历史消息列表。
            temperature:    覆盖默认温度。
            max_tokens:     覆盖最大输出 token。

        Returns:
            模型回复的文本内容。
        """
        messages = self._build_messages(system_prompt, user_message, history)

        if self._cfg.provider == "anthropic":
            return self._chat_anthropic(messages, temperature, max_tokens)
        else:
            return self._chat_openai(messages, temperature, max_tokens)

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _build_client(self):
        """根据 provider 创建对应的客户端实例。"""
        if self._cfg.provider == "anthropic":
            import anthropic

            return anthropic.Anthropic(api_key=self._cfg.api_key)
        else:
            from openai import OpenAI

            kwargs = {"api_key": self._cfg.api_key}
            if self._cfg.api_base:
                kwargs["base_url"] = self._cfg.api_base
            return OpenAI(**kwargs)

    def _build_messages(
        self,
        system_prompt: str,
        user_message: str,
        history: Optional[list[dict]],
    ) -> list[dict]:
        """构建请求消息列表。"""
        messages: list[dict] = [{"role": "system", "content": system_prompt}]

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": user_message})
        return messages

    def _chat_openai(
        self,
        messages: list[dict],
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> str:
        """OpenAI 兼容接口调用。"""
        resp = self._client.chat.completions.create(
            model=self._cfg.model,
            messages=messages,
            temperature=temperature if temperature is not None else self._cfg.temperature,
            max_tokens=max_tokens if max_tokens is not None else self._cfg.max_tokens,
        )
        return resp.choices[0].message.content or ""

    def _chat_anthropic(
        self,
        messages: list[dict],
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> str:
        """Anthropic API 调用。

        Anthropic 的 Messages API 要求 system 单独传递，messages 中
        不能包含 system 角色，且第一条必须是 user。
        """
        # 分离 system prompt
        system_content = ""
        filtered: list[dict] = []
        for m in messages:
            if m["role"] == "system":
                system_content = m["content"]
            else:
                filtered.append(m)

        resp = self._client.messages.create(
            model=self._cfg.model,
            system=system_content,
            messages=filtered,
            temperature=temperature if temperature is not None else self._cfg.temperature,
            max_tokens=max_tokens if max_tokens is not None else self._cfg.max_tokens,
        )
        # Anthropic 返回的是一个 ContentBlock 列表，取第一个 text 块
        for block in resp.content:
            if block.type == "text":
                return block.text
        return ""
