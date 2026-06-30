"""配置管理单元测试。

覆盖：环境变量加载、默认值、类型转换边界情况。
"""

import os

import pytest

from car_advisor.src.config import LLMConfig, AppConfig, load_config


class TestDefaults:
    """默认值测试 — 确保无环境变量时各字段为预期值。"""

    def setup_method(self):
        for k in ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL",
                   "DEEPSEEK_TEMPERATURE", "DEEPSEEK_MAX_TOKENS",
                   "MAX_HISTORY", "VERBOSE"):
            os.environ.pop(k, None)

    def test_default_api_key(self):
        assert load_config().llm.api_key == ""

    def test_default_base_url(self):
        assert load_config().llm.base_url == "https://api.deepseek.com/v1"

    def test_default_model(self):
        assert load_config().llm.model == "deepseek-chat"

    def test_default_temperature(self):
        assert load_config().llm.temperature == 0.7

    def test_default_max_tokens(self):
        assert load_config().llm.max_tokens == 4096

    def test_default_max_history(self):
        assert load_config().max_history_turns == 20

    def test_default_verbose(self):
        assert load_config().verbose is False


class TestEnvVars:
    """环境变量加载测试。"""

    def setup_method(self):
        for k in ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL",
                   "DEEPSEEK_TEMPERATURE", "DEEPSEEK_MAX_TOKENS",
                   "MAX_HISTORY", "VERBOSE"):
            os.environ.pop(k, None)

    def test_custom_api_key(self):
        os.environ["DEEPSEEK_API_KEY"] = "sk-test-123"
        assert load_config().llm.api_key == "sk-test-123"

    def test_custom_base_url(self):
        os.environ["DEEPSEEK_BASE_URL"] = "https://custom.api.com/v1"
        assert load_config().llm.base_url == "https://custom.api.com/v1"

    def test_custom_model(self):
        os.environ["DEEPSEEK_MODEL"] = "deepseek-reasoner"
        assert load_config().llm.model == "deepseek-reasoner"

    def test_custom_temperature(self):
        os.environ["DEEPSEEK_TEMPERATURE"] = "0.3"
        assert load_config().llm.temperature == 0.3

    def test_custom_max_tokens(self):
        os.environ["DEEPSEEK_MAX_TOKENS"] = "2048"
        assert load_config().llm.max_tokens == 2048

    def test_custom_max_history(self):
        os.environ["MAX_HISTORY"] = "10"
        assert load_config().max_history_turns == 10

    def test_verbose_true(self):
        os.environ["VERBOSE"] = "true"
        assert load_config().verbose is True

    def test_verbose_yes(self):
        os.environ["VERBOSE"] = "yes"
        assert load_config().verbose is True

    def test_verbose_1(self):
        os.environ["VERBOSE"] = "1"
        assert load_config().verbose is True


class TestEdgeCases:
    """边界情况测试。"""

    def setup_method(self):
        for k in ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL",
                   "DEEPSEEK_TEMPERATURE", "DEEPSEEK_MAX_TOKENS",
                   "MAX_HISTORY", "VERBOSE"):
            os.environ.pop(k, None)

    def test_invalid_temperature_falls_back_to_default(self):
        """非法温度值应优雅降级为默认值，不再抛出 ValueError。"""
        os.environ["DEEPSEEK_TEMPERATURE"] = "hot"
        cfg = load_config()
        assert cfg.llm.temperature == 0.7

    def test_invalid_max_tokens_falls_back_to_default(self):
        os.environ["DEEPSEEK_MAX_TOKENS"] = "many"
        cfg = load_config()
        assert cfg.llm.max_tokens == 4096

    def test_invalid_max_history_falls_back_to_default(self):
        os.environ["MAX_HISTORY"] = "all"
        cfg = load_config()
        assert cfg.max_history_turns == 20

    def test_empty_env_var_uses_default(self):
        os.environ["DEEPSEEK_MODEL"] = ""
        assert load_config().llm.model == "deepseek-chat"

    def test_appconfig_default_llm_config(self):
        app = AppConfig()
        assert isinstance(app.llm, LLMConfig)
        assert app.llm.api_key == ""
