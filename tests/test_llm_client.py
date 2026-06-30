"""LLM 客户端单元测试。

覆盖：JSON 解析、异常翻译、客户端构建。
"""

from unittest import mock

import pytest

from car_advisor.src.llm_client import (
    LLMClient,
    LLMError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMConnectionError,
    LLMResponseError,
)
from car_advisor.src.config import AppConfig, LLMConfig


# ------------------------------------------------------------------
# JSON 解析测试
# ------------------------------------------------------------------


class TestParseJson:
    """测试 LLMClient._parse_json 的三种解析策略。"""

    def test_pure_json_object(self):
        result = LLMClient._parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_pure_json_array(self):
        result = LLMClient._parse_json('[{"name": "A"}, {"name": "B"}]')
        assert result == [{"name": "A"}, {"name": "B"}]

    def test_json_with_whitespace(self):
        result = LLMClient._parse_json('  \n  {"a": 1}  \n  ')
        assert result == {"a": 1}

    def test_markdown_code_block_with_json_tag(self):
        raw = '```json\n{"understanding": "test", "recommended_models": [], "follow_up_question": null}\n```'
        result = LLMClient._parse_json(raw)
        assert result["understanding"] == "test"
        assert result["recommended_models"] == []
        assert result["follow_up_question"] is None

    def test_markdown_code_block_without_tag(self):
        raw = '```\n{"x": 1}\n```'
        result = LLMClient._parse_json(raw)
        assert result == {"x": 1}

    def test_markdown_code_block_multiline(self):
        raw = '介绍一下：\n```json\n{\n  "name": "测试",\n  "price": 10\n}\n```\n希望对你有帮助！'
        result = LLMClient._parse_json(raw)
        assert result == {"name": "测试", "price": 10}

    def test_json_with_prefix_text(self):
        raw = '好的，以下是我的推荐：\n{"cars": [{"name": "Model A"}]}'
        result = LLMClient._parse_json(raw)
        assert result == {"cars": [{"name": "Model A"}]}

    def test_json_with_suffix_text(self):
        raw = '{"cars": [{"name": "Model A"}]}\n希望以上推荐对你有帮助！'
        result = LLMClient._parse_json(raw)
        assert result == {"cars": [{"name": "Model A"}]}

    def test_nested_json_extraction(self):
        """确保提取的是最外层 JSON，而不是内嵌的。"""
        raw = '{"data": {"inner": "value"}, "list": [1, 2, 3]}'
        result = LLMClient._parse_json(raw)
        assert result == {"data": {"inner": "value"}, "list": [1, 2, 3]}

    def test_unparseable_text_raises_error(self):
        with pytest.raises(LLMResponseError, match="无法将模型输出解析为 JSON"):
            LLMClient._parse_json("这只是一段普通文本，不包含任何 JSON 结构。")

    def test_empty_string_raises_error(self):
        with pytest.raises(LLMResponseError):
            LLMClient._parse_json("")

    def test_incomplete_json_raises_error(self):
        with pytest.raises(LLMResponseError):
            LLMClient._parse_json('{"a": 1, "b":')

    def test_realistic_response(self):
        """模拟真实的完整推荐响应。"""
        raw = """```json
{
  "understanding": "用户预算约20万，偏好SUV，家用为主，注重油耗经济性。",
  "recommended_models": [
    {
      "name": "比亚迪宋PLUS DM-i 110km 旗舰型",
      "price_range": "15-17万",
      "pros": ["插电混动油耗低", "空间表现好", "刀片电池安全"],
      "cons": ["高速亏电油耗偏高", "底盘偏软"],
      "reason": "兼顾油耗和空间，非常适合注重经济性的家庭用户。"
    },
    {
      "name": "本田CR-V 240TURBO 两驱风尚版",
      "price_range": "19-21万",
      "pros": ["口碑好保值率高", "空间利用率好", "1.5T油耗经济"],
      "cons": ["配置不如自主品牌丰富", "隔音表现一般"],
      "reason": "合资SUV标杆，油耗控制在同级别领先，保值率优秀。"
    }
  ],
  "follow_up_question": null
}
```"""
        result = LLMClient._parse_json(raw)
        assert result["understanding"] is not None
        assert len(result["recommended_models"]) == 2
        assert result["recommended_models"][0]["name"] == "比亚迪宋PLUS DM-i 110km 旗舰型"
        assert result["follow_up_question"] is None


# ------------------------------------------------------------------
# 异常翻译测试
# ------------------------------------------------------------------


class TestTranslateError:
    """测试 _translate_error 方法对各种异常消息的分类。"""

    def test_authentication_401(self):
        exc = Exception("401 Unauthorized: invalid api key")
        result = LLMClient._translate_error(exc)
        assert isinstance(result, LLMAuthenticationError)

    def test_authentication_invalid_key(self):
        exc = Exception("Invalid API key provided")
        result = LLMClient._translate_error(exc)
        assert isinstance(result, LLMAuthenticationError)

    def test_rate_limit_429(self):
        exc = Exception("429 Too Many Requests: rate limit exceeded")
        result = LLMClient._translate_error(exc)
        assert isinstance(result, LLMRateLimitError)

    def test_connection_error(self):
        exc = Exception("Connection error: timeout")
        result = LLMClient._translate_error(exc)
        assert isinstance(result, LLMConnectionError)

    def test_connection_timeout(self):
        exc = Exception("Request timed out after 30 seconds")
        result = LLMClient._translate_error(exc)
        assert isinstance(result, LLMConnectionError)

    def test_unknown_error(self):
        exc = Exception("Something completely unexpected happened")
        result = LLMClient._translate_error(exc)
        # 应为基类 LLMError，而不是子类
        assert isinstance(result, LLMError)
        assert not isinstance(result, LLMAuthenticationError)
        assert not isinstance(result, LLMRateLimitError)
        assert not isinstance(result, LLMConnectionError)


# ------------------------------------------------------------------
# 客户端构建测试
# ------------------------------------------------------------------


class TestBuildClient:
    """测试 _build_client 和 api_key 检查。"""

    def test_empty_api_key_logs_warning(self):
        app = AppConfig(llm=LLMConfig(api_key="", base_url="https://x.com/v1"))
        with mock.patch("logging.Logger.warning") as mock_warn:
            LLMClient(app)
            mock_warn.assert_called_once()
            assert "DEEPSEEK_API_KEY" in mock_warn.call_args[0][0]

    def test_client_created_with_valid_config(self):
        app = AppConfig(
            llm=LLMConfig(
                api_key="sk-valid",
                base_url="https://api.deepseek.com/v1",
                model="deepseek-chat",
            )
        )
        client = LLMClient(app)
        assert client._cfg.api_key == "sk-valid"
        assert client._cfg.base_url == "https://api.deepseek.com/v1"
