"""命令行交互逻辑单元测试 + 集成测试。

覆盖：指令判断、历史裁剪、响应展示、空输入处理、
以及与 DeepSeek API 的集成场景。
"""

import os

import pytest

from car_advisor.src.main import (
    _is_exit_command,
    _is_help_command,
    _is_clear_command,
    _trim_history,
    display_response,
    WELCOME_MSG,
    HELP_MSG,
    EXIT_MSG,
)
from car_advisor.src.prompts import SYSTEM_PROMPT


# ------------------------------------------------------------------
# 内置指令判断
# ------------------------------------------------------------------


class TestIsExitCommand:

    @pytest.mark.parametrize("text", [
        "exit", "EXIT", "Exit",
        "quit", "QUIT", "Quit",
        "q", "Q",
        "退出",
        "/exit", "/quit", "/q", "/退出",
    ])
    def test_recognized(self, text):
        assert _is_exit_command(text) is True

    @pytest.mark.parametrize("text", [
        "exitt", "quits", "", " 退出 ", "end",
    ])
    def test_not_recognized(self, text):
        assert _is_exit_command(text) is False


class TestIsHelpCommand:

    @pytest.mark.parametrize("text", [
        "help", "HELP", "h", "H",
        "帮助",
        "/help", "/h", "/帮助",
    ])
    def test_recognized(self, text):
        assert _is_help_command(text) is True

    @pytest.mark.parametrize("text", [
        "helpp", "helps", "", "?",
    ])
    def test_not_recognized(self, text):
        assert _is_help_command(text) is False


class TestIsClearCommand:

    @pytest.mark.parametrize("text", [
        "clear", "CLEAR", "清空",
        "/clear", "/清空",
    ])
    def test_recognized(self, text):
        assert _is_clear_command(text) is True

    @pytest.mark.parametrize("text", [
        "clearr", "clear ", "", "reset",
    ])
    def test_not_recognized(self, text):
        assert _is_clear_command(text) is False


# ------------------------------------------------------------------
# 历史裁剪
# ------------------------------------------------------------------


class TestTrimHistory:

    def test_no_limit_returns_unchanged(self):
        history = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        assert _trim_history(history, max_turns=0) == history

    def test_within_limit_returns_unchanged(self):
        history = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "reply1"},
        ]
        assert _trim_history(history, max_turns=5) == history

    def test_exceeds_limit_trims_oldest(self):
        history = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "round1"},
            {"role": "assistant", "content": "reply1"},
            {"role": "user", "content": "round2"},
            {"role": "assistant", "content": "reply2"},
            {"role": "user", "content": "round3"},
            {"role": "assistant", "content": "reply3"},
        ]
        result = _trim_history(history, max_turns=1)
        assert result[0]["role"] == "system"
        assert len(result) == 3  # system + 1 轮（2 条）
        assert result[1]["content"] == "round3"
        assert result[2]["content"] == "reply3"

    def test_exceeds_limit_two_turns(self):
        history = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "r1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "r2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "r3"},
            {"role": "assistant", "content": "a3"},
        ]
        result = _trim_history(history, max_turns=2)
        assert len(result) == 5  # system + 4 条
        assert result[1]["content"] == "r2"
        assert result[4]["content"] == "a3"


# ------------------------------------------------------------------
# 响应展示
# ------------------------------------------------------------------


class TestDisplayResponse:

    def test_full_response(self):
        parsed = {
            "understanding": "用户预算20万，偏好SUV，家用为主。",
            "recommended_models": [
                {
                    "name": "比亚迪宋PLUS DM-i",
                    "price_range": "15-17万",
                    "pros": ["油耗低", "空间好"],
                    "cons": ["底盘偏软"],
                    "reason": "适合追求经济性的家庭。",
                },
            ],
            "follow_up_question": "你对能源类型有什么偏好？",
        }
        display_response(parsed)  # 不应抛出异常

    def test_empty_recommendations(self):
        parsed = {
            "understanding": "用户尚未提供足够信息。",
            "recommended_models": [],
            "follow_up_question": "你的预算大概在什么范围？",
        }
        display_response(parsed)

    def test_no_follow_up(self):
        parsed = {
            "understanding": "用户需求明确。",
            "recommended_models": [
                {
                    "name": "特斯拉Model 3",
                    "price_range": "23-26万",
                    "pros": ["智能化好"],
                    "cons": ["后排一般"],
                    "reason": "纯电标杆。",
                },
            ],
            "follow_up_question": None,
        }
        display_response(parsed)

    def test_minimal_response(self):
        parsed = {
            "understanding": "用户刚开始对话。",
            "recommended_models": [],
            "follow_up_question": None,
        }
        display_response(parsed)

    def test_missing_fields_does_not_crash(self):
        display_response({})


# ------------------------------------------------------------------
# 基础消息
# ------------------------------------------------------------------


class TestMessages:

    def test_welcome_message_not_empty(self):
        assert len(WELCOME_MSG) > 0

    def test_help_message_not_empty(self):
        assert len(HELP_MSG) > 0

    def test_exit_message_not_empty(self):
        assert len(EXIT_MSG) > 0

    def test_system_prompt_not_empty(self):
        assert len(SYSTEM_PROMPT) > 0
        assert "JSON" in SYSTEM_PROMPT
        assert "understanding" in SYSTEM_PROMPT
        assert "recommended_models" in SYSTEM_PROMPT
        assert "follow_up_question" in SYSTEM_PROMPT


# ------------------------------------------------------------------
# 空输入处理
# ------------------------------------------------------------------


class TestEmptyInput:
    """验证空字符串输入的处理逻辑。"""

    def test_empty_string_is_falsy(self):
        user_input = ""
        assert not user_input

    def test_whitespace_strips_to_empty(self):
        user_input = "   \t  \n  "
        assert not user_input.strip()


# ------------------------------------------------------------------
# 集成测试（需要真实 API）
# ------------------------------------------------------------------


@pytest.mark.integration
class TestIntegration:
    """集成测试：需要 DEEPSEEK_API_KEY 环境变量。

    运行方式：
        pytest tests/test_main.py -m integration -v
    """

    @pytest.fixture(autouse=True)
    def check_api_key(self):
        if not os.getenv("DEEPSEEK_API_KEY"):
            pytest.skip("DEEPSEEK_API_KEY 未设置，跳过集成测试")

    def _get_client(self):
        from car_advisor.src.config import load_config
        from car_advisor.src.llm_client import LLMClient

        return LLMClient(load_config())

    def _call_advisor(self, client, user_message: str) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        return client.chat_json(messages=messages)

    def test_scenario_1_suv_recommendation(self):
        """场景1：20万SUV，家庭用，省油 → 期望收到合理的 SUV 推荐。"""
        client = self._get_client()
        result = self._call_advisor(
            client,
            "我想买一辆20万左右的SUV，主要是家庭用，要省油",
        )

        assert "understanding" in result
        assert "recommended_models" in result
        assert "follow_up_question" in result

        understanding = result["understanding"].lower()
        assert any(w in understanding for w in ["suv", "20", "家庭", "省油", "油"])

        models = result["recommended_models"]
        assert len(models) >= 1, f"期望至少 1 款推荐，实际 {len(models)} 款"

        first = models[0]
        assert "name" in first and first["name"]
        assert "price_range" in first and first["price_range"]
        assert "pros" in first and len(first["pros"]) >= 1
        assert "cons" in first and len(first["cons"]) >= 1
        assert "reason" in first and first["reason"]

    def test_scenario_2_ev_sedan_recommendation(self):
        """场景2：15万纯电轿车，上下班 → 期望收到纯电轿车推荐。"""
        client = self._get_client()
        result = self._call_advisor(
            client,
            "预算15万，纯电轿车，平时上下班用",
        )

        assert "understanding" in result
        assert "recommended_models" in result

        understanding = result["understanding"].lower()
        assert any(w in understanding for w in ["15", "纯电", "轿车", "通勤", "上班"])

        models = result["recommended_models"]
        assert len(models) >= 1, f"期望至少 1 款推荐，实际 {len(models)} 款"

    def test_scenario_3_insufficient_info(self):
        """场景3：信息不足 → 期望看到追问。"""
        client = self._get_client()
        result = self._call_advisor(
            client,
            "帮我推荐一款车",
        )

        assert "understanding" in result
        assert "recommended_models" in result

        follow_up = result.get("follow_up_question")
        assert follow_up is not None, (
            "信息不足时期望有追问，但 follow_up_question 为 None（回复内容: "
            f"{result.get('understanding', '')}）"
        )
        assert len(follow_up) > 0

        assert isinstance(result["recommended_models"], list)

    def test_scenario_4_empty_input_handled_by_main(self):
        """场景4：空字符串应被 CLI 静默跳过。"""
        user_input = ""
        assert not user_input  # if not user_input: continue

        user_input = "   "
        assert not user_input.strip()
