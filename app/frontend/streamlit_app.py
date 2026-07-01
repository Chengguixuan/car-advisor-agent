"""买车智能体 — Streamlit 前端界面。

提供对话式购车顾问体验，支持多轮对话和流式展示。

启动方式:
    streamlit run app/frontend/streamlit_app.py

前置条件:
    - 后端 API 已启动 (uvicorn app.main:app --port 8000)
"""

import json
import time
import uuid
from typing import Optional

import requests
import streamlit as st

# ==========================================================================
# 配置
# ==========================================================================

API_BASE = "http://localhost:8000"
CHAT_URL = f"{API_BASE}/chat"
STREAM_URL = f"{API_BASE}/chat/stream"

st.set_page_config(
    page_title="购车智能体",
    page_icon="🚗",
    layout="wide",
)

# ==========================================================================
# 会话状态初始化
# ==========================================================================

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"web_{uuid.uuid4().hex[:8]}"

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "你好！我是购车智能体 🚗\n\n我可以帮你：\n- 根据预算和需求推荐车型\n- 对比多款车型的优缺点\n- 分析燃油/混动/纯电的选择\n\n告诉我你的购车需求吧！",
            "recommendation": None,
            "thinking": None,
        }
    ]


# ==========================================================================
# 辅助函数
# ==========================================================================


def call_chat(user_input: str, thread_id: str) -> dict:
    """调用后端同步接口。"""
    resp = requests.post(
        CHAT_URL,
        json={"user_input": user_input, "thread_id": thread_id},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def call_chat_stream(user_input: str, thread_id: str):
    """调用后端流式接口，逐步 yield 事件。"""
    resp = requests.post(
        STREAM_URL,
        json={"user_input": user_input, "thread_id": thread_id},
        timeout=120,
        stream=True,
    )
    for line in resp.iter_lines():
        if line and line.startswith("data:"):
            data = json.loads(line[5:].strip())
            yield data
        elif line and line.startswith("event:"):
            pass  # SSE event type
        elif line:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                pass


def render_recommendation(rec: dict):
    """渲染推荐结果卡片。"""
    understanding = rec.get("understanding", "")
    if understanding:
        st.markdown(f"**📋 需求理解：** {understanding}")

    models = rec.get("recommended_models", [])
    if models:
        st.markdown("---")
        st.markdown("### 🚘 推荐车型")
        cols = st.columns(min(len(models), 3))
        for i, model in enumerate(models):
            with cols[i % 3]:
                _render_model_card(model, i + 1)

    follow_up = rec.get("follow_up_question")
    if follow_up:
        st.info(f"💬 {follow_up}")


def _render_model_card(model: dict, index: int):
    """渲染单个车型卡片。"""
    name = model.get("name", "未知车型")
    price = model.get("price_range", "价格待查")
    pros = model.get("pros", [])
    cons = model.get("cons", [])
    reason = model.get("reason", "")

    with st.container(border=True):
        st.markdown(f"**#{index}  {name}**")
        st.caption(f"💰 {price}")

        if pros:
            st.markdown("✅ **优点**")
            for p in pros:
                st.markdown(f"- {p}")

        if cons:
            st.markdown("⚠️ **注意**")
            for c in cons:
                st.markdown(f"- {c}")

        if reason:
            st.markdown(f"💡 *{reason}*")


# ==========================================================================
# 页面渲染
# ==========================================================================

# 侧边栏
with st.sidebar:
    st.title("🚗 购车智能体")
    st.caption("基于 LangGraph + RAG 的智能购车顾问")

    st.markdown("---")
    st.markdown("**会话 ID**")
    st.code(st.session_state.thread_id, language=None)

    if st.button("🔄 新对话", use_container_width=True):
        st.session_state.thread_id = f"web_{uuid.uuid4().hex[:8]}"
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "你好！我是购车智能体 🚗\n\n告诉我你的购车需求吧！",
                "recommendation": None,
                "thinking": None,
            }
        ]
        st.rerun()

    st.markdown("---")
    st.markdown("**示例问题**")
    examples = [
        "推荐一款20万左右的混动SUV",
        "预算15万，纯电轿车",
        "比亚迪宋PLUS和本田CR-V哪个好？",
        "有没有适合二胎家庭的7座车？",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending_input = ex
            st.rerun()

# 主区域
st.title("🚗 购车智能体")
st.caption("你的专业购车顾问 — 告诉我需求，帮你找到最合适的车")

# 显示历史对话
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # 可折叠的思考过程
        if msg.get("thinking"):
            with st.expander("🔍 查看分析过程", expanded=False):
                for step in msg["thinking"]:
                    st.caption(f"• {step}")

        # 推荐卡片
        if msg.get("recommendation"):
            render_recommendation(msg["recommendation"])

# 输入区域
user_input = st.chat_input("描述你的购车需求...")

# 处理预填充的示例问题
if "pending_input" in st.session_state:
    user_input = st.session_state.pop("pending_input")

if user_input:
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": user_input, "thinking": None, "recommendation": None})

    with st.chat_message("user"):
        st.markdown(user_input)

    # 调用 API
    with st.chat_message("assistant"):
        placeholder = st.empty()
        status = st.status("🔍 正在分析...", expanded=True)

        thinking_steps = []
        response_text = ""
        recommendation = None

        try:
            # 使用流式接口展示进度
            for event in call_chat_stream(user_input, st.session_state.thread_id):
                ev_type = event.get("event", "")
                ev_data = event.get("data", {})

                if ev_type == "start":
                    status.write("🚀 开始分析...")

                elif ev_type == "tool_call":
                    tools = ev_data.get("tools", [])
                    for t in tools:
                        tool_name = t.get("name", "?")
                        step = f"🔧 调用工具: {tool_name}"
                        thinking_steps.append(step)
                        status.write(step)

                elif ev_type == "tool_result":
                    msgs = ev_data.get("messages", [])
                    if msgs:
                        thinking_steps.append(f"✅ 获取到 {len(msgs)} 条数据")
                        status.write(f"✅ 获取到 {len(msgs)} 条数据")

                elif ev_type == "message":
                    response_text = ev_data.get("content", "")
                    placeholder.markdown(response_text)

                elif ev_type == "final_recommendation":
                    recommendation = ev_data
                    status.write("🎯 生成最终推荐...")

                elif ev_type == "done":
                    elapsed = ev_data.get("elapsed_ms", 0)
                    status.write(f"⏱️ 完成（{elapsed/1000:.1f}秒）")

        except Exception as e:
            response_text = f"❌ 调用失败：{e}"
            placeholder.error(response_text)
            thinking_steps.append(f"❌ 错误: {e}")

        status.update(label="✅ 分析完成", state="complete", expanded=False)

        if not response_text and recommendation:
            response_text = "已生成推荐，详见下方卡片"

        if not response_text:
            response_text = "分析完成"

        placeholder.markdown(response_text)

        # 展示推荐
        if recommendation:
            render_recommendation(recommendation)

        # 保存助手消息
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "thinking": thinking_steps if thinking_steps else None,
            "recommendation": recommendation,
        })
