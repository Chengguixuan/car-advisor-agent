"""买车智能体 — Streamlit 前端界面。

提供对话式购车顾问体验，支持多轮对话、流式展示和会话历史切换。

启动方式:
    streamlit run app/frontend/streamlit_app.py
"""

import json
import os
import sqlite3
import sys
import uuid
from pathlib import Path

import msgpack
import requests
import streamlit as st

# 确保可导入 car_advisor
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# ==========================================================================
# 配置
# ==========================================================================

API_BASE = "http://localhost:8000"
CHAT_URL = f"{API_BASE}/chat"
STREAM_URL = f"{API_BASE}/chat/stream"
CHECKPOINTS_DB = _project_root / "checkpoints.db"

st.set_page_config(page_title="购车智能体", page_icon="🚗", layout="wide")

# ==========================================================================
# 会话状态初始化
# ==========================================================================

DEFAULT_WELCOME = "你好！我是购车智能体 🚗\n\n我可以帮你：\n- 根据预算和需求推荐车型\n- 对比多款车型的优缺点\n- 分析燃油/混动/纯电的选择\n\n告诉我你的购车需求吧！"

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"web_{uuid.uuid4().hex[:8]}"

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": DEFAULT_WELCOME, "recommendation": None, "thinking": None}
    ]

if "session_label" not in st.session_state:
    st.session_state.session_label = "新会话"


# ==========================================================================
# 会话历史管理
# ==========================================================================


def load_sessions() -> list[dict]:
    """从 checkpoints.db 读取所有会话摘要。

    Returns:
        [{"thread_id": ..., "label": ..., "msg_count": ...}, ...]
    """
    if not CHECKPOINTS_DB.exists():
        return []

    sessions = []
    try:
        conn = sqlite3.connect(str(CHECKPOINTS_DB))
        cur = conn.cursor()
        cur.execute(
            "SELECT thread_id, MAX(checkpoint_id), COUNT(*) "
            "FROM checkpoints GROUP BY thread_id "
            "ORDER BY MAX(checkpoint_id) DESC"
        )
        for tid, _, chk_count in cur.fetchall():
            # 尝试从第一条 checkpoint 提取用户消息内容作为标签
            label = tid[:12]
            try:
                cur.execute(
                    "SELECT checkpoint FROM checkpoints WHERE thread_id=? ORDER BY checkpoint_id ASC LIMIT 1",
                    (tid,),
                )
                row = cur.fetchone()
                if row:
                    data = msgpack.unpackb(row[1], raw=False)
                    msgs = data.get("channel_values", {}).get("messages", [])
                    for m in msgs:
                        if isinstance(m, dict):
                            content = m.get("content", "") or m.get("text", "")
                            if content and len(content) > 0:
                                label = content[:24] + ("..." if len(content) > 24 else "")
                                break
                        elif hasattr(m, "content"):
                            content = str(getattr(m, "content", ""))
                            if content:
                                label = content[:24] + ("..." if len(content) > 24 else "")
                                break
            except Exception:
                pass

            sessions.append({
                "thread_id": tid,
                "label": label,
                "msg_count": chk_count * 2,  # 估算：每个 checkpoint 约 2 条消息
                "last_id": _,
            })
        conn.close()
    except Exception as e:
        st.sidebar.warning(f"加载历史会话失败: {e}")

    return sessions


def load_history_from_api(thread_id: str) -> list[dict]:
    """通过后端 API 加载指定会话的消息历史。

    向 /chat 发一个空查询，获取 state 中已有的完整消息列表。
    如果失败则返回空列表（侧边栏显示提示）。
    """
    try:
        resp = requests.post(
            f"{API_BASE}/chat",
            json={"user_input": "继续", "thread_id": thread_id},
            timeout=60,
        )
        if resp.status_code == 200:
            data = resp.json()
            msgs = data.get("messages", [])
            if msgs:
                # 从完整 messages 中提取最后一段对话
                result = []
                for m in msgs:
                    role = m.get("role", "assistant") if isinstance(m, dict) else getattr(m, "role", "assistant")
                    content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
                    # 转换给 Streamlit 显示
                    role_map = {"human": "user", "ai": "assistant", "system": "assistant", "tool": "assistant"}
                    display_role = role_map.get(role, "assistant") if isinstance(role, str) else "assistant"
                    if content:
                        result.append({
                            "role": display_role,
                            "content": content[:200],
                            "recommendation": None,
                            "thinking": None,
                        })
                return result
    except Exception:
        pass
    return []


def switch_session(thread_id: str, label: str, messages: list[dict]):
    """切换到指定会话。"""
    st.session_state.thread_id = thread_id
    st.session_state.session_label = label
    st.session_state.messages = messages
    st.session_state.pop("pending_input", None)


# ==========================================================================
# 辅助函数
# ==========================================================================


def call_chat_stream(user_input: str, thread_id: str):
    """调用后端流式接口，逐步 yield 事件。"""
    resp = requests.post(
        STREAM_URL, json={"user_input": user_input, "thread_id": thread_id},
        timeout=120, stream=True,
    )
    for line in resp.iter_lines():
        if line and line.startswith("data:"):
            yield json.loads(line[5:].strip())
        elif line and not line.startswith("event:"):
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
    with st.container(border=True):
        st.markdown(f"**#{index}  {model.get('name', '未知车型')}**")
        st.caption(f"💰 {model.get('price_range', '价格待查')}")
        for p in model.get("pros", []):
            st.markdown(f"- ✅ {p}")
        for c in model.get("cons", []):
            st.markdown(f"- ⚠️ {c}")
        reason = model.get("reason", "")
        if reason:
            st.markdown(f"💡 *{reason}*")


# ==========================================================================
# 页面渲染
# ==========================================================================

# ---- 侧边栏 ----
with st.sidebar:
    st.title("🚗 购车智能体")
    st.caption("基于 LangGraph + RAG 的智能购车顾问")

    # 新建对话
    if st.button("🔄 新对话", use_container_width=True):
        new_id = f"web_{uuid.uuid4().hex[:8]}"
        st.session_state.thread_id = new_id
        st.session_state.session_label = "新会话"
        st.session_state.messages = [
            {"role": "assistant", "content": DEFAULT_WELCOME, "recommendation": None, "thinking": None}
        ]
        st.rerun()

    st.markdown("---")

    # 历史会话列表
    st.markdown("**📋 历史会话**")
    sessions = load_sessions()
    current_tid = st.session_state.thread_id

    if sessions:
        for s in sessions:
            tid = s["thread_id"]
            label = s["label"]
            count = s["msg_count"]
            is_current = (tid == current_tid)

            prefix = "🔵 " if is_current else "   "
            btn_label = f"{prefix}{label}  (~{count}条)"

            if st.button(btn_label, key=f"session_{tid}", use_container_width=True,
                         type="primary" if is_current else "tertiary"):
                if not is_current:
                    with st.spinner("加载历史消息..."):
                        msgs = load_history_from_api(tid)
                    if not msgs:
                        msgs = [{"role": "assistant", "content": f"会话 {label}（消息加载失败，已切换到该会话，可以继续对话）",
                                  "recommendation": None, "thinking": None}]
                    switch_session(tid, label, msgs)
                    st.rerun()
    else:
        st.caption("暂无历史会话")

    # 示例问题
    st.markdown("---")
    st.markdown("**💡 示例问题**")
    examples = [
        "推荐一款20万左右的混动SUV",
        "预算15万，纯电轿车",
        "比亚迪宋PLUS和本田CR-V哪个好？",
        "有没有适合二胎家庭的7座车？",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:10]}", use_container_width=True):
            st.session_state.pending_input = ex
            st.rerun()

# ---- 主区域 ----
st.title("🚗 购车智能体")
st.caption(f"当前会话：{st.session_state.session_label}")

# 显示历史对话
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("thinking"):
            with st.expander("🔍 查看分析过程", expanded=False):
                for step in msg["thinking"]:
                    st.caption(f"• {step}")
        if msg.get("recommendation"):
            render_recommendation(msg["recommendation"])

# 输入区域
user_input = st.chat_input("描述你的购车需求...")
if "pending_input" in st.session_state:
    user_input = st.session_state.pop("pending_input")

if user_input:
    # 添加用户消息
    st.session_state.messages.append({
        "role": "user", "content": user_input, "thinking": None, "recommendation": None,
    })
    # 更新会话标签
    st.session_state.session_label = user_input[:24] + ("..." if len(user_input) > 24 else "")

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        status = st.status("🔍 正在分析...", expanded=True)

        thinking_steps = []
        response_text = ""
        recommendation = None

        try:
            for event in call_chat_stream(user_input, st.session_state.thread_id):
                ev_type = event.get("event", "")
                ev_data = event.get("data", {})

                if ev_type == "tool_call":
                    for t in ev_data.get("tools", []):
                        step = f"🔧 调用工具: {t.get('name', '?')}"
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
                    status.write(f"⏱️ 完成（{ev_data.get('elapsed_ms', 0)/1000:.1f}秒）")

        except Exception as e:
            response_text = f"❌ 调用失败：{e}"
            placeholder.error(response_text)

        status.update(label="✅ 分析完成", state="complete", expanded=False)

        if not response_text:
            response_text = "已生成推荐，详见下方卡片" if recommendation else "分析完成"
        placeholder.markdown(response_text)

        if recommendation:
            render_recommendation(recommendation)

        st.session_state.messages.append({
            "role": "assistant", "content": response_text,
            "thinking": thinking_steps if thinking_steps else None,
            "recommendation": recommendation,
        })
