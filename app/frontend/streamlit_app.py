"""买车智能体 — Streamlit 前端界面。"""

import json
import sys
import uuid
from pathlib import Path

import requests
import streamlit as st

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from car_advisor.src.session_store import get_sessions, get_messages

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="购车智能体", page_icon="🚗", layout="wide")

st.markdown("""<style>
    #MainMenu, footer {visibility: hidden;}
    .stButton button {border-radius: 10px; transition: all 0.2s;}
    .stButton button:hover {transform: scale(1.01);}
</style>""", unsafe_allow_html=True)

DEFAULT_WELCOME = "你好！我是你的购车顾问 🚗\n\n根据预算、车型、能源类型等需求，帮你找到最合适的车。\n\n试试：推荐一款 20 万左右的混动 SUV"
DEFAULT_HELLO = {"role": "assistant", "content": DEFAULT_WELCOME, "recommendation": None}

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"web_{uuid.uuid4().hex[:8]}"
if "messages" not in st.session_state:
    st.session_state.messages = [DEFAULT_HELLO]
if "session_label" not in st.session_state:
    st.session_state.session_label = "新会话"


def switch_session(tid, label):
    msgs = get_messages(tid) or [DEFAULT_HELLO]
    st.session_state.thread_id = tid
    st.session_state.session_label = label
    st.session_state.messages = msgs
    st.session_state.pop("pending_input", None)


def call_chat_stream(user_input, thread_id):
    resp = requests.post(f"{API_BASE}/chat/stream",
        json={"user_input": user_input, "thread_id": thread_id}, timeout=120, stream=True)
    for raw in resp.iter_lines():
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        if not line: continue
        if line.startswith("data:"): yield json.loads(line[5:].strip())
        elif not line.startswith("event:"):
            try: yield json.loads(line)
            except json.JSONDecodeError: pass


def _parse_num(s):
    """从字符串提取数值，如 '4.5L/100km'→4.5, '110km'→110, '-'→None"""
    import re
    s = str(s).strip()
    if s in ("-", "", "无"):
        return None
    m = re.search(r"(\d+\.?\d*)", s)
    return float(m.group(1)) if m else None


def _green(val):
    return f"<span style='color:#2e7d32;font-weight:bold'>{val}</span>"
def _red(val):
    return f"<span style='color:#c62828;font-weight:bold'>{val}</span>"


def _render_comparison_table(table_data: dict):
    """渲染参数对比表格，颜色标注最优/最差值。"""
    headers = table_data.get("headers", [])
    rows = table_data.get("rows", [])
    if not rows:
        return

    # 指标 → 方向：越大越好(left) 还是 越小越好(right)
    bigger_better = {"续航", "纯电续航", "综合续航", "轴距", "功率", "后备箱", "加速"}
    smaller_better = {"价格", "油耗", "综合油耗", "电耗", "百公里加速"}

    styled_rows = []
    for row in rows:
        label = row[0]
        vals = row[1:]
        nums = [_parse_num(v) for v in vals]
        styled = [label]

        if any(n is not None for n in nums):
            valid = [(i, n) for i, n in enumerate(nums) if n is not None]
            if valid:
                if label in bigger_better:
                    best_i = max(valid, key=lambda x: x[1])[0]
                    for i, v in enumerate(vals):
                        styled.append(_green(v) if i == best_i else v)
                elif label in smaller_better:
                    best_i = min(valid, key=lambda x: x[1])[0]
                    worst_i = max(valid, key=lambda x: x[1])[0]
                    for i, v in enumerate(vals):
                        if i == best_i: styled.append(_green(v))
                        elif i == worst_i and best_i != worst_i: styled.append(_red(v))
                        else: styled.append(v)
                else:
                    styled.extend(vals)
        else:
            styled.extend(vals)
        styled_rows.append(styled)

    html = "<table style='width:100%;border-collapse:collapse;margin:12px 0'>"
    html += "<tr style='background:#f5f5f5'>" + "".join(f"<th style='padding:8px;border:1px solid #ddd'>{h}</th>" for h in headers) + "</tr>"
    for row in styled_rows:
        html += "<tr>" + "".join(f"<td style='padding:8px;border:1px solid #ddd'>{c}</td>" for c in row) + "</tr>"
    html += "</table>"
    st.markdown(html, unsafe_allow_html=True)


def render_recommendation(rec):
    if rec.get("understanding"):
        st.info(f"📋 {rec['understanding']}")
    if rec.get("tradeoff_summary"):
        st.markdown(f"⚖️ {rec['tradeoff_summary']}")
    table = rec.get("comparison_table")
    if table and table.get("rows"):
        _render_comparison_table(table)
    models = rec.get("recommended_models", [])
    if models:
        if not table:
            st.markdown("### 🚘 推荐车型")
            for i, m in enumerate(models, 1):
                with st.container(border=True):
                    st.markdown(f"**{i}. {m.get('name', '?')}**  |  💰 {m.get('price_range', '?')}")
                    if m.get("pros"): st.caption(" ✅  " + "　".join(m["pros"][:3]))
                    if m.get("cons"): st.caption(" ⚠️  " + "　".join(m["cons"][:2]))
                    if m.get("reason"): st.caption(f"💡 {m['reason']}")
    if rec.get("follow_up_question"):
        st.success(f"💬 {rec['follow_up_question']}")


# ── 侧边栏 ──
with st.sidebar:
    st.markdown("## 🚗 购车智能体")
    if st.button("＋ 新对话", key="btn_new_chat", use_container_width=True):
        st.session_state.thread_id = f"web_{uuid.uuid4().hex[:8]}"
        st.session_state.session_label = "新会话"
        st.session_state.messages = [DEFAULT_HELLO]
        st.session_state.pop("pending_input", None)
        st.rerun()

    st.divider()
    st.caption("**📋 历史会话**")
    sessions = get_sessions()
    current_tid = st.session_state.thread_id
    confirm_target = st.session_state.get("_confirm_delete")

    if sessions:
        for s in sessions:
            tid, label, count = s["thread_id"], s["title"] or s["thread_id"][:12], s["msg_count"]
            is_current = (tid == current_tid)

            if confirm_target == tid:
                st.warning(f"确认删除「{label}」？")
                ca, cb = st.columns(2)
                if ca.button("✅ 确认", key=f"ok_{tid}", use_container_width=True):
                    try:
                        requests.delete(f"{API_BASE}/conversation/{tid}", timeout=5)
                        st.session_state.pop("_confirm_delete", None)
                        if is_current:
                            st.session_state.thread_id = f"web_{uuid.uuid4().hex[:8]}"
                            st.session_state.session_label = "新会话"
                            st.session_state.messages = [DEFAULT_HELLO]
                        st.rerun()
                    except Exception:
                        st.toast("删除失败")
                if cb.button("取消", key=f"cancel_{tid}", use_container_width=True):
                    st.session_state.pop("_confirm_delete", None)
                    st.rerun()
            else:
                c1, c2 = st.columns([9, 1])
                c1.button(f"{'● ' if is_current else ''}{label}", key=f"s_{tid}",
                          use_container_width=True, type="primary" if is_current else "tertiary",
                          on_click=switch_session if not is_current else None,
                          args=(tid, label) if not is_current else ())
                if c2.button("🗑", key=f"d_{tid}", help="删除"):
                    st.session_state["_confirm_delete"] = tid
                    st.rerun()
    else:
        st.caption("暂无历史会话")

    st.divider()
    st.caption("**💡 试试这些**")
    for ex in ["推荐 20 万左右的混动 SUV", "预算 15 万，纯电轿车", "比亚迪和丰田对比"]:
        if st.button(ex, key=f"ex_{ex[:6]}", use_container_width=True):
            st.session_state.pending_input = ex
            st.rerun()

# ── 主区域 ──
st.markdown("### 🚗 购车智能体")
st.caption(f"当前：{st.session_state.session_label}")

def _render_assistant_msg(content: str):
    """从 assistant 回复中提取结构化内容展示，过滤掉 JSON 原文。"""
    if not content or not content.strip():
        return
    # 尝试直接解析完整 JSON
    for text in [content.strip()]:
        if text.startswith("{"):
            try:
                render_recommendation(json.loads(text))
                return
            except json.JSONDecodeError:
                pass
    # 尝试从 markdown 代码块或混杂文本中提取 JSON
    import re
    m = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', content)
    if not m:
        m = re.search(r'(\{[^{}]*"understanding"[^{}]*"recommended_models"[\s\S]*?\})', content)
    if m:
        try:
            render_recommendation(json.loads(m.group(1)))
            return
        except json.JSONDecodeError:
            pass
    # 兜底：去掉明显是 JSON 的行，只显示纯文本部分
    lines = content.split("\n")
    cleaned = [l for l in lines if not l.strip().startswith(("{", "}", '  "', '  ]', '  }'))]
    if cleaned:
        st.markdown("\n".join(cleaned))


# 历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        elif msg.get("recommendation"):
            render_recommendation(msg["recommendation"])
        elif msg.get("content"):
            st.markdown(msg["content"])

# ── 输入 ──
user_input = st.chat_input("描述你的购车需求...")
if "pending_input" in st.session_state:
    user_input = st.session_state.pop("pending_input")

if user_input:
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": user_input, "recommendation": None})
    st.session_state.session_label = user_input[:24] + ("..." if len(user_input) > 24 else "")

    # 立即显示用户消息
    with st.chat_message("user"):
        st.markdown(user_input)

    # 流式调用
    with st.chat_message("assistant"):
        status_text = st.empty()
        response_text, recommendation = "", None
        seen_tools = set()

        try:
            status_text.info("🔍 正在分析您的问题...")
            for event in call_chat_stream(user_input, st.session_state.thread_id):
                et, d = event.get("event", ""), event.get("data", {})
                if et == "tool_call":
                    for t in d.get("tools", []):
                        tname = t.get("name", "")
                        if tname not in seen_tools:
                            seen_tools.add(tname)
                            if "search_local" in tname:
                                status_text.info("🔍 正在检索符合条件的车型...")
                            elif "online" in tname:
                                status_text.info("🌐 正在搜索相关车型近期优惠活动...")
                            elif "compare" in tname:
                                status_text.info("📊 正在对比车型参数...")
                            else:
                                status_text.info("🔧 正在查询数据...")
                elif et == "final_recommendation":
                    recommendation = d
        except Exception as e:
            status_text.error(f"❌ {e}")

        status_text.empty()
        if recommendation:
            render_recommendation(recommendation)
            response_text = recommendation.get("understanding", "")
        elif response_text:
            st.markdown(response_text)

        st.session_state.messages.append({
            "role": "assistant", "content": response_text,
            "recommendation": recommendation,
        })
