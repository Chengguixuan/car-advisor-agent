"""会话存储 — 用独立 SQLite 管理对话历史和消息。

不依赖 LangGraph checkpoints，每条消息直接存数据库。
"""

import json
import sqlite3
import time
from pathlib import Path

_DB_PATH = Path(__file__).resolve().parent.parent.parent / "sessions.db"


def _conn():
    c = sqlite3.connect(str(_DB_PATH))
    c.execute(
        "CREATE TABLE IF NOT EXISTS sessions ("
        "  thread_id TEXT PRIMARY KEY,"
        "  title TEXT DEFAULT '',"
        "  msg_count INTEGER DEFAULT 0,"
        "  created_at REAL,"
        "  updated_at REAL"
        ")"
    )
    c.execute(
        "CREATE TABLE IF NOT EXISTS messages ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  thread_id TEXT NOT NULL,"
        "  role TEXT NOT NULL,"
        "  content TEXT NOT NULL,"
        "  recommendation TEXT DEFAULT '',"
        "  created_at REAL,"
        "  FOREIGN KEY(thread_id) REFERENCES sessions(thread_id)"
        ")"
    )
    c.commit()
    return c


def save_session(thread_id: str, user_input: str, title: str = ""):
    """新建或更新会话元数据。"""
    conn = _conn()
    now = time.time()
    title = title or user_input[:30]
    conn.execute(
        "INSERT INTO sessions (thread_id, title, msg_count, created_at, updated_at) "
        "VALUES (?, ?, 1, ?, ?) "
        "ON CONFLICT(thread_id) DO UPDATE SET title=?, msg_count=msg_count+1, updated_at=?",
        (thread_id, title, now, now, title, now),
    )
    conn.commit()
    conn.close()


def save_message(thread_id: str, role: str, content: str, recommendation: dict = None):
    """保存一条消息。"""
    conn = _conn()
    rec_json = json.dumps(recommendation, ensure_ascii=False) if recommendation else ""
    conn.execute(
        "INSERT INTO messages (thread_id, role, content, recommendation, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (thread_id, role, content, rec_json, time.time()),
    )
    conn.commit()
    conn.close()


def get_sessions() -> list[dict]:
    """获取所有会话列表，按更新时间降序。"""
    conn = _conn()
    rows = conn.execute(
        "SELECT thread_id, title, msg_count, created_at FROM sessions ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return [
        {"thread_id": r[0], "title": r[1], "msg_count": r[2], "created_at": r[3]}
        for r in rows
    ]


def get_messages(thread_id: str) -> list[dict]:
    """获取指定会话的全部消息。"""
    conn = _conn()
    rows = conn.execute(
        "SELECT role, content, recommendation FROM messages WHERE thread_id=? ORDER BY id ASC",
        (thread_id,),
    ).fetchall()
    conn.close()
    result = []
    for role, content, rec_json in rows:
        rec = None
        if rec_json:
            try:
                rec = json.loads(rec_json)
            except json.JSONDecodeError:
                pass
        result.append({
            "role": role,
            "content": content,
            "recommendation": rec,
            "thinking": None,
        })
    return result


def delete_session(thread_id: str):
    conn = _conn()
    conn.execute("DELETE FROM messages WHERE thread_id=?", (thread_id,))
    conn.execute("DELETE FROM sessions WHERE thread_id=?", (thread_id,))
    conn.commit()
    conn.close()
