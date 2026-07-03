"""公共工具函数。

提供消息处理、序列化等跨模块复用的函数，消除重复代码。
"""


def get_role(msg) -> str:
    """兼容 dict / LangChain Message 对象，获取消息角色。

    Args:
        msg: dict 或 LangChain Message 对象。

    Returns:
        消息角色字符串（system / user / assistant / ai / human / tool）。
    """
    if isinstance(msg, dict):
        return msg.get("role", "") or msg.get("type", "")
    return getattr(msg, "role", "") or getattr(msg, "type", "") or ""


def get_content(msg) -> str:
    """兼容 dict / LangChain Message 对象，获取消息文本。

    Args:
        msg: dict 或 LangChain Message 对象。

    Returns:
        消息内容字符串。
    """
    if isinstance(msg, dict):
        return msg.get("content", "") or ""
    return getattr(msg, "content", "") or ""


def get_tool_calls(msg) -> list:
    """兼容 dict / LangChain Message 对象，获取 tool_calls。

    Args:
        msg: dict 或 LangChain Message 对象。

    Returns:
        tool_calls 列表。
    """
    if isinstance(msg, dict):
        return msg.get("tool_calls", []) or []
    return getattr(msg, "tool_calls", []) or []


def serialize_messages(messages: list) -> list[dict]:
    """将 LangChain Message 对象序列化为可 JSON 化的 dict 列表。

    Args:
        messages: 消息列表。

    Returns:
        字典列表，每项含 role 和 content。
    """
    result = []
    for msg in messages:
        result.append({
            "role": get_role(msg),
            "content": get_content(msg),
        })
    return result
