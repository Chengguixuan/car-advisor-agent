"""LangChain 兼容的 Retriever 和 RAG 搜索工具。

提供 get_retriever() 工厂函数和 create_rag_tool() 工具创建函数，
供 LangGraph Agent 在推理过程中使用。
"""

import json
import logging
from typing import Optional

from langchain_core.tools import tool

from .vector_store import CarVectorStore

logger = logging.getLogger(__name__)

# 全局单例，避免重复加载 Embedding 模型和向量索引
_store: Optional[CarVectorStore] = None


def _get_store() -> CarVectorStore:
    """延迟初始化全局 CarVectorStore 单例。"""
    global _store
    if _store is None:
        _store = CarVectorStore()
    return _store


def get_retriever(k: int = 5):
    """返回 LangChain 兼容的 Retriever 对象。

    可用于 LangGraph Agent 的 create_retriever_tool 或
    直接作为 graph 节点的检索器。

    Args:
        k: 每次检索返回的文档数。

    Returns:
        Chroma 向量存储的 Retriever。
    """
    store = _get_store()
    return store.as_retriever(k=k)


@tool
def search_car_docs(query: str, k: int = 3) -> str:
    """搜索车型文档库，获取详细车型信息。

    基于语义相似度从向量数据库中检索与查询最相关的车型文档片段。
    返回的文档包含品牌、型号、价格、参数、优缺点、适用场景等信息。

    适用场景：
    - 用户想了解某款车型的详细参数和口碑
    - 模糊搜索：如"省油的家用SUV"、"20万左右的纯电车"
    - 对比多个候选车型的具体信息

    Args:
        query: 搜索查询词。例如 "省油的SUV 家庭用"、"纯电轿车 续航"、"混动和燃油怎么选"
        k:     返回文档数量，默认3条

    Returns:
        JSON 字符串，包含相关文档的内容和元数据。
    """
    store = _get_store()
    docs = store.search(query, k=k)

    results = []
    for doc in docs:
        results.append({
            "content": doc.page_content[:500],
            "title": doc.metadata.get("title", ""),
            "category": doc.metadata.get("category", ""),
            "energy_type": doc.metadata.get("energy_type", ""),
            "price_range": doc.metadata.get("price_range", ""),
            "section": doc.metadata.get("section", ""),
        })

    logger.info("search_car_docs: '%s' → %d results", query[:60], len(results))
    return json.dumps(results, ensure_ascii=False, indent=2)


def get_rag_tools():
    """返回 RAG 检索相关的工具列表。

    供 build_agent() 中绑定到 LLM 使用。

    Returns:
        工具列表，包含 search_car_docs。
    """
    return [search_car_docs]
