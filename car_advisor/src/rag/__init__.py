"""RAG 检索模块。

提供基于 Chroma 向量数据库的车型文档语义检索能力。
"""

from .vector_store import CarVectorStore
from .retriever import get_retriever

__all__ = ["CarVectorStore", "get_retriever"]
