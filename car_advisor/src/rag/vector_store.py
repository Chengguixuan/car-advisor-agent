"""车型向量存储。

基于 Chroma 向量数据库 + 本地 Embedding 模型，
将车型文档切分为语义块并建立向量索引，支持相似度检索。
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

# HuggingFace 国内镜像（解决模型下载问题）
if not os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 禁用 Windows 上的 symlink 警告
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# 默认路径
_DOCS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "car_docs"
_DEFAULT_PERSIST_DIR = Path(__file__).resolve().parent.parent.parent.parent / "chroma_db"
_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# 文本分割参数
_CHUNK_SIZE = 500      # 每块最大字符数
_CHUNK_OVERLAP = 100   # 块间重叠字符数（保持语义连贯）


class CarVectorStore:
    """车型文档向量存储。

    负责加载、切分、向量化车型文档，并对外提供语义检索接口。

    用法示例:
        store = CarVectorStore()
        store.build_index()
        results = store.search("省油的SUV", k=3)
        for doc in results:
            print(doc.page_content)
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        embedding_model: str = _EMBEDDING_MODEL,
    ):
        """
        Args:
            persist_dir:    向量库持久化目录（默认 ./chroma_db/）
            embedding_model: HuggingFace 嵌入模型名称
        """
        self.persist_dir = str(persist_dir or _DEFAULT_PERSIST_DIR)
        self.embedding_model = embedding_model

        logger.info("loading embedding model: %s (HF_ENDPOINT=%s)",
                     embedding_model, os.environ.get("HF_ENDPOINT", "default"))

        # 优先使用新版 langchain_huggingface，回退到旧版
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            logger.info("using langchain_huggingface.HuggingFaceEmbeddings")
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore[no-redef]
            logger.info("using langchain_community.embeddings.HuggingFaceEmbeddings (legacy)")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False},
        )

        self._vector_store: Optional[Chroma] = None

    # ------------------------------------------------------------------
    # 文档加载
    # ------------------------------------------------------------------

    def load_documents(self, docs_dir: Optional[str] = None) -> list[Document]:
        """从 JSON 文件加载车型文档，转换为 LangChain Document 列表。

        每个 JSON 文件的 selling_points、use_scenarios、user_feedback
        等文本字段会被分别提取为独立的 Document，保留品牌/型号作为元数据。

        Args:
            docs_dir: 车型文档目录（默认 data/car_docs/）

        Returns:
            Document 列表。
        """
        docs_dir = Path(docs_dir or _DOCS_DIR)
        documents: list[Document] = []

        json_files = sorted(docs_dir.glob("*.json"))
        if not json_files:
            logger.warning("未找到车型文档（%s）", docs_dir)
            return documents

        for file_path in json_files:
            with open(file_path, "r", encoding="utf-8") as f:
                car = json.load(f)

            brand = car.get("brand", "")
            model_name = car.get("model", "")
            title = f"{brand} {model_name}"

            # 基础元数据
            base_meta = {
                "source": str(file_path),
                "brand": brand,
                "model": model_name,
                "category": car.get("category", ""),
                "energy_type": car.get("energy_type", ""),
                "price_range": car.get("price_range", ""),
                "guide_price": car.get("guide_price", 0),
            }

            # 核心卖点（长文本）
            selling = car.get("selling_points", "")
            if selling:
                documents.append(Document(
                    page_content=selling,
                    metadata={**base_meta, "section": "selling_points", "title": title},
                ))

            # 适用场景
            scenarios = car.get("use_scenarios", [])
            if scenarios:
                documents.append(Document(
                    page_content="\n".join(scenarios),
                    metadata={**base_meta, "section": "use_scenarios", "title": title},
                ))

            # 用户口碑
            feedback = car.get("user_feedback", {})
            if feedback:
                pros = "\n".join(f"- {p}" for p in feedback.get("pros", []))
                cons = "\n".join(f"- {c}" for c in feedback.get("cons", []))
                feedback_text = f"优点：\n{pros}\n\n缺点：\n{cons}"
                documents.append(Document(
                    page_content=feedback_text,
                    metadata={**base_meta, "section": "user_feedback", "title": title},
                ))

            # 详细参数（格式化）
            specs = car.get("specs", {})
            if specs:
                spec_lines = []
                for key, val in specs.items():
                    spec_lines.append(f"- {key}: {val}")
                documents.append(Document(
                    page_content="车型参数：\n" + "\n".join(spec_lines),
                    metadata={**base_meta, "section": "specs", "title": title},
                ))

        logger.info("loaded %d documents from %d cars", len(documents), len(json_files))
        return documents

    # ------------------------------------------------------------------
    # 文档切分
    # ------------------------------------------------------------------

    def split_documents(
        self,
        documents: list[Document],
        chunk_size: int = _CHUNK_SIZE,
        chunk_overlap: int = _CHUNK_OVERLAP,
    ) -> list[Document]:
        """用 RecursiveCharacterTextSplitter 切分文档。

        按段落、句子、字符的优先级递归切分，确保语义块连贯。

        Args:
            documents:    原始 Document 列表。
            chunk_size:   每块最大字符数。
            chunk_overlap: 块间重叠字符数。

        Returns:
            切分后的 Document 列表。
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", ". ", " ", ""],
            length_function=len,
        )
        chunks = splitter.split_documents(documents)
        logger.info("split %d documents → %d chunks (size=%d, overlap=%d)",
                     len(documents), len(chunks), chunk_size, chunk_overlap)
        return chunks

    # ------------------------------------------------------------------
    # 索引构建
    # ------------------------------------------------------------------

    def build_index(
        self,
        docs_dir: Optional[str] = None,
        chunk_size: int = _CHUNK_SIZE,
        chunk_overlap: int = _CHUNK_OVERLAP,
        force_rebuild: bool = False,
    ) -> Chroma:
        """构建车型文档向量索引。

        完整流程：加载 → 切分 → 向量化 → 持久化到 Chroma。

        Args:
            docs_dir:      文档目录（默认 data/car_docs/）
            chunk_size:    切分大小
            chunk_overlap: 重叠大小
            force_rebuild: 是否强制重建（删除已有索引）

        Returns:
            Chroma 向量存储实例。
        """
        if force_rebuild and os.path.exists(self.persist_dir):
            import shutil
            logger.info("force rebuild: removing %s", self.persist_dir)
            shutil.rmtree(self.persist_dir)

        documents = self.load_documents(docs_dir)
        if not documents:
            logger.warning("没有可索引的文档")
            return Chroma(embedding_function=self.embeddings)

        chunks = self.split_documents(documents, chunk_size, chunk_overlap)

        logger.info("building vector index → %s (%d chunks)", self.persist_dir, len(chunks))
        self._vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name="car_advisor",
        )

        logger.info("index built: %d vectors in collection 'car_advisor'", len(chunks))
        return self._vector_store

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 5) -> list[Document]:
        """语义检索——返回与查询最相关的车型文档片段。

        Args:
            query: 自然语言查询，如 "省油的SUV 家用"
            k:     返回结果数

        Returns:
            相关度排序的 Document 列表。
        """
        store = self._get_store()
        results = store.similarity_search(query, k=k)
        logger.info("search: '%s' → %d results", query[:60], len(results))
        return results

    def search_with_scores(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        """带相似度分数的语义检索。"""
        store = self._get_store()
        results = store.similarity_search_with_relevance_scores(query, k=k)
        logger.info("search_with_scores: '%s' → %d results", query[:60], len(results))
        return results

    def as_retriever(self, k: int = 5):
        """返回 LangChain 兼容的 Retriever 对象。"""
        store = self._get_store()
        return store.as_retriever(search_kwargs={"k": k})

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _get_store(self) -> Chroma:
        """获取或初始化向量存储（优先加载已有索引）。"""
        if self._vector_store is not None:
            return self._vector_store

        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            logger.info("loading existing index from %s", self.persist_dir)
            self._vector_store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
                collection_name="car_advisor",
            )
        else:
            logger.info("no existing index, building new one")
            self.build_index()

        return self._vector_store
