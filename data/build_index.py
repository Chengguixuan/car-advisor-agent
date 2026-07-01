#!/usr/bin/env python
"""构建车型文档向量索引。

运行后从 data/car_docs/ 加载所有车型 JSON 文档，
用 all-MiniLM-L6-v2 模型生成 Embedding，
存入 Chroma 向量数据库（持久化到 ./chroma_db/）。

用法:
    python data/build_index.py          # 增量构建（已有索引则加载）
    python data/build_index.py --force  # 强制重建
    python data/build_index.py --test   # 构建后运行测试查询
"""

import argparse
import logging
import sys
from pathlib import Path

# 确保 car_advisor 包可导入
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from car_advisor.src.rag.vector_store import CarVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("build_index")


def main():
    parser = argparse.ArgumentParser(description="构建车型文档向量索引")
    parser.add_argument("--force", action="store_true", help="强制重建已有索引")
    parser.add_argument("--test", action="store_true", help="索引构建后运行测试查询")
    args = parser.parse_args()

    store = CarVectorStore()

    # 构建索引
    logger.info("开始构建向量索引...")
    store.build_index(force_rebuild=args.force)
    logger.info("索引构建完成 ✅")

    # 测试查询
    if args.test:
        _run_tests(store)


def _run_tests(store: CarVectorStore):
    """运行一组测试查询验证索引效果。"""
    test_queries = [
        "省油的SUV，家庭用，20万左右",
        "纯电轿车，上下班通勤，15万",
        "长途自驾游开什么车好",
    ]

    print("\n" + "=" * 60)
    print("  索引测试查询")
    print("=" * 60)

    for query in test_queries:
        print(f"\n查询: {query}")
        print("-" * 40)
        results = store.search_with_scores(query, k=3)
        for i, (doc, score) in enumerate(results, 1):
            title = doc.metadata.get("title", "?")
            section = doc.metadata.get("section", "?")
            content_preview = doc.page_content[:80].replace("\n", " ")
            print(f"  #{i} [{title}] (section={section}, score={score:.3f})")
            print(f"     {content_preview}...")

    print("\nIndex test completed [OK]")


if __name__ == "__main__":
    main()
