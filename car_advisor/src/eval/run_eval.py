"""RAG 检索评估运行器。

批量运行评估数据集中的所有问题，评估检索效果：
- 召回率 (Recall@k)：期望文档是否被检索到
- 准确率 (Precision@k)：检索结果中相关文档的比例
- MRR (Mean Reciprocal Rank)：第一个相关文档的排名
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# 确保项目根目录在 path 中
_project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from car_advisor.src.rag.vector_store import CarVectorStore
from car_advisor.src.eval.eval_dataset import EVAL_QUESTIONS

logger = logging.getLogger(__name__)


def evaluate_retrieval(
    k: int = 5,
    verbose: bool = True,
) -> dict[str, Any]:
    """运行检索评估。

    Args:
        k:        检索返回文档数。
        verbose:  是否打印详细结果。

    Returns:
        评估汇总字典，包含 metrics 和 per_question 两个字段。
    """
    store = CarVectorStore()
    per_question: list[dict] = []

    total_recall = 0.0
    total_precision = 0.0
    total_mrr = 0.0
    total_hit = 0  # 至少命中一个期望文档的问题数

    for q in EVAL_QUESTIONS:
        qid = q["id"]
        question = q["question"]
        category = q["category"]
        expected = [m.lower() for m in q.get("expected_models", [])]

        t0 = time.perf_counter()
        docs = store.search(question, k=k)
        elapsed = time.perf_counter() - t0

        # 提取检索到的车型名
        retrieved_titles: list[str] = []
        for doc in docs:
            title = (doc.metadata.get("title", "") or doc.metadata.get("model", "") or "").lower()
            retrieved_titles.append(title)

        # ---- 计算指标 ----

        # Recall@k：期望文档中有多少被检索到
        if expected:
            recalled = sum(1 for e in expected if any(e in rt for rt in retrieved_titles))
            recall = recalled / len(expected)
        else:
            recall = 1.0  # 无期望文档时跳过

        # Precision@k：检索结果中有相关文档的比例
        relevant = sum(1 for rt in retrieved_titles if any(e in rt for e in expected))
        precision = relevant / len(retrieved_titles) if retrieved_titles else 0.0

        # MRR：第一个相关文档的倒数排名
        mrr = 0.0
        for rank, rt in enumerate(retrieved_titles, 1):
            if any(e in rt for e in expected):
                mrr = 1.0 / rank
                break

        total_recall += recall
        total_precision += precision
        total_mrr += mrr
        if recall > 0:
            total_hit += 1

        per_question.append({
            "id": qid,
            "question": question,
            "category": category,
            "expected": expected,
            "retrieved": retrieved_titles[:k],
            "recall": round(recall, 3),
            "precision": round(precision, 3),
            "mrr": round(mrr, 3),
            "time_ms": round(elapsed * 1000, 1),
        })

        if verbose:
            status = "OK" if recall > 0 else "MISS"
            print(f"  [{status}] {qid}: {question[:40]:40s}  "
                  f"recall={recall:.2f}  precision={precision:.2f}  "
                  f"mrr={mrr:.2f}  time={elapsed*1000:.0f}ms")

    n = len(EVAL_QUESTIONS)
    summary = {
        "total_questions": n,
        "avg_recall": round(total_recall / n, 3),
        "avg_precision": round(total_precision / n, 3),
        "mrr": round(total_mrr / n, 3),
        "hit_rate": round(total_hit / n, 3),
        "per_question": per_question,
    }

    return summary


def main():
    """运行评估并输出报告。"""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    print("=" * 60)
    print("  RAG 检索评估")
    print("=" * 60)
    print()

    summary = evaluate_retrieval(k=5, verbose=True)

    print()
    print("=" * 60)
    print("  评估汇总")
    print("=" * 60)
    print(f"  问题总数:    {summary['total_questions']}")
    print(f"  平均召回率:  {summary['avg_recall']:.2%}")
    print(f"  平均准确率:  {summary['avg_precision']:.2%}")
    print(f"  MRR:        {summary['mrr']:.3f}")
    print(f"  命中率:      {summary['hit_rate']:.2%}")

    # 按类别分组统计
    by_category: dict[str, list] = {}
    for pq in summary["per_question"]:
        cat = pq["category"]
        by_category.setdefault(cat, []).append(pq["recall"])

    print()
    print("  按类别：")
    for cat, recalls in sorted(by_category.items()):
        avg = sum(recalls) / len(recalls) if recalls else 0
        print(f"    {cat:8s}: recall={avg:.2%}  ({sum(1 for r in recalls if r > 0)}/{len(recalls)})")

    print()
    print("  推荐改进：")
    if summary["avg_recall"] < 0.5:
        print("    - 考虑增加车型文档数量或丰富文档描述")
    if summary["avg_precision"] < 0.5:
        print("    - 考虑调整 chunk_size 或使用更好的嵌入模型")
    if summary["mrr"] < 0.5:
        print("    - 考虑调整检索参数或引入重排序 (rerank)")

    # 保存结果
    output_path = _project_root / "data" / "eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  详细结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
