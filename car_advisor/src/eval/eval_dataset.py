"""RAG 评估数据集。

定义 12 个测试问题，覆盖不同场景（精确搜索、模糊搜索、
对比分析、边界情况），每个问题标注期望的参考文档来源。
"""

from typing import Any

# ==========================================================================
# 评估问题定义
# ==========================================================================

EVAL_QUESTIONS: list[dict[str, Any]] = [
    # ---- 精确搜索（结构化匹配应命中）----
    {
        "id": "q01",
        "question": "推荐20万左右的混动SUV",
        "category": "精确搜索",
        "expected_brands": ["本田", "丰田"],
        "expected_models": ["CR-V", "RAV4"],
        "expected_category": "SUV",
        "expected_energy": ["混动", "HEV", "PHEV"],
        "min_results": 2,
    },
    {
        "id": "q02",
        "question": "15万以内的纯电轿车有哪些？",
        "category": "精确搜索",
        "expected_brands": ["比亚迪", "广汽"],
        "expected_models": ["秦PLUS EV", "AION S"],
        "expected_category": "轿车",
        "expected_energy": ["纯电", "BEV"],
        "min_results": 2,
    },
    {
        "id": "q03",
        "question": "10万以下的燃油轿车推荐",
        "category": "精确搜索",
        "expected_brands": ["长安"],
        "expected_models": ["逸达"],
        "expected_category": "轿车",
        "expected_energy": ["燃油"],
        "min_results": 1,
    },
    {
        "id": "q04",
        "question": "我想买一辆25万以上的MPV",
        "category": "精确搜索",
        "expected_brands": ["别克"],
        "expected_models": ["GL8"],
        "expected_category": "MPV",
        "min_results": 1,
    },
    # ---- 模糊搜索（需语义检索 RAG 兜底）----
    {
        "id": "q05",
        "question": "适合新手开的车有哪些？",
        "category": "模糊搜索",
        "expected_keywords": ["容易", "轻快", "代步", "实用"],
        "min_results": 2,
    },
    {
        "id": "q06",
        "question": "省油的家用SUV，空间要大",
        "category": "模糊搜索",
        "expected_brands": ["本田", "丰田", "比亚迪"],
        "expected_models": ["CR-V", "RAV4", "宋PLUS"],
        "expected_keywords": ["省油", "油耗", "空间"],
        "min_results": 2,
    },
    {
        "id": "q07",
        "question": "长途自驾游开什么车比较舒服？",
        "category": "模糊搜索",
        "expected_models": ["GL8"],
        "expected_keywords": ["长途", "舒适", "NVH", "空间"],
        "min_results": 1,
    },
    {
        "id": "q08",
        "question": "安全性最好的家用车推荐",
        "category": "模糊搜索",
        "expected_models": ["星越L"],
        "expected_keywords": ["安全", "CMA", "碰撞"],
        "min_results": 1,
    },
    # ---- 对比分析 ----
    {
        "id": "q09",
        "question": "对比比亚迪宋PLUS和本田CR-V，哪个更适合家用？",
        "category": "对比分析",
        "expected_models": ["宋PLUS", "CR-V"],
        "min_results": 2,
    },
    {
        "id": "q10",
        "question": "特斯拉Model 3和比亚迪秦PLUS EV续航对比",
        "category": "对比分析",
        "expected_models": ["Model 3", "秦PLUS EV"],
        "expected_keywords": ["续航", "电池", "充电"],
        "min_results": 2,
    },
    # ---- 用途导向 ----
    {
        "id": "q11",
        "question": "家里有两个小孩，预算20万，推荐什么车？",
        "category": "用途导向",
        "expected_keywords": ["家庭", "空间", "安全"],
        "min_results": 2,
    },
    {
        "id": "q12",
        "question": "上下班通勤单程30公里，不想加油，买什么车？",
        "category": "用途导向",
        "expected_energy": ["纯电", "BEV"],
        "expected_keywords": ["通勤", "纯电", "续航", "充电"],
        "min_results": 1,
    },
]

# ==========================================================================
# 评估指标
# ==========================================================================


def get_expected_ids(question: dict) -> list[str]:
    """返回期望的文档来源列表。"""
    return question.get("expected_models", [])


def get_all_expected_ids() -> set[str]:
    """返回所有问题涉及的车型。"""
    ids: set[str] = set()
    for q in EVAL_QUESTIONS:
        for m in q.get("expected_models", []):
            ids.add(m.lower())
    return ids
