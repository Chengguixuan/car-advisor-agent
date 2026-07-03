"""RAG 评估数据集。

定义 32 个测试问题，覆盖不同场景（精确搜索、模糊搜索、对比分析、
用途导向、边界测试），每个问题标注期望的参考文档来源。
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
        "expected_models": ["CR-V", "RAV4荣放双擎"],
    },
    {
        "id": "q02",
        "question": "15万以内的纯电轿车有哪些？",
        "category": "精确搜索",
        "expected_models": ["秦PLUS EV", "AION S Plus"],
    },
    {
        "id": "q03",
        "question": "10万以下的燃油轿车推荐",
        "category": "精确搜索",
        "expected_models": ["逸达"],
    },
    {
        "id": "q04",
        "question": "我想买一辆25万以上的MPV",
        "category": "精确搜索",
        "expected_models": ["GL8"],
    },
    {
        "id": "q13",
        "question": "10万左右的家用轿车推荐",
        "category": "精确搜索",
        "expected_models": ["逸达", "轩逸"],
    },
    {
        "id": "q14",
        "question": "15万以内的纯电轿车",
        "category": "精确搜索",
        "expected_models": ["秦PLUS EV", "启源A06"],
    },
    {
        "id": "q15",
        "question": "20万以下的混动SUV",
        "category": "精确搜索",
        "expected_models": ["宋PLUS DM-i", "皓影混动", "CS75 PLUS混动"],
    },
    {
        "id": "q16",
        "question": "25万左右的纯电SUV",
        "category": "精确搜索",
        "expected_models": ["Model Y", "小鹏G6"],
    },
    {
        "id": "q17",
        "question": "10万以下的燃油车",
        "category": "精确搜索",
        "expected_models": ["逸达", "海鸥", "博越L"],
    },
    # ---- 模糊搜索（需语义检索 RAG 兜底）----
    {
        "id": "q05",
        "question": "适合新手开的车有哪些？",
        "category": "模糊搜索",
        "expected_keywords": ["容易", "轻快", "代步", "实用"],
    },
    {
        "id": "q06",
        "question": "省油的家用SUV，空间要大",
        "category": "模糊搜索",
        "expected_models": ["宋PLUS DM-i", "RAV4荣放双擎", "皓影混动"],
    },
    {
        "id": "q07",
        "question": "长途自驾游开什么车比较舒服？",
        "category": "模糊搜索",
        "expected_models": ["GL8", "理想L6", "零跑C16"],
    },
    {
        "id": "q08",
        "question": "安全性最好的家用车推荐",
        "category": "模糊搜索",
        "expected_models": ["星越L"],
    },
    {
        "id": "q18",
        "question": "省油的家用SUV",
        "category": "模糊搜索",
        "expected_models": ["宋PLUS DM-i", "RAV4荣放双擎", "皓影混动"],
    },
    {
        "id": "q19",
        "question": "适合长途自驾的车",
        "category": "模糊搜索",
        "expected_models": ["GL8", "理想L6", "零跑C16"],
    },
    {
        "id": "q20",
        "question": "空间最大的家用车",
        "category": "模糊搜索",
        "expected_models": ["GL8", "理想L6", "宋L DM-i"],
    },
    {
        "id": "q21",
        "question": "适合新手开的车",
        "category": "模糊搜索",
        "expected_models": ["海鸥", "逸达", "轩逸"],
    },
    {
        "id": "q22",
        "question": "动力强加速快的车",
        "category": "模糊搜索",
        "expected_models": ["理想L6", "Model Y", "深蓝SL03纯电"],
    },
    # ---- 对比分析 ----
    {
        "id": "q09",
        "question": "对比比亚迪宋PLUS和本田CR-V，哪个更适合家用？",
        "category": "对比分析",
        "expected_models": ["宋PLUS DM-i", "CR-V"],
    },
    {
        "id": "q10",
        "question": "特斯拉Model 3和比亚迪秦PLUS EV续航对比",
        "category": "对比分析",
        "expected_models": ["Model Y", "秦PLUS EV"],
    },
    {
        "id": "q23",
        "question": "对比比亚迪宋PLUS和哈弗H6",
        "category": "对比分析",
        "expected_models": ["宋PLUS DM-i", "哈弗H6"],
    },
    {
        "id": "q24",
        "question": "对比理想L6和零跑C16",
        "category": "对比分析",
        "expected_models": ["理想L6", "零跑C16"],
    },
    {
        "id": "q25",
        "question": "对比Model Y和小鹏G6",
        "category": "对比分析",
        "expected_models": ["Model Y", "小鹏G6"],
    },
    {
        "id": "q26",
        "question": "对比秦PLUS EV和AION S Plus",
        "category": "对比分析",
        "expected_models": ["秦PLUS EV", "AION S Plus"],
    },
    # ---- 用途导向 ----
    {
        "id": "q11",
        "question": "家里有两个小孩，预算20万，推荐什么车？",
        "category": "用途导向",
        "expected_keywords": ["家庭", "空间", "安全"],
    },
    {
        "id": "q12",
        "question": "上下班通勤单程30公里，不想加油，买什么车？",
        "category": "用途导向",
        "expected_keywords": ["通勤", "纯电", "续航", "充电"],
    },
    {
        "id": "q27",
        "question": "二胎家庭7座车推荐",
        "category": "用途导向",
        "expected_models": ["GL8", "零跑C16"],
    },
    {
        "id": "q28",
        "question": "上下班通勤不想加油",
        "category": "用途导向",
        "expected_models": ["秦PLUS EV", "海鸥", "深蓝SL03纯电"],
    },
    {
        "id": "q29",
        "question": "商务接待用车推荐",
        "category": "用途导向",
        "expected_models": ["GL8", "红旗H5", "凯美瑞双擎"],
    },
    {
        "id": "q30",
        "question": "预算20万左右家庭SUV",
        "category": "用途导向",
        "expected_models": ["宋PLUS DM-i", "CR-V", "RAV4荣放双擎"],
    },
    # ---- 边界测试 ----
    {
        "id": "q31",
        "question": "30万以内的豪华品牌车",
        "category": "边界测试",
        "expected_result": "empty",
    },
    {
        "id": "q32",
        "question": "会飞的车",
        "category": "边界测试",
        "expected_result": "empty",
    },
]

# ==========================================================================
# 辅助函数
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