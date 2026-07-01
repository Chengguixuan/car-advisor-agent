"""购车智能体工具集。

提供本地车型搜索、对比和在线搜索（模拟）三个工具，
供 LangGraph Agent 在推理过程中调用。
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# 车型数据库路径
_CAR_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "car_data.json"

# ==========================================================================
# 内部辅助函数
# ==========================================================================


def load_car_db() -> list[dict]:
    """从 data/car_data.json 加载车型数据库。

    Returns:
        车型字典列表。如果文件不存在或格式错误，返回空列表。
    """
    try:
        with open(_CAR_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error("加载车型数据失败: %s", e)
        return []


def _parse_budget(budget_str: str) -> tuple[float, float]:
    """解析预算字符串，返回 (最低价, 最高价) 的数值区间。

    支持的格式：
        - "15万"       → (13.5, 16.5) 含 10% 容差
        - "15-20万"    → (15, 20)
        - "20万左右"   → (18, 22) 含 10% 容差
        - "10万以内"   → (0, 10)
        - "30万以上"   → (30, 999)

    Args:
        budget_str: 用户输入的预算字符串。

    Returns:
        (min_price, max_price) 元组，单位为万元。
    """
    budget_str = budget_str.strip()

    # "X-Y万" 格式
    range_match = re.match(r"(\d+)\s*[-~到至]\s*(\d+)\s*万?", budget_str)
    if range_match:
        return (float(range_match.group(1)), float(range_match.group(2)))

    # "X万以内" / "X万以下"
    under_match = re.match(r"(\d+)\s*万?\s*(以内|以下|内)", budget_str)
    if under_match:
        return (0, float(under_match.group(1)))

    # "X万以上" / "X万以上"
    over_match = re.match(r"(\d+)\s*万?\s*(以上|以上)", budget_str)
    if over_match:
        return (float(over_match.group(1)), 999)

    # "X万左右" / "X万" / "X万多"
    approx_match = re.match(r"(\d+)\s*万?", budget_str)
    if approx_match:
        val = float(approx_match.group(1))
        margin = val * 0.1  # ±10%
        return (round(val - margin, 1), round(val + margin, 1))

    # 无法解析，返回全范围
    return (0, 999)


def _match_type(car_type: str, target: str) -> bool:
    """检查车型是否匹配目标类型。

    Args:
        car_type: 车型类型（SUV / 轿车 / MPV）。
        target:   用户想要的类型。

    Returns:
        是否匹配。
    """
    if not target:
        return True
    target = target.strip().lower()
    # 模糊匹配
    return target in car_type.lower() or car_type.lower() in target


# ==========================================================================
# LangChain 工具
# ==========================================================================


@tool
def search_local_cars(budget: str, vehicle_type: str = "") -> str:
    """从本地车型数据库中搜索符合条件的车型。

    根据用户提供的预算区间和车型偏好筛选数据库中的车型，
    返回匹配车型的 JSON 字符串列表。

    Args:
        budget:        预算区间。例如 "15-20万"、"20万左右"、"10万以内"、"30万以上"。
        vehicle_type:  车型偏好。例如 "SUV"、"轿车"、"MPV"。留空表示不限。

    Returns:
        JSON 字符串，包含匹配的车型列表。如果没有匹配的车型则返回空数组。
        每个车型包含 name、price_range、type、fuel、pros、cons 等字段。
    """
    cars = load_car_db()
    if not cars:
        return json.dumps([], ensure_ascii=False)

    min_price, max_price = _parse_budget(budget)
    results: list[dict] = []

    for car in cars:
        price = car.get("price", 0)
        car_type = car.get("type", "")

        # 价格匹配
        if not (min_price <= price <= max_price):
            continue

        # 车型匹配
        if vehicle_type and not _match_type(car_type, vehicle_type):
            continue

        # 返回精简版字段（不含 engine、power 等详细参数，供列表展示）
        results.append({
            "name": car.get("name", ""),
            "brand": car.get("brand", ""),
            "price_range": car.get("price_range", ""),
            "type": car.get("type", ""),
            "fuel": car.get("fuel", ""),
            "fuel_economy": car.get("fuel_economy", ""),
            "pros": car.get("pros", []),
            "cons": car.get("cons", []),
        })

    return json.dumps(results, ensure_ascii=False, indent=2)


@tool
def compare_cars(car_names: str) -> str:
    """对比指定车型的详细参数。

    接收逗号分隔的车型名称，从数据库中查找对应车型并返回
    包含价格、动力、油耗、优缺点等完整信息的对比结果。

    Args:
        car_names: 车型名称，逗号分隔。例如 "宋PLUS, CR-V"。

    Returns:
        JSON 字符串，包含匹配车型的完整信息列表。
        未匹配到的车型名称会在 unmatched 字段中列出。
    """
    cars = load_car_db()
    if not cars:
        return json.dumps({"matched": [], "unmatched": []}, ensure_ascii=False)

    names = [n.strip() for n in car_names.split(",") if n.strip()]
    matched: list[dict] = []
    matched_names: set[str] = set()

    for query in names:
        found = False
        for car in cars:
            # 大小写不敏感的模糊匹配（名称或品牌+车型）
            full_name = car.get("name", "")
            brand = car.get("brand", "")
            if query.lower() in full_name.lower() or query.lower() in brand.lower():
                matched.append(car)
                matched_names.add(query)
                found = True
                break
        # 如果精确匹配失败，尝试更宽松的匹配
        if not found:
            for car in cars:
                full_name = car.get("name", "")
                # 检查每个查询词是否都在车名中出现
                query_chars = query.replace(" ", "").lower()
                name_chars = full_name.replace(" ", "").lower()
                if all(c in name_chars for c in query_chars if c.isalpha()):
                    matched.append(car)
                    matched_names.add(query)
                    found = True
                    break

    unmatched = [n for n in names if n not in matched_names]

    result = {
        "matched": matched,
        "unmatched": unmatched,
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def search_online(query: str) -> str:
    """在线搜索汽车相关信息（当前为模拟数据）。

    用于获取本地数据库中没有的最新资讯，如：
    - 最新优惠政策
    - 车主真实口碑
    - 专业评测
    - 保值率数据

    Args:
        query: 搜索关键词。例如 "比亚迪宋PLUS 2025款 最新优惠"。

    Returns:
        JSON 字符串，包含模拟的搜索结果。
    """
    # 模拟搜索结果 — 后续可接入真实的搜索 API
    mock_results = {
        "query": query,
        "source": "模拟在线搜索",
        "disclaimer": "以下为模拟数据，实际购车请以官方渠道为准。",
        "results": [
            {
                "title": f"关于「{query}」的搜索建议",
                "snippet": (
                    "建议您访问以下渠道获取最新信息："
                    "1) 汽车之家 (autohome.com.cn) — 车主口碑和评测；"
                    "2) 懂车帝 (dongchedi.com) — 车型对比和优惠行情；"
                    "3) 品牌官网 — 最新配置和官方指导价；"
                    "4) 4S 店 — 实际落地价和金融方案。"
                ),
                "url": "https://www.autohome.com.cn",
            },
            {
                "title": "注意事项",
                "snippet": (
                    "网络报价仅供参考，实际价格以当地 4S 店报价为准。"
                    "建议多对比 2-3 家经销商的报价，注意区分裸车价和落地价。"
                ),
            },
        ],
    }
    return json.dumps(mock_results, ensure_ascii=False, indent=2)


# ==========================================================================
# 工具列表
# ==========================================================================

tools = [search_local_cars, compare_cars, search_online]
