"""购车智能体工具集。

提供本地车型搜索、车型对比和在线搜索三个工具，
供 LangGraph Agent 在推理过程中调用。

工具选择指南：
- 用户给出明确预算/车型 → search_local_cars（结构化匹配）
- 用户对比多款车 → compare_cars（并排对比）
- 查最新行情/优惠 → search_online（Tavily 实时搜索，失败时模拟）
"""

import json
import logging
import os
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


def _parse_budget(budget_str: str) -> Optional[tuple[float, float]]:
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
        (min_price, max_price) 元组；无法解析时返回 None。
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

    # 无法解析，返回 None 让上层触发追问
    return None


# 车型同义词映射表（用户常用词 → 数据库标准类型）
_CAR_TYPE_SYNONYMS = {
    "suv":    ["suv", "越野车", "城市越野", "吉普", "suv车型", "城市suv", "越野"],
    "轿车":   ["轿车", "小轿车", "三厢车", "家用轿车", "sedan", "房车", "家轿", "小汽车"],
    "mpv":    ["mpv", "商务车", "保姆车", "多用途车", "家用mpv", "七座车", "7座车"],
    "跑车":   ["跑车", "小跑", "轿跑", "coupe"],
    "旅行车": ["旅行车", "瓦罐", "wagon", "休旅车"],
    "皮卡":   ["皮卡", "皮卡丘"],
}


def _normalize_car_type(raw: str) -> str:
    """将用户输入的同义词归一化为数据库中的标准类型名。

    Args:
        raw: 原始车型名称，如 "越野车"、"房车"、"商务车"。

    Returns:
        标准车型名（suv / 轿车 / mpv / 跑车 / 旅行车 / 皮卡）。
        未匹配时返回原字符串的小写形式。
    """
    raw_lower = raw.strip().lower()
    for std_name, synonyms in _CAR_TYPE_SYNONYMS.items():
        if raw_lower in synonyms:
            return std_name
    return raw_lower


def _match_type(car_type: str, target: str) -> bool:
    """检查车型是否匹配目标类型（支持同义词）。

    Args:
        car_type: 车型类型（SUV / 轿车 / MPV 等数据库中的值）。
        target:   用户输入的类型，支持同义词如 "越野车"→SUV、"房车"→轿车。

    Returns:
        是否匹配。
    """
    if not target:
        return True

    # 归一化后比较
    std_target = _normalize_car_type(target)
    std_car = _normalize_car_type(car_type)

    if std_target == std_car:
        return True
    # 兜底：原模糊匹配
    return std_target in std_car or std_car in std_target


# 排除品牌关键词 → 实际品牌名
_EXCLUDE_BRAND_MAP = {
    "日系": ["本田", "丰田", "日产", "马自达", "斯巴鲁", "三菱", "铃木", "雷克萨斯"],
    "德系": ["大众", "奔驰", "宝马", "奥迪", "保时捷"],
    "美系": ["福特", "别克", "雪佛兰", "凯迪拉克", "特斯拉"],
    "国产": ["比亚迪", "吉利", "长安", "哈弗", "红旗", "零跑", "小鹏", "理想", "深蓝", "广汽埃安"],
}


def _match_fuel(car_fuel: str, user_fuel: str) -> bool:
    """检查车型能源类型是否匹配用户需求（支持模糊匹配）。

    用户说"混动" → 匹配 "油电混动(HEV)" 和 "插电混动(PHEV)"
    用户说"纯电" → 匹配 "纯电动(BEV)"
    """
    uf = user_fuel.strip()
    cf = car_fuel.strip().lower()

    if uf == cf:
        return True
    if uf in cf:
        return True
    # 模糊同义
    if uf in ("混动", "油电混合"):
        return "混动" in cf or "hev" in cf
    if uf in ("插混", "插电"):
        return "插电" in cf or "phev" in cf
    if uf in ("纯电", "电动"):
        return "纯电" in cf or "bev" in cf
    if uf in ("燃油", "汽油"):
        return "燃油" in cf
    if uf in ("增程", "增程式"):
        return "增程" in cf
    return False


# ==========================================================================
# LangChain 工具
# ==========================================================================


def _extract_fuel_number(fuel_economy: str) -> Optional[float]:
    """从 fuel_economy 字符串中提取油耗数值，如 '4.5L' → 4.5。"""
    m = re.search(r"(\d+[\.]?\d*)\s*[Ll升]", fuel_economy)
    return float(m.group(1)) if m else None


def _extract_range_km(fuel_economy: str) -> Optional[int]:
    """从 fuel_economy 字符串中提取纯电续航，如 '纯电续航110km' → 110。"""
    m = re.search(r"(?:纯电续航|续航|CLTC)\s*(\d+)\s*(?:km|公里)", fuel_economy)
    return int(m.group(1)) if m else None


@tool
def search_local_cars(
    budget: str,
    vehicle_type: str = "",
    fuel: str = "",
    max_fuel_consumption: Optional[float] = None,
    min_range: Optional[int] = None,
    brand: str = "",
    exclude_brand: str = "",
) -> str:
    """按预算/车型/能源/油耗/续航/品牌等条件筛选车型。

    负责：固定参数筛选（价格、车型、能源、油耗、续航、品牌）。
    不负责：车型对比（用compare_cars）、实时行情/优惠/口碑（用search_online）。
    触发：用户给出明确筛选条件。例如 "15万SUV"、"油耗5L以下"。

    Args:
        budget:              预算区间。例如 "15-20万"、"20万左右"。
        vehicle_type:        车型偏好。例如 "SUV"、"轿车"、"MPV"。留空不限。
        fuel:                能源类型。例如 "燃油"、"混动"、"纯电动"、"插电混动"、"增程式"。
        max_fuel_consumption: 油耗上限（L/100km）。
        min_range:            纯电续航下限（km）。
        brand:                品牌偏好。
        exclude_brand:        排除品牌，如"日系"、"德系"。

    Returns:
        JSON 字符串，包含匹配的车型列表。
    """
    cars = load_car_db()
    if not cars:
        return json.dumps([], ensure_ascii=False)

    parsed = _parse_budget(budget)
    if parsed is None:
        return json.dumps({
            "error": "budget_not_parsed",
            "message": "无法理解你的预算范围，请明确告诉我，比如'15-20万'或'20万左右'。",
        }, ensure_ascii=False)

    # 排除品牌关键词映射
    exclude_keywords = _EXCLUDE_BRAND_MAP.get(exclude_brand.strip(), [exclude_brand.strip()]) if exclude_brand.strip() else []

    min_price, max_price = parsed
    results: list[dict] = []

    for car in cars:
        price = car.get("price", 0)
        car_type = car.get("type", "")
        car_brand = car.get("brand", "")
        car_fuel = car.get("fuel", "")
        fuel_economy = car.get("fuel_economy", "")

        # 价格匹配
        if not (min_price <= price <= max_price):
            continue

        # 车型匹配
        if vehicle_type and not _match_type(car_type, vehicle_type):
            continue

        # 能源类型匹配（模糊匹配，如用户说"混动"匹配"油电混动"和"插电混动"）
        if fuel and not _match_fuel(car_fuel, fuel):
            continue

        # 油耗上限
        if max_fuel_consumption is not None:
            car_fc = _extract_fuel_number(fuel_economy)
            if car_fc is None or car_fc > max_fuel_consumption:
                continue

        # 纯电续航下限
        if min_range is not None:
            car_range = _extract_range_km(fuel_economy)
            if car_range is None or car_range < min_range:
                continue

        # 品牌偏好
        if brand and brand.strip() not in car_brand:
            continue

        # 排除品牌
        if exclude_keywords:
            if any(kw in car_brand for kw in exclude_keywords):
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
def compare_cars(car_names: str, exclude_brand: str = "") -> str:
    """对比两款或多款车型的参数（价格/油耗/续航/尺寸/动力）。

    负责：多车型并排对比。
    不负责：按条件筛选（用search_local_cars）、实时行情（用search_online）。
    触发：用户明确说"A和B哪个好"、"对比A和B"、"比较一下"。

    即使车型属于历史排除范围，仍正常返回，并在note中提醒。

    Args:
        car_names:    车型名称，逗号分隔。例如 "宋PLUS, CR-V"。
        exclude_brand: 历史排除的品牌（如"日系"），仅用于提示。

    Returns:
        JSON 字符串，含 matched / unmatched / note 字段。
    """
    cars = load_car_db()
    if not cars:
        return json.dumps({"matched": [], "unmatched": [], "note": ""}, ensure_ascii=False)

    exclude_keywords = _EXCLUDE_BRAND_MAP.get(exclude_brand.strip(), [exclude_brand.strip()]) if exclude_brand.strip() else []
    names = [n.strip() for n in car_names.split(",") if n.strip()]
    matched: list[dict] = []
    matched_names: set[str] = set()
    excluded_matched: list[str] = []

    for query in names:
        found = False
        for car in cars:
            full_name = car.get("name", "")
            brand = car.get("brand", "")
            if query.lower() in full_name.lower() or query.lower() in brand.lower():
                matched.append(car)
                matched_names.add(query)
                if exclude_keywords and any(kw in brand for kw in exclude_keywords):
                    excluded_matched.append(car.get("name", ""))
                found = True
                break
        if not found:
            for car in cars:
                full_name = car.get("name", "")
                query_chars = query.replace(" ", "").lower()
                name_chars = full_name.replace(" ", "").lower()
                if all(c in name_chars for c in query_chars if c.isalpha()):
                    matched.append(car)
                    matched_names.add(query)
                    if exclude_keywords and any(kw in car.get("brand", "") for kw in exclude_keywords):
                        excluded_matched.append(car.get("name", ""))
                    found = True
                    break

    unmatched = [n for n in names if n not in matched_names]

    # 兜底：本地找不到的车型，尝试在线搜索补齐数据
    if unmatched:
        for name in unmatched[:]:
            try:
                online_raw = search_online.invoke({"query": f"{name} 车型参数 价格 油耗 优缺点"})
                online_data = json.loads(online_raw)
                answer = online_data.get("answer", "")
                if answer:
                    matched.append({
                        "name": name,
                        "brand": "",
                        "price_range": "在线查询",
                        "type": "",
                        "fuel": "",
                        "fuel_economy": "",
                        "pros": [],
                        "cons": [],
                        "source": "在线搜索",
                        "note": f"以下信息来自网络搜索：{answer[:200]}",
                    })
                    matched_names.add(name)
                    unmatched.remove(name)
            except Exception:
                pass

    note = ""
    if excluded_matched:
        note = f"你之前排除了{exclude_brand}，以下包含{'、'.join(excluded_matched)}（{exclude_brand}），请重新评估是否接受"

    result = {"matched": matched, "unmatched": unmatched, "note": note}
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def search_online(query: str) -> str:
    """联网搜索最新实时汽车信息（Tavily）。**这是唯一能获取实时数据的工具。**

    **重要**：用户问优惠/降价/多少钱/口碑/评测/销量/行情/新闻时，
    **必须调用此工具**，不能用训练数据或本地数据库回答。

    返回：最新优惠政策、车主口碑、行业新闻、保值率/销量数据。

    Args:
        query: 搜索关键词。例如 "比亚迪宋PLUS 2025款 最新优惠"。
    Returns:
        JSON 字符串，包含 Tavily 实时搜索结果。
    """
    try:
        from langchain_tavily import TavilySearch

        api_key = os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            logger.warning("TAVILY_API_KEY not set, falling back to mock")
            return json.dumps(_mock_search(query), ensure_ascii=False, indent=2)

        tavily = TavilySearch(
            api_key=api_key,
            max_results=5,
            search_depth="basic",
            include_answer=True,
        )
        results = tavily.invoke(query)

        formatted = {
            "query": query,
            "source": "Tavily 实时搜索",
            "answer": results.get("answer", ""),
            "results": [
                {
                    "title": r.get("title", ""),
                    "snippet": (r.get("content", "") or "")[:300],
                    "url": r.get("url", ""),
                }
                for r in results.get("results", [])[:5]
            ],
        }
        logger.info("search_online (Tavily): '%s' → %d results", query[:60], len(formatted["results"]))
        return json.dumps(formatted, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.warning("Tavily search failed: %s, falling back to mock", e)
        return json.dumps(_mock_search(query), ensure_ascii=False, indent=2)


def _mock_search(query: str) -> dict:
    """Tavily 不可用时的兜底结果。"""
    return {
        "query": query,
        "source": "模拟在线搜索（Tavily 不可用）",
        "disclaimer": "以下为模拟数据，设置 TAVILY_API_KEY 可启用实时搜索。",
        "results": [
            {
                "title": f"关于「{query}」的搜索建议",
                "snippet": (
                    "建议访问以下渠道获取最新信息："
                    "1) 汽车之家 (autohome.com.cn) — 车主口碑和评测；"
                    "2) 懂车帝 (dongchedi.com) — 车型对比和优惠行情；"
                    "3) 品牌官网 — 最新配置和官方指导价；"
                    "4) 4S 店 — 实际落地价和金融方案。"
                ),
                "url": "https://www.autohome.com.cn",
            },
        ],
    }


# ==========================================================================
# 工具列表
# ==========================================================================

tools = [search_local_cars, compare_cars, search_online]
