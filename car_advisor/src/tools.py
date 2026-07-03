"""购车智能体工具集。

提供本地车型搜索、RAG 语义检索、车型对比和在线搜索
四个工具，供 LangGraph Agent 在推理过程中调用。

工具选择指南：
- 用户给出明确预算/车型 → search_local_cars（结构化匹配 + RAG 兜底）
- 用户想深入了解某款车 → rag_search（语义检索口碑/卖点/参数）
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
    """在线搜索汽车最新资讯（通过 Tavily 实时搜索）。

    用于获取本地数据库中没有的实时信息：
    - 最新优惠政策 / 降价信息
    - 车主真实口碑和评测
    - 行业新闻 / 新车型发布
    - 保值率 / 销量数据

    适用场景：
    - 用户询问最新行情："XX车现在有什么优惠？"
    - 查询口碑："XX车车主评价怎么样？"
    - 了解市场动态："2025年最值得买的SUV"

    Args:
        query: 搜索关键词。例如 "比亚迪宋PLUS 2025款 最新优惠 口碑"。

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


@tool
def rag_search(query: str) -> str:
    """语义搜索车型文档库，获取深度信息。

    当用户的需求比较模糊、或者结构化搜索匹配不到结果时，
    使用此工具基于语义相似度从车型文档中检索相关内容。

    覆盖的信息包括：
    - 车型核心卖点和详细介绍（~260字/车）
    - 用户口碑（优缺点）
    - 适用场景推荐
    - 详细参数列表

    适用场景：
    - 用户需求模糊：如"我想要一辆舒服的车"、"适合新手开的车"
    - 结构化搜索返回太少结果时自动兜底
    - 需要了解车型口碑和实际使用体验

    Args:
        query: 自然语言查询。例如 "省油的家用SUV推荐"、"长途开什么车舒服"。

    Returns:
        JSON 字符串，包含相关文档片段列表。
        每个结果包含 content（文本内容）、title（车型名）、
        category（类别）、energy_type（能源类型）、
        price_range（价格区间）、section（文档类型）等字段。
    """
    try:
        from .rag.retriever import _get_store
    except ImportError:
        return json.dumps(
            {"error": "RAG 模块未加载，请先运行 data/build_index.py 构建索引"},
            ensure_ascii=False,
        )

    store = _get_store()
    docs = store.search(query, k=5)

    results = []
    for doc in docs:
        results.append({
            "content": doc.page_content[:400],
            "title": doc.metadata.get("title", ""),
            "category": doc.metadata.get("category", ""),
            "energy_type": doc.metadata.get("energy_type", ""),
            "price_range": doc.metadata.get("price_range", ""),
            "section": doc.metadata.get("section", ""),
            "source": doc.metadata.get("source", ""),
        })

    logger.info("rag_search: '%s' → %d results", query[:60], len(results))
    return json.dumps(results, ensure_ascii=False, indent=2)


# ==========================================================================
# 内部：去重合并
# ==========================================================================


def _merge_and_dedupe(
    structured: list[dict],
    rag_results: list[dict],
    max_items: int = 5,
) -> list[dict]:
    """合并结构化搜索和 RAG 结果，按车型名去重。

    Args:
        structured: 结构化搜索结果。
        rag_results: RAG 语义搜索结果。
        max_items:   返回最大条数。

    Returns:
        去重合并后的车型列表。
    """
    seen: set[str] = set()
    merged: list[dict] = []

    def _key(item: dict) -> str:
        return (item.get("title", "") or item.get("name", "")).lower()

    # 先放结构化结果
    for item in structured:
        k = _key(item)
        if k and k not in seen:
            seen.add(k)
            merged.append(item)

    # 再补充 RAG 结果
    for item in rag_results:
        k = _key(item)
        if k and k not in seen:
            seen.add(k)
            # 转换为简化格式
            merged.append({
                "name": item.get("title", ""),
                "category": item.get("category", ""),
                "energy_type": item.get("energy_type", ""),
                "price_range": item.get("price_range", ""),
                "section": item.get("section", ""),
                "content": item.get("content", "")[:200],
                "source": "语义检索",
            })

    return merged[:max_items]


# ==========================================================================
# 修改 search_local_cars：结果不足时自动 RAG 兜底
# ==========================================================================

# 保存原始的 search_local_cars 逻辑，用于重新包装
_search_local_cars_original = search_local_cars


@tool
def search_local_cars(budget: str, vehicle_type: str = "") -> str:  # type: ignore[no-redef]
    """搜索符合预算和车型偏好的车辆（结构化匹配 + 语义兜底）。

    优先使用价格区间和车型类型进行精确筛选。当匹配结果少于
    3 条时，自动调用语义搜索补充相关车型，确保用户总能获得
    足够的选择参考。

    适用场景：
    - 用户明确给出预算和/或车型偏好
    - 如 "15-20万SUV"、"10万以内的轿车"
    - 这是大多数场景的首选工具

    Args:
        budget:       预算区间。例如 "15-20万"、"20万左右"、"10万以内"。
        vehicle_type: 车型偏好。例如 "SUV"、"轿车"、"MPV"。留空不限。

    Returns:
        JSON 字符串，包含匹配的车型列表（≥3条时有兜底）。
        每项含 name、price_range、type、fuel、pros、cons 等。
    """
    # 先用结构化搜索
    raw = _search_local_cars_original.invoke({"budget": budget, "vehicle_type": vehicle_type})
    results = json.loads(raw)

    # 结果 >= 3 条，直接返回
    if len(results) >= 3:
        logger.info("search_local_cars: %d results (no RAG needed)", len(results))
        return json.dumps({"results": results, "source": "结构化搜索"}, ensure_ascii=False, indent=2)

    # 结果不足，RAG 兜底
    logger.info("search_local_cars: only %d results, falling back to RAG", len(results))

    # 构建语义搜索查询
    rag_query_parts = []
    if vehicle_type:
        rag_query_parts.append(vehicle_type)
    rag_query_parts.append(f"预算{budget}")
    rag_query = " ".join(rag_query_parts)

    try:
        rag_raw = rag_search.invoke({"query": rag_query})
        rag_results = json.loads(rag_raw)
    except Exception:
        logger.warning("RAG fallback failed, returning structured results only")
        return json.dumps(
            {"results": results, "source": "结构化搜索（结果不足，语义搜索不可用）"},
            ensure_ascii=False, indent=2,
        )

    merged = _merge_and_dedupe(results, rag_results)
    logger.info("search_local_cars: merged → %d results", len(merged))

    return json.dumps(
        {"results": merged, "source": f"结构化({len(results)}条) + 语义搜索({len(rag_results)}条)"},
        ensure_ascii=False, indent=2,
    )


# ==========================================================================
# 工具列表
# ==========================================================================

tools = [search_local_cars, compare_cars, search_online, rag_search]
