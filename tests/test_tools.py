"""tools.py 新增功能的单元测试。

覆盖：_parse_budget/None、_match_fuel、_normalize_car_type、
search_local_cars 多维筛选、_EXCLUDE_BRAND_MAP。
"""

import json

import pytest

from car_advisor.src.tools import (
    _normalize_car_type,
    _match_fuel,
    _match_type,
    _parse_budget,
    _extract_fuel_number,
    _extract_range_km,
    _EXCLUDE_BRAND_MAP,
    search_local_cars,
)


class TestParseBudget:
    def test_valid_formats(self):
        assert _parse_budget("15万") == (13.5, 16.5)
        assert _parse_budget("15-20万") == (15.0, 20.0)
        assert _parse_budget("20万左右") == (18.0, 22.0)
        assert _parse_budget("10万以内") == (0, 10.0)
        assert _parse_budget("30万以上") == (30.0, 999)

    @pytest.mark.parametrize("text", ["随便", "合适的价格", "", "abc", "不知道"])
    def test_invalid_returns_none(self, text):
        assert _parse_budget(text) is None


class TestNormalizeCarType:
    def test_synonyms(self):
        assert _normalize_car_type("越野车") == "suv"
        assert _normalize_car_type("吉普") == "suv"
        assert _normalize_car_type("房车") == "轿车"
        assert _normalize_car_type("小轿车") == "轿车"
        assert _normalize_car_type("商务车") == "mpv"
        assert _normalize_car_type("七座车") == "mpv"
        assert _normalize_car_type("瓦罐") == "旅行车"
        assert _normalize_car_type("sedan") == "轿车"
        assert _normalize_car_type("wagon") == "旅行车"

    def test_unknown_preserves_lowercase(self):
        assert _normalize_car_type("直升机") == "直升机"


class TestMatchType:
    def test_synonym_matching(self):
        assert _match_type("SUV", "越野车") is True
        assert _match_type("轿车", "房车") is True
        assert _match_type("MPV", "商务车") is True
        assert _match_type("SUV", "吉普") is True

    def test_no_match(self):
        assert _match_type("轿车", "suv") is False
        assert _match_type("SUV", "皮卡") is False

    def test_empty_target_matches_all(self):
        assert _match_type("SUV", "") is True


class TestMatchFuel:
    def test_exact(self):
        assert _match_fuel("燃油", "燃油") is True

    def test_hybrid_covers_both(self):
        assert _match_fuel("油电混动(HEV)", "混动") is True
        assert _match_fuel("插电混动(PHEV)", "混动") is True

    def test_ev(self):
        assert _match_fuel("纯电动(BEV)", "纯电") is True
        assert _match_fuel("纯电动(BEV)", "电动") is True

    def test_fuel(self):
        assert _match_fuel("燃油", "汽油") is True

    def test_range_extender(self):
        assert _match_fuel("增程式", "增程") is True

    def test_no_match(self):
        assert _match_fuel("燃油", "纯电") is False


class TestExtractFuelNumber:
    def test_standard(self):
        assert _extract_fuel_number("WLTC 7.3L/100km") == 7.3
        assert _extract_fuel_number("亏电油耗4.5L/100km，纯电续航110km") == 4.5

    def test_no_number(self):
        assert _extract_fuel_number("CLTC 500km，电耗12.5kWh/100km") is None


class TestExtractRangeKm:
    def test_standard(self):
        assert _extract_range_km("纯电续航110km") == 110
        assert _extract_range_km("CLTC 510km") == 510

    def test_no_range(self):
        assert _extract_range_km("WLTC 7.3L/100km") is None


class TestExcludeBrandMap:
    def test_japanese(self):
        assert "本田" in _EXCLUDE_BRAND_MAP["日系"]
        assert "丰田" in _EXCLUDE_BRAND_MAP["日系"]
        assert "日产" in _EXCLUDE_BRAND_MAP["日系"]

    def test_german(self):
        assert "大众" in _EXCLUDE_BRAND_MAP["德系"]

    def test_chinese(self):
        assert "比亚迪" in _EXCLUDE_BRAND_MAP["国产"]


class TestSearchLocalCars:
    """集成测试：不需要 API key，只测本地数据筛选。"""

    def test_fuel_filter(self):
        r = json.loads(search_local_cars.invoke({"budget": "10-25万", "fuel": "混动"}))
        assert len(r) > 0
        for car in r:
            assert "混动" in car["fuel"] or "HEV" in car["fuel"]

    def test_max_fuel_consumption(self):
        r = json.loads(search_local_cars.invoke({"budget": "10-25万", "max_fuel_consumption": 5.0}))
        assert len(r) > 0
        for car in r:
            fc = _extract_fuel_number(car.get("fuel_economy", ""))
            assert fc is not None
            assert fc <= 5.0

    def test_min_range(self):
        r = json.loads(search_local_cars.invoke({"budget": "10-25万", "min_range": 200}))
        for car in r:
            rng = _extract_range_km(car.get("fuel_economy", ""))
            assert rng is not None
            assert rng >= 200

    def test_brand_filter(self):
        r = json.loads(search_local_cars.invoke({"budget": "10-25万", "brand": "比亚迪"}))
        for car in r:
            assert car["brand"] == "比亚迪"

    def test_exclude_brand(self):
        r = json.loads(search_local_cars.invoke({"budget": "10-25万", "exclude_brand": "日系"}))
        jp = _EXCLUDE_BRAND_MAP["日系"]
        for car in r:
            assert car["brand"] not in jp

    def test_budget_not_parsed(self):
        r = json.loads(search_local_cars.invoke({"budget": "随便"}))
        assert "error" in r
        assert r["error"] == "budget_not_parsed"

    def test_no_results(self):
        r = json.loads(search_local_cars.invoke({"budget": "100万以上"}))
        assert len(r) == 0

    def test_all_filter_combined(self):
        r = json.loads(search_local_cars.invoke({
            "budget": "10-20万", "fuel": "混动",
            "max_fuel_consumption": 6.0, "brand": "比亚迪",
        }))
        for car in r:
            assert car["brand"] == "比亚迪"
            assert "混动" in car["fuel"] or "HEV" in car["fuel"]
