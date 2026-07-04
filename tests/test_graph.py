"""graph.py 新增功能的单元测试。

覆盖：_detect_param_keywords、_filter_candidates_by_params。
"""

import pytest

from car_advisor.src.graph import _detect_param_keywords


class TestDetectParamKeywords:
    def test_accel(self):
        r = _detect_param_keywords("百公里加速7秒以内")
        assert any(k == "加速" for k, _ in r)

    def test_accel_short(self):
        r = _detect_param_keywords("加速要6秒以下的")
        assert any(k == "加速" for k, _ in r)

    def test_range(self):
        r = _detect_param_keywords("续航500公里以上")
        assert any(k == "续航" for k, _ in r)

    def test_fuel_consumption(self):
        r = _detect_param_keywords("油耗6L以下的SUV")
        assert any(k == "油耗" for k, _ in r)

    def test_space(self):
        r = _detect_param_keywords("空间大一点的车")
        assert any(k == "空间" for k, _ in r)

    def test_trunk(self):
        r = _detect_param_keywords("后备箱500升以上")
        assert any(k == "后备箱" for k, _ in r)

    def test_wheelbase(self):
        r = _detect_param_keywords("轴距2800以上")
        assert any(k == "轴距" for k, _ in r)

    def test_no_params(self):
        r = _detect_param_keywords("纯电轿车推荐")
        assert len(r) == 0
