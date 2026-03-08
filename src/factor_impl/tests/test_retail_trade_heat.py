import numpy as np
import pandas as pd
import pytest

import importlib
import importlib.util
import pathlib
import sys
import types

# 绕过 factors/__init__.py（其中导入了尚未实现的因子模块）。
# 先将 factors 包替换为空壳，再逐个加载 base 和目标模块。
_factors_dir = pathlib.Path(__file__).resolve().parent.parent / "factors"

# 1. 用空壳替换 factors 包，阻止原始 __init__.py 执行
_pkg = types.ModuleType("factors")
_pkg.__path__ = [str(_factors_dir)]
sys.modules["factors"] = _pkg

# 2. 加载 factors.base
_base_spec = importlib.util.spec_from_file_location("factors.base", _factors_dir / "base.py")
_base_mod = importlib.util.module_from_spec(_base_spec)
sys.modules["factors.base"] = _base_mod
_base_spec.loader.exec_module(_base_mod)

# 3. 加载 factors.retail_trade_heat
_rth_spec = importlib.util.spec_from_file_location(
    "factors.retail_trade_heat", _factors_dir / "retail_trade_heat.py"
)
_rth_mod = importlib.util.module_from_spec(_rth_spec)
sys.modules["factors.retail_trade_heat"] = _rth_mod
_rth_spec.loader.exec_module(_rth_mod)

RetailTradeHeatFactor = _rth_mod.RetailTradeHeatFactor


@pytest.fixture
def factor():
    return RetailTradeHeatFactor()


# --------------------------------------------------------------------------- #
# 基本元信息
# --------------------------------------------------------------------------- #

class TestMetadata:
    def test_name(self, factor):
        assert factor.name == "TURN_RETAIL"

    def test_category(self, factor):
        assert factor.category == "资金流向"

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "TURN_RETAIL"
        assert "description" in meta


# --------------------------------------------------------------------------- #
# 手工验算
# --------------------------------------------------------------------------- #

class TestHandCalculation:
    """用极简数据手工验证公式正确性。"""

    def test_constant_inputs(self, factor):
        """常数输入：每天 inflow=100, outflow=100, cap=1000 => daily=0.2，均值=0.2"""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        stocks = ["A"]

        inflow = pd.DataFrame(100.0, index=dates, columns=stocks)
        outflow = pd.DataFrame(100.0, index=dates, columns=stocks)
        cap = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(inflow_retail=inflow, outflow_retail=outflow, cap=cap, m=3)

        # 每天 daily_turnover = 200/1000 = 0.2，滚动均值始终 0.2
        expected = pd.DataFrame(0.2, index=dates, columns=stocks)
        pd.testing.assert_frame_equal(result, expected)

    def test_varying_inputs(self, factor):
        """变化输入，手工计算滚动均值。"""
        dates = pd.date_range("2024-01-01", periods=4, freq="B")
        stocks = ["A"]

        inflow = pd.DataFrame([100, 200, 300, 400], index=dates, columns=stocks, dtype=float)
        outflow = pd.DataFrame([0, 0, 0, 0], index=dates, columns=stocks, dtype=float)
        cap = pd.DataFrame([1000, 1000, 1000, 1000], index=dates, columns=stocks, dtype=float)

        result = factor.compute(inflow_retail=inflow, outflow_retail=outflow, cap=cap, m=3)

        # daily_turnover: [0.1, 0.2, 0.3, 0.4]
        # rolling(3, min_periods=1):
        #   day0: mean([0.1])         = 0.1
        #   day1: mean([0.1, 0.2])    = 0.15
        #   day2: mean([0.1, 0.2, 0.3]) = 0.2
        #   day3: mean([0.2, 0.3, 0.4]) = 0.3
        expected_vals = [0.1, 0.15, 0.2, 0.3]
        np.testing.assert_allclose(result["A"].values, expected_vals, atol=1e-12)

    def test_multiple_stocks(self, factor):
        """多只股票并行计算，互不干扰。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        stocks = ["A", "B"]

        inflow = pd.DataFrame(
            {"A": [100.0, 200.0, 300.0], "B": [10.0, 20.0, 30.0]}, index=dates
        )
        outflow = pd.DataFrame(0.0, index=dates, columns=stocks)
        cap = pd.DataFrame({"A": [1000.0] * 3, "B": [100.0] * 3}, index=dates)

        result = factor.compute(inflow_retail=inflow, outflow_retail=outflow, cap=cap, m=2)

        # A daily: [0.1, 0.2, 0.3] => rolling(2): [0.1, 0.15, 0.25]
        # B daily: [0.1, 0.2, 0.3] => rolling(2): [0.1, 0.15, 0.25]
        np.testing.assert_allclose(result["A"].values, [0.1, 0.15, 0.25], atol=1e-12)
        np.testing.assert_allclose(result["B"].values, [0.1, 0.15, 0.25], atol=1e-12)


# --------------------------------------------------------------------------- #
# 边界情况
# --------------------------------------------------------------------------- #

class TestEdgeCases:
    def test_zero_cap_produces_nan_or_inf(self, factor):
        """市值为零时，除法产生 inf/NaN，不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        stocks = ["A"]

        inflow = pd.DataFrame(100.0, index=dates, columns=stocks)
        outflow = pd.DataFrame(100.0, index=dates, columns=stocks)
        cap = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(inflow_retail=inflow, outflow_retail=outflow, cap=cap, m=2)

        assert result.shape == (3, 1)
        # 200/0 在 pandas 中产生 inf，rolling mean(inf) 仍为 inf；
        # 但 0/0 产生 NaN。此处 inflow+outflow=200>0，所以应为 inf。
        # 实际上 pandas 对 float 除以 0 的行为：正数/0 -> inf, 0/0 -> NaN
        # 这里 200.0/0.0 -> inf, rolling mean of inf -> NaN (pandas 行为)
        # 关键是不抛异常，结果中每个值要么 inf 要么 NaN
        assert np.all(np.isinf(result.values) | np.isnan(result.values))

    def test_nan_propagation(self, factor):
        """输入含 NaN 时，rolling mean(min_periods=1) 会跳过 NaN 计算均值。"""
        dates = pd.date_range("2024-01-01", periods=4, freq="B")
        stocks = ["A"]

        inflow = pd.DataFrame([100.0, np.nan, 100.0, 100.0], index=dates, columns=stocks)
        outflow = pd.DataFrame(100.0, index=dates, columns=stocks)
        cap = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(inflow_retail=inflow, outflow_retail=outflow, cap=cap, m=2)

        # daily_turnover: [0.2, NaN, 0.2, 0.2]
        # rolling(2, min_periods=1).mean() 跳过 NaN：
        #   day0: mean([0.2])       = 0.2
        #   day1: mean([0.2, NaN])  = 0.2  (NaN 被跳过，仅 1 个有效值)
        #   day2: mean([NaN, 0.2])  = 0.2  (同上)
        #   day3: mean([0.2, 0.2])  = 0.2
        expected = pd.DataFrame([0.2, 0.2, 0.2, 0.2], index=dates, columns=stocks)
        pd.testing.assert_frame_equal(result, expected)

    def test_m_equals_one(self, factor):
        """m=1 时，结果等于日度换手率本身。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        stocks = ["A"]

        inflow = pd.DataFrame([100.0, 200.0, 300.0], index=dates, columns=stocks)
        outflow = pd.DataFrame(0.0, index=dates, columns=stocks)
        cap = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(inflow_retail=inflow, outflow_retail=outflow, cap=cap, m=1)

        expected = pd.DataFrame([0.1, 0.2, 0.3], index=dates, columns=stocks)
        pd.testing.assert_frame_equal(result, expected)


# --------------------------------------------------------------------------- #
# 输出形状与类型
# --------------------------------------------------------------------------- #

class TestOutputShape:
    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        stocks = ["A", "B", "C"]

        inflow = pd.DataFrame(1.0, index=dates, columns=stocks)
        outflow = pd.DataFrame(1.0, index=dates, columns=stocks)
        cap = pd.DataFrame(100.0, index=dates, columns=stocks)

        result = factor.compute(inflow_retail=inflow, outflow_retail=outflow, cap=cap)
        assert isinstance(result, pd.DataFrame)

    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        stocks = ["A", "B"]

        inflow = pd.DataFrame(1.0, index=dates, columns=stocks)
        outflow = pd.DataFrame(1.0, index=dates, columns=stocks)
        cap = pd.DataFrame(100.0, index=dates, columns=stocks)

        result = factor.compute(inflow_retail=inflow, outflow_retail=outflow, cap=cap, m=5)
        assert result.shape == inflow.shape

    def test_output_index_and_columns_preserved(self, factor):
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        stocks = ["000001.SZ", "600000.SH"]

        inflow = pd.DataFrame(1.0, index=dates, columns=stocks)
        outflow = pd.DataFrame(1.0, index=dates, columns=stocks)
        cap = pd.DataFrame(100.0, index=dates, columns=stocks)

        result = factor.compute(inflow_retail=inflow, outflow_retail=outflow, cap=cap)
        pd.testing.assert_index_equal(result.index, dates)
        assert list(result.columns) == stocks
