import numpy as np
import pandas as pd
import pytest

import importlib
import importlib.util
import pathlib
import sys
import types

_factors_dir = pathlib.Path(__file__).resolve().parent.parent / "factors"

_pkg = types.ModuleType("factors")
_pkg.__path__ = [str(_factors_dir)]
sys.modules["factors"] = _pkg

_base_spec = importlib.util.spec_from_file_location("factors.base", _factors_dir / "base.py")
_base_mod = importlib.util.module_from_spec(_base_spec)
sys.modules["factors.base"] = _base_mod
_base_spec.loader.exec_module(_base_mod)

_mod_spec = importlib.util.spec_from_file_location(
    "factors.inst_trade_heat", _factors_dir / "inst_trade_heat.py"
)
_mod = importlib.util.module_from_spec(_mod_spec)
sys.modules["factors.inst_trade_heat"] = _mod
_mod_spec.loader.exec_module(_mod)

InstTradeHeatFactor = _mod.InstTradeHeatFactor


@pytest.fixture
def factor():
    return InstTradeHeatFactor()


# --------------------------------------------------------------------------- #
# 基本元信息
# --------------------------------------------------------------------------- #

class TestMetadata:
    def test_name(self, factor):
        assert factor.name == "INST_TRADE_HEAT"

    def test_category(self, factor):
        assert factor.category == "投资者注意力"

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "INST_TRADE_HEAT"
        assert "description" in meta


# --------------------------------------------------------------------------- #
# 手工验算
# --------------------------------------------------------------------------- #

class TestHandCalculation:
    def test_constant_inputs(self, factor):
        """常数输入：每天 inflow=1e6, outflow=1e6, cap=1e9 => daily=0.002，均值=0.002"""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        stocks = ["A"]

        inflow = pd.DataFrame(1e6, index=dates, columns=stocks)
        outflow = pd.DataFrame(1e6, index=dates, columns=stocks)
        cap = pd.DataFrame(1e9, index=dates, columns=stocks)

        result = factor.compute(inflow_inst=inflow, outflow_inst=outflow, cap=cap, m=3)

        expected = pd.DataFrame(0.002, index=dates, columns=stocks)
        pd.testing.assert_frame_equal(result, expected)

    def test_varying_inputs(self, factor):
        """变化输入，手工计算滚动均值。"""
        dates = pd.date_range("2024-01-01", periods=4, freq="B")
        stocks = ["A"]

        inflow = pd.DataFrame([100, 200, 300, 400], index=dates, columns=stocks, dtype=float)
        outflow = pd.DataFrame([0, 0, 0, 0], index=dates, columns=stocks, dtype=float)
        cap = pd.DataFrame([1000, 1000, 1000, 1000], index=dates, columns=stocks, dtype=float)

        result = factor.compute(inflow_inst=inflow, outflow_inst=outflow, cap=cap, m=3)

        # daily_turnover: [0.1, 0.2, 0.3, 0.4]
        # rolling(3, min_periods=1):
        #   day0: mean([0.1])           = 0.1
        #   day1: mean([0.1, 0.2])      = 0.15
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

        result = factor.compute(inflow_inst=inflow, outflow_inst=outflow, cap=cap, m=2)

        # A daily: [0.1, 0.2, 0.3] => rolling(2): [0.1, 0.15, 0.25]
        # B daily: [0.1, 0.2, 0.3] => rolling(2): [0.1, 0.15, 0.25]
        np.testing.assert_allclose(result["A"].values, [0.1, 0.15, 0.25], atol=1e-12)
        np.testing.assert_allclose(result["B"].values, [0.1, 0.15, 0.25], atol=1e-12)


# --------------------------------------------------------------------------- #
# 边界情况
# --------------------------------------------------------------------------- #

class TestEdgeCases:
    def test_zero_cap_produces_inf(self, factor):
        """市值为零时，除法产生 inf，不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        stocks = ["A"]

        inflow = pd.DataFrame(1e6, index=dates, columns=stocks)
        outflow = pd.DataFrame(1e6, index=dates, columns=stocks)
        cap = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(inflow_inst=inflow, outflow_inst=outflow, cap=cap, m=2)

        assert result.shape == (3, 1)
        assert np.all(np.isinf(result.values) | np.isnan(result.values))

    def test_nan_propagation(self, factor):
        """输入含 NaN 时，rolling mean(min_periods=1) 会跳过 NaN。"""
        dates = pd.date_range("2024-01-01", periods=4, freq="B")
        stocks = ["A"]

        inflow = pd.DataFrame([100.0, np.nan, 100.0, 100.0], index=dates, columns=stocks)
        outflow = pd.DataFrame(100.0, index=dates, columns=stocks)
        cap = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(inflow_inst=inflow, outflow_inst=outflow, cap=cap, m=2)

        # daily_turnover: [0.2, NaN, 0.2, 0.2]
        # rolling(2, min_periods=1): [0.2, 0.2, 0.2, 0.2]
        expected = pd.DataFrame([0.2, 0.2, 0.2, 0.2], index=dates, columns=stocks)
        pd.testing.assert_frame_equal(result, expected)

    def test_m_equals_one(self, factor):
        """m=1 时，结果等于日度换手率本身。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        stocks = ["A"]

        inflow = pd.DataFrame([100.0, 200.0, 300.0], index=dates, columns=stocks)
        outflow = pd.DataFrame(0.0, index=dates, columns=stocks)
        cap = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(inflow_inst=inflow, outflow_inst=outflow, cap=cap, m=1)

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

        result = factor.compute(inflow_inst=inflow, outflow_inst=outflow, cap=cap)
        assert isinstance(result, pd.DataFrame)

    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        stocks = ["A", "B"]

        inflow = pd.DataFrame(1.0, index=dates, columns=stocks)
        outflow = pd.DataFrame(1.0, index=dates, columns=stocks)
        cap = pd.DataFrame(100.0, index=dates, columns=stocks)

        result = factor.compute(inflow_inst=inflow, outflow_inst=outflow, cap=cap, m=5)
        assert result.shape == inflow.shape

    def test_output_index_and_columns_preserved(self, factor):
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        stocks = ["000001.SZ", "600000.SH"]

        inflow = pd.DataFrame(1.0, index=dates, columns=stocks)
        outflow = pd.DataFrame(1.0, index=dates, columns=stocks)
        cap = pd.DataFrame(100.0, index=dates, columns=stocks)

        result = factor.compute(inflow_inst=inflow, outflow_inst=outflow, cap=cap)
        pd.testing.assert_index_equal(result.index, dates)
        assert list(result.columns) == stocks
