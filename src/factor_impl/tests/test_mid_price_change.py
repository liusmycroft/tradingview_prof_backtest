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
    "factors.mid_price_change", _factors_dir / "mid_price_change.py"
)
_mod = importlib.util.module_from_spec(_mod_spec)
sys.modules["factors.mid_price_change"] = _mod
_mod_spec.loader.exec_module(_mod)

MidPriceChangeFactor = _mod.MidPriceChangeFactor


@pytest.fixture
def factor():
    return MidPriceChangeFactor()


class TestMidPriceChangeMetadata:
    def test_name(self, factor):
        assert factor.name == "MPC"

    def test_category(self, factor):
        assert factor.category == "高频流动性"

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "MPC"
        assert meta["category"] == "高频流动性"

    def test_repr(self, factor):
        assert "MPC" in repr(factor)


class TestMidPriceChangeCompute:
    def test_constant_input(self, factor):
        """常数输入时，滚动均值等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(0.005, index=dates, columns=stocks)

        result = factor.compute(daily_mid_price_change=data, T=5)
        np.testing.assert_allclose(result["A"].values, 0.005, atol=1e-15)

    def test_rolling_mean_T3(self, factor):
        """T=3 滚动均值手算验证。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        data = pd.DataFrame([0.01, 0.02, 0.03, 0.04, 0.05], index=dates, columns=stocks)

        result = factor.compute(daily_mid_price_change=data, T=3)

        # rolling(3, min_periods=1):
        #   day0: mean([0.01]) = 0.01
        #   day1: mean([0.01, 0.02]) = 0.015
        #   day2: mean([0.01, 0.02, 0.03]) = 0.02
        #   day3: mean([0.02, 0.03, 0.04]) = 0.03
        #   day4: mean([0.03, 0.04, 0.05]) = 0.04
        expected = [0.01, 0.015, 0.02, 0.03, 0.04]
        np.testing.assert_allclose(result["A"].values, expected, atol=1e-15)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame(
            {"A": [0.01] * 5, "B": [0.05] * 5}, index=dates
        )

        result = factor.compute(daily_mid_price_change=data, T=3)
        np.testing.assert_allclose(result["A"].values, 0.01, atol=1e-15)
        np.testing.assert_allclose(result["B"].values, 0.05, atol=1e-15)


class TestMidPriceChangeEdgeCases:
    def test_single_value(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        data = pd.DataFrame([0.01], index=dates, columns=["A"])

        result = factor.compute(daily_mid_price_change=data, T=20)
        assert result.iloc[0, 0] == pytest.approx(0.01)

    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame([0.01, np.nan, 0.03, 0.04, 0.05], index=dates, columns=["A"])

        result = factor.compute(daily_mid_price_change=data, T=3)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (5, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame(np.nan, index=dates, columns=["A"])

        result = factor.compute(daily_mid_price_change=data, T=3)
        assert result.isna().all().all()

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        data = pd.DataFrame(np.random.uniform(0, 0.01, (30, 3)), index=dates, columns=stocks)

        result = factor.compute(daily_mid_price_change=data, T=20)
        assert result.shape == data.shape
        assert isinstance(result, pd.DataFrame)

    def test_output_no_leading_nan(self, factor):
        """min_periods=1, 第一行就有值。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(np.ones(10) * 0.01, index=dates, columns=["A"])

        result = factor.compute(daily_mid_price_change=data, T=20)
        assert result.iloc[0].notna().all()
