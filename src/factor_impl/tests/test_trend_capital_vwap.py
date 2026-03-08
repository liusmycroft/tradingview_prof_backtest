import numpy as np
import pandas as pd
import pytest

from factors.trend_capital_vwap import TrendCapitalVWAPFactor


@pytest.fixture
def factor():
    return TrendCapitalVWAPFactor()


class TestTrendCapitalVWAPMetadata:
    def test_name(self, factor):
        assert factor.name == "TREND_CAPITAL_VWAP"

    def test_category(self, factor):
        assert factor.category == "量价因子改进"

    def test_repr(self, factor):
        assert "TREND_CAPITAL_VWAP" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "TREND_CAPITAL_VWAP"
        assert meta["category"] == "量价因子改进"


class TestTrendCapitalVWAPHandCalculated:
    def test_constant_input(self, factor):
        """常数输入时，滚动均值应等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(0.01, index=dates, columns=stocks)

        result = factor.compute(daily_trend_vwap_ratio=daily, T=20)
        np.testing.assert_array_almost_equal(result["A"].values, 0.01)

    def test_rolling_mean_manual_T3(self, factor):
        """手动验证滚动均值 T=3, min_periods=1。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(
            [0.01, 0.02, 0.03, 0.04, 0.05], index=dates, columns=stocks
        )

        result = factor.compute(daily_trend_vwap_ratio=daily, T=3)

        assert result.iloc[0, 0] == pytest.approx(0.01, rel=1e-6)
        assert result.iloc[1, 0] == pytest.approx(0.015, rel=1e-6)
        assert result.iloc[2, 0] == pytest.approx(0.02, rel=1e-6)
        assert result.iloc[3, 0] == pytest.approx(0.03, rel=1e-6)
        assert result.iloc[4, 0] == pytest.approx(0.04, rel=1e-6)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        daily = pd.DataFrame(
            {"A": [0.01] * 10, "B": [-0.02] * 10}, index=dates
        )

        result = factor.compute(daily_trend_vwap_ratio=daily, T=5)
        np.testing.assert_array_almost_equal(result["A"].values, 0.01)
        np.testing.assert_array_almost_equal(result["B"].values, -0.02)


class TestTrendCapitalVWAPEdgeCases:
    def test_single_value(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame([0.005], index=dates, columns=stocks)

        result = factor.compute(daily_trend_vwap_ratio=daily, T=20)
        assert result.iloc[0, 0] == pytest.approx(0.005, rel=1e-10)

    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        values = np.ones(10) * 0.01
        values[3] = np.nan
        daily = pd.DataFrame(values, index=dates, columns=stocks)

        result = factor.compute(daily_trend_vwap_ratio=daily, T=5)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_trend_vwap_ratio=daily, T=5)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_trend_vwap_ratio=daily, T=5)
        for val in result["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)


class TestTrendCapitalVWAPOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily = pd.DataFrame(
            np.random.uniform(-0.02, 0.02, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_trend_vwap_ratio=daily, T=20)
        assert result.shape == daily.shape
        assert list(result.columns) == list(daily.columns)
        assert list(result.index) == list(daily.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame([0.01, 0.02, 0.03, 0.04, 0.05], index=dates, columns=stocks)

        result = factor.compute(daily_trend_vwap_ratio=daily, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_no_leading_nan_min_periods_1(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        daily = pd.DataFrame(
            np.random.uniform(-0.02, 0.02, (10, 2)), index=dates, columns=stocks
        )

        result = factor.compute(daily_trend_vwap_ratio=daily, T=20)
        assert result.iloc[0].notna().all()
        assert result.notna().all().all()
