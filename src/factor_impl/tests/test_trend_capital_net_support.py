import numpy as np
import pandas as pd
import pytest

from factors.trend_capital_net_support import TrendCapitalNetSupportFactor


@pytest.fixture
def factor():
    return TrendCapitalNetSupportFactor()


class TestTrendCapitalNetSupportMetadata:
    def test_name(self, factor):
        assert factor.name == "TREND_CAPITAL_NET_SUPPORT"

    def test_category(self, factor):
        assert factor.category == "量价因子改进"

    def test_repr(self, factor):
        assert "TREND_CAPITAL_NET_SUPPORT" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "TREND_CAPITAL_NET_SUPPORT"
        assert meta["category"] == "量价因子改进"


class TestTrendCapitalNetSupportHandCalculated:
    def test_constant_input(self, factor):
        """常数输入时，滚动均值应等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        data = pd.DataFrame(0.02, index=dates, columns=["A"])

        result = factor.compute(daily_trend_net_support=data, T=10)

        np.testing.assert_array_almost_equal(result["A"].values, 0.02)

    def test_rolling_mean_T3(self, factor):
        """T=3 滚动均值手算验证 (min_periods=1)。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame(
            [0.01, 0.02, 0.03, 0.04, 0.05], index=dates, columns=["A"]
        )

        result = factor.compute(daily_trend_net_support=data, T=3)

        assert result.iloc[0, 0] == pytest.approx(0.01, rel=1e-10)
        assert result.iloc[1, 0] == pytest.approx(0.015, rel=1e-10)
        assert result.iloc[2, 0] == pytest.approx(0.02, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(0.03, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(0.04, rel=1e-10)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(
            {"A": [0.01] * 10, "B": [0.05] * 10}, index=dates
        )

        result = factor.compute(daily_trend_net_support=data, T=5)

        np.testing.assert_array_almost_equal(result["A"].values, 0.01)
        np.testing.assert_array_almost_equal(result["B"].values, 0.05)

    def test_negative_values(self, factor):
        """负值（阻力大于支撑）应正确处理。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame(
            [-0.03, -0.02, -0.01, 0.0, 0.01], index=dates, columns=["A"]
        )

        result = factor.compute(daily_trend_net_support=data, T=5)
        # 最后一个值 = mean([-0.03, -0.02, -0.01, 0.0, 0.01]) = -0.01
        assert result.iloc[4, 0] == pytest.approx(-0.01, rel=1e-10)


class TestTrendCapitalNetSupportEdgeCases:
    def test_single_value(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        data = pd.DataFrame([0.05], index=dates, columns=["A"])

        result = factor.compute(daily_trend_net_support=data, T=20)
        assert result.iloc[0, 0] == pytest.approx(0.05, rel=1e-10)

    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        values = np.ones(10) * 0.02
        values[3] = np.nan
        data = pd.DataFrame(values, index=dates, columns=["A"])

        result = factor.compute(daily_trend_net_support=data, T=5)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(np.nan, index=dates, columns=["A"])

        result = factor.compute(daily_trend_net_support=data, T=5)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(0.0, index=dates, columns=["A"])

        result = factor.compute(daily_trend_net_support=data, T=5)
        for val in result["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)


class TestTrendCapitalNetSupportOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        data = pd.DataFrame(
            np.random.uniform(-0.05, 0.05, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_trend_net_support=data, T=20)

        assert result.shape == data.shape
        assert list(result.columns) == list(data.columns)
        assert list(result.index) == list(data.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame(
            [0.01, 0.02, 0.03, 0.04, 0.05], index=dates, columns=["A"]
        )

        result = factor.compute(daily_trend_net_support=data, T=3)
        assert isinstance(result, pd.DataFrame)
