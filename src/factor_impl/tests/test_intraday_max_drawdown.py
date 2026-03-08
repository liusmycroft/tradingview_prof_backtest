import numpy as np
import pandas as pd
import pytest

from factors.intraday_max_drawdown import IntradayMaxDrawdownFactor


@pytest.fixture
def factor():
    return IntradayMaxDrawdownFactor()


class TestIntradayMaxDrawdownMetadata:
    def test_name(self, factor):
        assert factor.name == "INTRADAY_MAX_DRAWDOWN"

    def test_category(self, factor):
        assert factor.category == "高频动量反转"

    def test_repr(self, factor):
        assert "INTRADAY_MAX_DRAWDOWN" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "INTRADAY_MAX_DRAWDOWN"
        assert meta["category"] == "高频动量反转"


class TestIntradayMaxDrawdownHandCalculated:
    def test_constant_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        daily_mdd = pd.DataFrame(0.03, index=dates, columns=stocks)

        result = factor.compute(daily_max_drawdown=daily_mdd, T=20)
        np.testing.assert_allclose(result["A"].values, 0.03, atol=1e-10)

    def test_simple_mean_T3(self, factor):
        """T=3 rolling mean with min_periods=1."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_mdd = pd.DataFrame(
            [0.01, 0.02, 0.03, 0.04, 0.05], index=dates, columns=stocks
        )

        result = factor.compute(daily_max_drawdown=daily_mdd, T=3)
        assert result.iloc[0, 0] == pytest.approx(0.01, rel=1e-10)
        assert result.iloc[1, 0] == pytest.approx(0.015, rel=1e-10)
        assert result.iloc[2, 0] == pytest.approx(0.02, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(0.03, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(0.04, rel=1e-10)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        daily_mdd = pd.DataFrame(
            {"A": [0.02] * 10, "B": [0.08] * 10}, index=dates
        )

        result = factor.compute(daily_max_drawdown=daily_mdd, T=5)
        np.testing.assert_allclose(result["A"].values, 0.02, atol=1e-10)
        np.testing.assert_allclose(result["B"].values, 0.08, atol=1e-10)


class TestIntradayMaxDrawdownEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        values = np.ones(10) * 0.03
        values[4] = np.nan
        daily_mdd = pd.DataFrame(values, index=dates, columns=stocks)

        result = factor.compute(daily_max_drawdown=daily_mdd, T=5)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily_mdd = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_max_drawdown=daily_mdd, T=5)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily_mdd = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_max_drawdown=daily_mdd, T=5)
        np.testing.assert_allclose(result["A"].values, 0.0, atol=1e-15)


class TestIntradayMaxDrawdownOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily_mdd = pd.DataFrame(
            np.random.uniform(0.0, 0.1, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_max_drawdown=daily_mdd, T=20)
        assert result.shape == daily_mdd.shape
        assert list(result.columns) == list(daily_mdd.columns)
        assert list(result.index) == list(daily_mdd.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_mdd = pd.DataFrame(
            [0.01, 0.02, 0.03, 0.02, 0.01], index=dates, columns=stocks
        )

        result = factor.compute(daily_max_drawdown=daily_mdd, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_no_leading_nan_min_periods_1(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        daily_mdd = pd.DataFrame(
            np.random.uniform(0.0, 0.1, (10, 2)), index=dates, columns=stocks
        )

        result = factor.compute(daily_max_drawdown=daily_mdd, T=20)
        assert result.iloc[0].notna().all()
