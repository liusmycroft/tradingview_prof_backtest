import numpy as np
import pandas as pd
import pytest

from factors.trend_clarity import TrendClarityFactor


@pytest.fixture
def factor():
    return TrendClarityFactor()


class TestTrendClarityMetadata:
    def test_name(self, factor):
        assert factor.name == "TREND_CLARITY"

    def test_category(self, factor):
        assert factor.category == "量价因子改进"

    def test_repr(self, factor):
        assert "TREND_CLARITY" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "TREND_CLARITY"
        assert meta["category"] == "量价因子改进"


class TestTrendClarityHandCalculated:
    def test_perfect_linear_trend(self, factor):
        """Perfect linear trend -> R^2 = 1.0"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        close = pd.DataFrame([10.0, 20.0, 30.0, 40.0, 50.0], index=dates, columns=stocks)
        result = factor.compute(close=close, T=5)
        assert result.iloc[4, 0] == pytest.approx(1.0, rel=1e-10)

    def test_constant_price_r2_is_1(self, factor):
        """Constant price -> R^2 = 1.0 (perfect fit)"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        close = pd.DataFrame([100.0] * 5, index=dates, columns=stocks)
        result = factor.compute(close=close, T=5)
        assert result.iloc[4, 0] == pytest.approx(1.0, rel=1e-10)

    def test_first_T_minus_1_rows_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        close = pd.DataFrame(np.arange(1, 11, dtype=float), index=dates, columns=stocks)
        result = factor.compute(close=close, T=5)
        assert result.iloc[:4].isna().all().all()
        assert result.iloc[4:].notna().all().all()

    def test_r2_between_0_and_1(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        close = pd.DataFrame([10.0, 30.0, 15.0, 35.0, 20.0], index=dates, columns=stocks)
        result = factor.compute(close=close, T=5)
        val = result.iloc[4, 0]
        assert 0 <= val <= 1


class TestTrendClarityEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        values = np.arange(1, 11, dtype=float)
        values[3] = np.nan
        close = pd.DataFrame(values, index=dates, columns=stocks)
        result = factor.compute(close=close, T=5)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        close = pd.DataFrame(np.nan, index=dates, columns=stocks)
        result = factor.compute(close=close, T=5)
        assert result.isna().all().all()


class TestTrendClarityOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A", "B", "C"]
        close = pd.DataFrame(np.random.uniform(10, 100, (20, 3)), index=dates, columns=stocks)
        result = factor.compute(close=close, T=10)
        assert result.shape == close.shape
        assert list(result.columns) == list(close.columns)
        assert list(result.index) == list(close.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        close = pd.DataFrame(np.arange(1, 11, dtype=float), index=dates, columns=stocks)
        result = factor.compute(close=close, T=5)
        assert isinstance(result, pd.DataFrame)
