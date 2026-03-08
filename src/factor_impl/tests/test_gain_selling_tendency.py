import numpy as np
import pandas as pd
import pytest

from factors.gain_selling_tendency import GainSellingTendencyFactor


@pytest.fixture
def factor():
    return GainSellingTendencyFactor()


class TestGainSellingTendencyMetadata:
    def test_name(self, factor):
        assert factor.name == "GAIN_SELLING_TENDENCY"

    def test_category(self, factor):
        assert factor.category == "行为金融-处置效应"

    def test_repr(self, factor):
        assert "GAIN_SELLING_TENDENCY" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "GAIN_SELLING_TENDENCY"
        assert meta["category"] == "行为金融-处置效应"


class TestGainSellingTendencyHandCalculated:
    def test_all_gain_days(self, factor):
        """When close > vwap every day, gain is positive."""
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        stocks = ["A"]
        close = pd.DataFrame(110.0, index=dates, columns=stocks)
        vwap = pd.DataFrame(100.0, index=dates, columns=stocks)
        turnover = pd.DataFrame(0.05, index=dates, columns=stocks)

        result = factor.compute(close=close, vwap=vwap, turnover=turnover, T=60)
        assert result.iloc[-1, 0] > 0

    def test_no_gain_days(self, factor):
        """When close < vwap every day, gain is zero."""
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        stocks = ["A"]
        close = pd.DataFrame(90.0, index=dates, columns=stocks)
        vwap = pd.DataFrame(100.0, index=dates, columns=stocks)
        turnover = pd.DataFrame(0.05, index=dates, columns=stocks)

        result = factor.compute(close=close, vwap=vwap, turnover=turnover, T=60)
        assert result.iloc[-1, 0] == pytest.approx(0.0, abs=1e-10)

    def test_insufficient_window_is_nan(self, factor):
        """With T=60 and only 30 rows, all should be NaN."""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A"]
        close = pd.DataFrame(110.0, index=dates, columns=stocks)
        vwap = pd.DataFrame(100.0, index=dates, columns=stocks)
        turnover = pd.DataFrame(0.05, index=dates, columns=stocks)

        result = factor.compute(close=close, vwap=vwap, turnover=turnover, T=60)
        assert result.isna().all().all()


class TestGainSellingTendencyEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        stocks = ["A"]
        close = pd.DataFrame(110.0, index=dates, columns=stocks)
        close.iloc[10, 0] = np.nan
        vwap = pd.DataFrame(100.0, index=dates, columns=stocks)
        turnover = pd.DataFrame(0.05, index=dates, columns=stocks)

        result = factor.compute(close=close, vwap=vwap, turnover=turnover, T=60)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (60, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        stocks = ["A"]
        close = pd.DataFrame(np.nan, index=dates, columns=stocks)
        vwap = pd.DataFrame(np.nan, index=dates, columns=stocks)
        turnover = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(close=close, vwap=vwap, turnover=turnover, T=60)
        assert result.isna().all().all()

    def test_zero_turnover(self, factor):
        """Zero turnover means zero weights -> NaN result."""
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        stocks = ["A"]
        close = pd.DataFrame(110.0, index=dates, columns=stocks)
        vwap = pd.DataFrame(100.0, index=dates, columns=stocks)
        turnover = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(close=close, vwap=vwap, turnover=turnover, T=60)
        assert isinstance(result, pd.DataFrame)


class TestGainSellingTendencyOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=80, freq="D")
        stocks = ["A", "B", "C"]
        np.random.seed(42)
        close = pd.DataFrame(
            np.random.uniform(90, 110, (80, 3)), index=dates, columns=stocks
        )
        vwap = pd.DataFrame(
            np.random.uniform(90, 110, (80, 3)), index=dates, columns=stocks
        )
        turnover = pd.DataFrame(
            np.random.uniform(0.01, 0.1, (80, 3)), index=dates, columns=stocks
        )

        result = factor.compute(close=close, vwap=vwap, turnover=turnover, T=60)
        assert result.shape == close.shape
        assert list(result.columns) == list(close.columns)
        assert list(result.index) == list(close.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        stocks = ["A"]
        close = pd.DataFrame(100.0, index=dates, columns=stocks)
        vwap = pd.DataFrame(100.0, index=dates, columns=stocks)
        turnover = pd.DataFrame(0.05, index=dates, columns=stocks)

        result = factor.compute(close=close, vwap=vwap, turnover=turnover, T=60)
        assert isinstance(result, pd.DataFrame)
