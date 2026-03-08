import numpy as np
import pandas as pd
import pytest

from factors.large_buy_ratio import LargeBuyRatioFactor


@pytest.fixture
def factor():
    return LargeBuyRatioFactor()


class TestLargeBuyRatioMetadata:
    def test_name(self, factor):
        assert factor.name == "LARGE_BUY_RATIO"

    def test_category(self, factor):
        assert factor.category == "高频资金流"

    def test_repr(self, factor):
        assert "LARGE_BUY_RATIO" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "LARGE_BUY_RATIO"
        assert meta["category"] == "高频资金流"


class TestLargeBuyRatioHandCalculated:
    def test_constant_ratio(self, factor):
        """buy=100, total=1000 -> ratio=0.1, rolling mean=0.1."""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        buy = pd.DataFrame(100.0, index=dates, columns=stocks)
        total = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(large_buy_amount=buy, total_amount=total, T=20)
        assert result.iloc[-1, 0] == pytest.approx(0.1, rel=1e-10)

    def test_simple_mean_T3(self, factor):
        """T=3 rolling mean with min_periods=T."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        buy = pd.DataFrame(
            [100.0, 200.0, 300.0, 400.0, 500.0], index=dates, columns=stocks
        )
        total = pd.DataFrame(1000.0, index=dates, columns=stocks)
        # ratios: [0.1, 0.2, 0.3, 0.4, 0.5]

        result = factor.compute(large_buy_amount=buy, total_amount=total, T=3)
        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(0.2, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(0.3, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(0.4, rel=1e-10)

    def test_two_stocks(self, factor):
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        buy = pd.DataFrame(
            {"A": [100.0] * 20, "B": [500.0] * 20}, index=dates
        )
        total = pd.DataFrame(
            {"A": [1000.0] * 20, "B": [1000.0] * 20}, index=dates
        )

        result = factor.compute(large_buy_amount=buy, total_amount=total, T=20)
        assert result.iloc[-1]["A"] == pytest.approx(0.1, rel=1e-10)
        assert result.iloc[-1]["B"] == pytest.approx(0.5, rel=1e-10)


class TestLargeBuyRatioEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        buy = pd.DataFrame(100.0, index=dates, columns=stocks)
        buy.iloc[5, 0] = np.nan
        total = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(large_buy_amount=buy, total_amount=total, T=20)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (25, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        buy = pd.DataFrame(np.nan, index=dates, columns=stocks)
        total = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(large_buy_amount=buy, total_amount=total, T=20)
        assert result.isna().all().all()

    def test_zero_total(self, factor):
        """Zero total -> inf ratio."""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        buy = pd.DataFrame(100.0, index=dates, columns=stocks)
        total = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(large_buy_amount=buy, total_amount=total, T=20)
        assert isinstance(result, pd.DataFrame)


class TestLargeBuyRatioOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        np.random.seed(42)
        buy = pd.DataFrame(
            np.random.rand(30, 3) * 1e6, index=dates, columns=stocks
        )
        total = pd.DataFrame(
            np.random.rand(30, 3) * 1e7 + 1e6, index=dates, columns=stocks
        )

        result = factor.compute(large_buy_amount=buy, total_amount=total, T=20)
        assert result.shape == buy.shape
        assert list(result.columns) == list(buy.columns)
        assert list(result.index) == list(buy.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        buy = pd.DataFrame(100.0, index=dates, columns=stocks)
        total = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(large_buy_amount=buy, total_amount=total, T=20)
        assert isinstance(result, pd.DataFrame)
