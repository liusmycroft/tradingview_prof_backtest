import numpy as np
import pandas as pd
import pytest

from factors.gaming_factor import GamingFactor


@pytest.fixture
def factor():
    return GamingFactor()


class TestGamingFactorMetadata:
    def test_name(self, factor):
        assert factor.name == "STREN"

    def test_category(self, factor):
        assert factor.category == "高频资金流"

    def test_repr(self, factor):
        assert "STREN" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "STREN"
        assert meta["category"] == "高频资金流"


class TestGamingFactorHandCalculated:
    def test_equal_buy_sell(self, factor):
        """When buy == sell, ratio should be 1.0."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        buy = pd.DataFrame(1000.0, index=dates, columns=stocks)
        sell = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_weighted_buy_vol=buy, daily_weighted_sell_vol=sell
        )
        np.testing.assert_allclose(result["A"].values, 1.0, atol=1e-10)

    def test_known_ratio(self, factor):
        """buy=200, sell=100 -> ratio=2.0."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        buy = pd.DataFrame(200.0, index=dates, columns=stocks)
        sell = pd.DataFrame(100.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_weighted_buy_vol=buy, daily_weighted_sell_vol=sell
        )
        np.testing.assert_allclose(result["A"].values, 2.0, atol=1e-10)

    def test_two_stocks(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        buy = pd.DataFrame({"A": [300.0] * 5, "B": [100.0] * 5}, index=dates)
        sell = pd.DataFrame({"A": [100.0] * 5, "B": [200.0] * 5}, index=dates)

        result = factor.compute(
            daily_weighted_buy_vol=buy, daily_weighted_sell_vol=sell
        )
        np.testing.assert_allclose(result["A"].values, 3.0, atol=1e-10)
        np.testing.assert_allclose(result["B"].values, 0.5, atol=1e-10)


class TestGamingFactorEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        buy = pd.DataFrame(np.ones(10) * 100, index=dates, columns=stocks)
        buy.iloc[3, 0] = np.nan
        sell = pd.DataFrame(np.ones(10) * 50, index=dates, columns=stocks)

        result = factor.compute(
            daily_weighted_buy_vol=buy, daily_weighted_sell_vol=sell
        )
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)
        assert np.isnan(result.iloc[3, 0])

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        buy = pd.DataFrame(np.nan, index=dates, columns=stocks)
        sell = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(
            daily_weighted_buy_vol=buy, daily_weighted_sell_vol=sell
        )
        assert result.isna().all().all()

    def test_zero_sell_becomes_nan(self, factor):
        """Zero sell volume should produce NaN (division by zero handled)."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        buy = pd.DataFrame(100.0, index=dates, columns=stocks)
        sell = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_weighted_buy_vol=buy, daily_weighted_sell_vol=sell
        )
        assert result.isna().all().all()


class TestGamingFactorOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        np.random.seed(42)
        buy = pd.DataFrame(
            np.random.rand(30, 3) * 1e6 + 1e5, index=dates, columns=stocks
        )
        sell = pd.DataFrame(
            np.random.rand(30, 3) * 1e6 + 1e5, index=dates, columns=stocks
        )

        result = factor.compute(
            daily_weighted_buy_vol=buy, daily_weighted_sell_vol=sell
        )
        assert result.shape == buy.shape
        assert list(result.columns) == list(buy.columns)
        assert list(result.index) == list(buy.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        buy = pd.DataFrame([100.0] * 5, index=dates, columns=stocks)
        sell = pd.DataFrame([50.0] * 5, index=dates, columns=stocks)

        result = factor.compute(
            daily_weighted_buy_vol=buy, daily_weighted_sell_vol=sell
        )
        assert isinstance(result, pd.DataFrame)
