import numpy as np
import pandas as pd
import pytest

from factors.large_vol_price_corr import LargeVolPriceCorrFactor


@pytest.fixture
def factor():
    return LargeVolPriceCorrFactor()


class TestLargeVolPriceCorrMetadata:
    def test_name(self, factor):
        assert factor.name == "LARGE_VOL_PRICE_CORR"

    def test_category(self, factor):
        assert factor.category == "高频量价相关性"

    def test_repr(self, factor):
        assert "LARGE_VOL_PRICE_CORR" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "LARGE_VOL_PRICE_CORR"
        assert meta["category"] == "高频量价相关性"


class TestLargeVolPriceCorrHandCalculated:
    def test_constant_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(0.5, index=dates, columns=stocks)

        result = factor.compute(daily_large_vol_corr=data, T=20)
        assert result.iloc[-1, 0] == pytest.approx(0.5, rel=1e-10)

    def test_simple_mean_T3(self, factor):
        """T=3 rolling mean with min_periods=T."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(
            [0.1, 0.2, 0.3, 0.4, 0.5], index=dates, columns=stocks
        )

        result = factor.compute(daily_large_vol_corr=data, T=3)
        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(0.2, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(0.3, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(0.4, rel=1e-10)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        data = pd.DataFrame(
            {"A": [0.3] * 20, "B": [-0.3] * 20}, index=dates
        )

        result = factor.compute(daily_large_vol_corr=data, T=20)
        assert result.iloc[-1]["A"] == pytest.approx(0.3, rel=1e-10)
        assert result.iloc[-1]["B"] == pytest.approx(-0.3, rel=1e-10)


class TestLargeVolPriceCorrEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        values = np.ones(25) * 0.5
        values[5] = np.nan
        data = pd.DataFrame(values, index=dates, columns=stocks)

        result = factor.compute(daily_large_vol_corr=data, T=20)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (25, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_large_vol_corr=data, T=20)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_large_vol_corr=data, T=20)
        # First 19 NaN (min_periods=T=20), then 0.0
        assert result.iloc[-1, 0] == pytest.approx(0.0, abs=1e-15)


class TestLargeVolPriceCorrOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        data = pd.DataFrame(
            np.random.uniform(-0.8, 0.8, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_large_vol_corr=data, T=20)
        assert result.shape == data.shape
        assert list(result.columns) == list(data.columns)
        assert list(result.index) == list(data.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(
            np.random.uniform(-0.5, 0.5, 25), index=dates, columns=stocks
        )

        result = factor.compute(daily_large_vol_corr=data, T=20)
        assert isinstance(result, pd.DataFrame)
