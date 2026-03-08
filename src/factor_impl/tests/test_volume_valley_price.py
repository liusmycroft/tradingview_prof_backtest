import numpy as np
import pandas as pd
import pytest

from factors.volume_valley_price import VolumeValleyPriceFactor


@pytest.fixture
def factor():
    return VolumeValleyPriceFactor()


class TestVolumeValleyPriceMetadata:
    def test_name(self, factor):
        assert factor.name == "VOLUME_VALLEY_PRICE"

    def test_category(self, factor):
        assert factor.category == "高频量价相关性"

    def test_repr(self, factor):
        assert "VOLUME_VALLEY_PRICE" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "VOLUME_VALLEY_PRICE"
        assert meta["category"] == "高频量价相关性"


class TestVolumeValleyPriceHandCalculated:
    def test_equal_vwap(self, factor):
        """valley_vwap == daily_vwap => ratio = 1.0"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        vwap = pd.DataFrame(10.0, index=dates, columns=["A"])

        result = factor.compute(valley_vwap=vwap, daily_vwap=vwap, T=3)
        np.testing.assert_allclose(result["A"].values, 1.0, atol=1e-15)

    def test_known_values(self, factor):
        """Hand-calculated rolling mean with min_periods=1."""
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        valley = pd.DataFrame([9.0, 10.0, 11.0, 12.0], index=dates, columns=["A"])
        daily = pd.DataFrame([10.0, 10.0, 10.0, 10.0], index=dates, columns=["A"])

        result = factor.compute(valley_vwap=valley, daily_vwap=daily, T=3)

        # daily_ratio: [0.9, 1.0, 1.1, 1.2]
        # rolling(3, min_periods=1):
        #   day0: mean([0.9]) = 0.9
        #   day1: mean([0.9, 1.0]) = 0.95
        #   day2: mean([0.9, 1.0, 1.1]) = 1.0
        #   day3: mean([1.0, 1.1, 1.2]) = 1.1
        expected = [0.9, 0.95, 1.0, 1.1]
        np.testing.assert_allclose(result["A"].values, expected, atol=1e-10)

    def test_valley_below_daily(self, factor):
        """valley price < daily => ratio < 1"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        valley = pd.DataFrame(9.0, index=dates, columns=["A"])
        daily = pd.DataFrame(10.0, index=dates, columns=["A"])

        result = factor.compute(valley_vwap=valley, daily_vwap=daily, T=3)
        assert (result["A"] < 1.0).all()


class TestVolumeValleyPriceEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        valley = pd.DataFrame([9.0, np.nan, 11.0, 12.0, 13.0], index=dates, columns=["A"])
        daily = pd.DataFrame(10.0, index=dates, columns=["A"])

        result = factor.compute(valley_vwap=valley, daily_vwap=daily, T=3)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (5, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        valley = pd.DataFrame(np.nan, index=dates, columns=["A"])
        daily = pd.DataFrame(10.0, index=dates, columns=["A"])

        result = factor.compute(valley_vwap=valley, daily_vwap=daily, T=3)
        assert result.isna().all().all()

    def test_zero_daily_vwap(self, factor):
        """daily_vwap=0 produces inf/nan, should not raise."""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        valley = pd.DataFrame(10.0, index=dates, columns=["A"])
        daily = pd.DataFrame(0.0, index=dates, columns=["A"])

        result = factor.compute(valley_vwap=valley, daily_vwap=daily, T=2)
        assert result.shape == (3, 1)


class TestVolumeValleyPriceOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        valley = pd.DataFrame(np.random.uniform(9, 11, (30, 3)), index=dates, columns=stocks)
        daily = pd.DataFrame(np.random.uniform(9, 11, (30, 3)), index=dates, columns=stocks)

        result = factor.compute(valley_vwap=valley, daily_vwap=daily, T=20)
        assert result.shape == valley.shape
        assert list(result.columns) == list(valley.columns)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        valley = pd.DataFrame(9.0, index=dates, columns=["A"])
        daily = pd.DataFrame(10.0, index=dates, columns=["A"])

        result = factor.compute(valley_vwap=valley, daily_vwap=daily, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_no_leading_nan_min_periods_1(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        valley = pd.DataFrame(9.0, index=dates, columns=["A"])
        daily = pd.DataFrame(10.0, index=dates, columns=["A"])

        result = factor.compute(valley_vwap=valley, daily_vwap=daily, T=20)
        assert result.iloc[0].notna().all()
