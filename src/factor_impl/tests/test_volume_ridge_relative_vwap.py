import numpy as np
import pandas as pd
import pytest

from factors.volume_ridge_relative_vwap import VolumeRidgeRelativeVWAPFactor


@pytest.fixture
def factor():
    return VolumeRidgeRelativeVWAPFactor()


class TestVolumeRidgeRelativeVWAPMetadata:
    def test_name(self, factor):
        assert factor.name == "VOLUME_RIDGE_RELATIVE_VWAP"

    def test_category(self, factor):
        assert factor.category == "高频量价相关性"

    def test_repr(self, factor):
        assert "VOLUME_RIDGE_RELATIVE_VWAP" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "VOLUME_RIDGE_RELATIVE_VWAP"
        assert meta["category"] == "高频量价相关性"


class TestVolumeRidgeRelativeVWAPCompute:
    def test_constant_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        daily = pd.DataFrame(1.05, index=dates, columns=["A"])

        result = factor.compute(daily_ridge_vwap_ratio=daily, T=20)
        np.testing.assert_array_almost_equal(result["A"].values, 1.05)

    def test_rolling_mean(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        daily = pd.DataFrame([1.0, 1.1, 1.2, 1.3, 1.4], index=dates, columns=["A"])

        result = factor.compute(daily_ridge_vwap_ratio=daily, T=3)
        assert result.iloc[0, 0] == pytest.approx(1.0)
        assert result.iloc[1, 0] == pytest.approx(1.05)
        assert result.iloc[2, 0] == pytest.approx(1.1)
        assert result.iloc[3, 0] == pytest.approx(1.2)
        assert result.iloc[4, 0] == pytest.approx(1.3)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        daily = pd.DataFrame({"A": [1.0]*10, "B": [1.1]*10}, index=dates)

        result = factor.compute(daily_ridge_vwap_ratio=daily, T=5)
        np.testing.assert_array_almost_equal(result["A"].values, 1.0)
        np.testing.assert_array_almost_equal(result["B"].values, 1.1)

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily = pd.DataFrame(np.random.uniform(0.9, 1.1, (30, 3)),
                             index=dates, columns=stocks)

        result = factor.compute(daily_ridge_vwap_ratio=daily, T=20)
        assert result.shape == daily.shape


class TestVolumeRidgeRelativeVWAPEdgeCases:
    def test_single_value(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        daily = pd.DataFrame([1.02], index=dates, columns=["A"])

        result = factor.compute(daily_ridge_vwap_ratio=daily, T=20)
        assert result.iloc[0, 0] == pytest.approx(1.02)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        daily = pd.DataFrame(np.nan, index=dates, columns=["A"])

        result = factor.compute(daily_ridge_vwap_ratio=daily, T=5)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        daily = pd.DataFrame(0.0, index=dates, columns=["A"])

        result = factor.compute(daily_ridge_vwap_ratio=daily, T=5)
        for val in result["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)
