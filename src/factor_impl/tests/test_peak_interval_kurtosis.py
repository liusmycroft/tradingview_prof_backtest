import numpy as np
import pandas as pd
import pytest

from factors.peak_interval_kurtosis import PeakIntervalKurtosisFactor


@pytest.fixture
def factor():
    return PeakIntervalKurtosisFactor()


class TestPeakIntervalKurtosisMetadata:
    def test_name(self, factor):
        assert factor.name == "PEAK_INTERVAL_KURTOSIS"

    def test_category(self, factor):
        assert factor.category == "高频因子-成交分布类"

    def test_repr(self, factor):
        assert "PEAK_INTERVAL_KURTOSIS" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "PEAK_INTERVAL_KURTOSIS"


class TestPeakIntervalKurtosisCompute:
    def test_constant_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        peak_intervals = pd.DataFrame(3.0, index=dates, columns=stocks)

        result = factor.compute(peak_intervals=peak_intervals, T=20)
        np.testing.assert_array_almost_equal(result["A"].values, 3.0)

    def test_rolling_mean(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        peak_intervals = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0], index=dates, columns=stocks)

        result = factor.compute(peak_intervals=peak_intervals, T=3)
        assert result.iloc[2, 0] == pytest.approx(2.0, rel=1e-6)
        assert result.iloc[4, 0] == pytest.approx(4.0, rel=1e-6)

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        peak_intervals = pd.DataFrame(np.random.randn(30, 2), index=dates, columns=stocks)

        result = factor.compute(peak_intervals=peak_intervals, T=20)
        assert result.shape == peak_intervals.shape
        assert isinstance(result, pd.DataFrame)

    def test_min_periods_1(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        peak_intervals = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0], index=dates, columns=stocks)

        result = factor.compute(peak_intervals=peak_intervals, T=20)
        assert result.iloc[0].notna().all()
