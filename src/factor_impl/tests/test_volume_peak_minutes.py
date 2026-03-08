import numpy as np
import pandas as pd
import pytest

from factors.volume_peak_minutes import VolumePeakMinutesFactor


@pytest.fixture
def factor():
    return VolumePeakMinutesFactor()


class TestVolumePeakMinutesMetadata:
    def test_name(self, factor):
        assert factor.name == "VOLUME_PEAK_MINUTES"

    def test_category(self, factor):
        assert factor.category == "高频成交分布"

    def test_repr(self, factor):
        assert "VOLUME_PEAK_MINUTES" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "VOLUME_PEAK_MINUTES"
        assert meta["category"] == "高频成交分布"


class TestVolumePeakMinutesCompute:
    def test_constant_input(self, factor):
        """常数输入时，滚动均值应等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(5.0, index=dates, columns=stocks)

        result = factor.compute(daily_peak_minutes=daily, T=20)
        np.testing.assert_array_almost_equal(result["A"].values, 5.0)

    def test_rolling_mean(self, factor):
        """手动验证滚动均值。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame([2.0, 4.0, 6.0, 8.0, 10.0], index=dates, columns=stocks)

        result = factor.compute(daily_peak_minutes=daily, T=3)
        # T=3, min_periods=1
        assert result.iloc[0, 0] == pytest.approx(2.0)
        assert result.iloc[1, 0] == pytest.approx(3.0)
        assert result.iloc[2, 0] == pytest.approx(4.0)
        assert result.iloc[3, 0] == pytest.approx(6.0)
        assert result.iloc[4, 0] == pytest.approx(8.0)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        daily = pd.DataFrame({"A": [3.0]*10, "B": [7.0]*10}, index=dates)

        result = factor.compute(daily_peak_minutes=daily, T=5)
        np.testing.assert_array_almost_equal(result["A"].values, 3.0)
        np.testing.assert_array_almost_equal(result["B"].values, 7.0)

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily = pd.DataFrame(np.random.rand(30, 3), index=dates, columns=stocks)

        result = factor.compute(daily_peak_minutes=daily, T=20)
        assert result.shape == daily.shape


class TestVolumePeakMinutesEdgeCases:
    def test_single_value(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        daily = pd.DataFrame([3.0], index=dates, columns=["A"])

        result = factor.compute(daily_peak_minutes=daily, T=20)
        assert result.iloc[0, 0] == pytest.approx(3.0)

    def test_nan_handling(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        values = [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        daily = pd.DataFrame(values, index=dates, columns=["A"])

        result = factor.compute(daily_peak_minutes=daily, T=5)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        daily = pd.DataFrame(np.nan, index=dates, columns=["A"])

        result = factor.compute(daily_peak_minutes=daily, T=5)
        assert result.isna().all().all()
