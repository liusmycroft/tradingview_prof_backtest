import numpy as np
import pandas as pd
import pytest

from factors.volume_peak_count import VolumePeakCountFactor


@pytest.fixture
def factor():
    return VolumePeakCountFactor()


class TestVolumePeakCountMetadata:
    def test_name(self, factor):
        assert factor.name == "VOLUME_PEAK_COUNT"

    def test_category(self, factor):
        assert factor.category == "高频成交分布"

    def test_repr(self, factor):
        assert "VOLUME_PEAK_COUNT" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "VOLUME_PEAK_COUNT"
        assert meta["category"] == "高频成交分布"


class TestVolumePeakCountCompute:
    def test_constant_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        daily = pd.DataFrame(3.0, index=dates, columns=["A"])

        result = factor.compute(daily_peak_count=daily, T=20)
        np.testing.assert_array_almost_equal(result["A"].values, 3.0)

    def test_rolling_mean(self, factor):
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        daily = pd.DataFrame([1.0, 3.0, 5.0, 7.0], index=dates, columns=["A"])

        result = factor.compute(daily_peak_count=daily, T=2)
        assert result.iloc[0, 0] == pytest.approx(1.0)
        assert result.iloc[1, 0] == pytest.approx(2.0)
        assert result.iloc[2, 0] == pytest.approx(4.0)
        assert result.iloc[3, 0] == pytest.approx(6.0)

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        daily = pd.DataFrame(np.random.rand(30, 2), index=dates, columns=stocks)

        result = factor.compute(daily_peak_count=daily, T=20)
        assert result.shape == daily.shape


class TestVolumePeakCountDaily:
    def test_no_peaks(self):
        """所有值相同时，无波峰。"""
        vol = pd.Series([100.0] * 240, index=range(240))
        count = VolumePeakCountFactor.compute_daily_peak_count(vol)
        assert count == 0

    def test_single_spike(self):
        """单个极端值应产生1个波峰。"""
        vol = pd.Series([100.0] * 240, index=range(240))
        vol.iloc[120] = 10000.0
        count = VolumePeakCountFactor.compute_daily_peak_count(vol)
        assert count == 1

    def test_two_separated_spikes(self):
        """两个相隔较远的极端值应产生2个波峰。"""
        vol = pd.Series([100.0] * 240, index=range(240))
        vol.iloc[50] = 10000.0
        vol.iloc[200] = 10000.0
        count = VolumePeakCountFactor.compute_daily_peak_count(vol)
        assert count == 2

    def test_adjacent_spikes_count_as_one(self):
        """相邻的极端值只算1个波峰。"""
        vol = pd.Series([100.0] * 240, index=range(240))
        vol.iloc[100] = 10000.0
        vol.iloc[101] = 10000.0
        count = VolumePeakCountFactor.compute_daily_peak_count(vol)
        assert count == 1


class TestVolumePeakCountEdgeCases:
    def test_single_value(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        daily = pd.DataFrame([2.0], index=dates, columns=["A"])

        result = factor.compute(daily_peak_count=daily, T=20)
        assert result.iloc[0, 0] == pytest.approx(2.0)

    def test_nan_handling(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        values = np.ones(10) * 3.0
        values[5] = np.nan
        daily = pd.DataFrame(values, index=dates, columns=["A"])

        result = factor.compute(daily_peak_count=daily, T=5)
        assert isinstance(result, pd.DataFrame)
