import numpy as np
import pandas as pd
import pytest

from factors.volume_kurtosis import VolumeKurtosisFactor


@pytest.fixture
def factor():
    return VolumeKurtosisFactor()


class TestVolumeKurtosisMetadata:
    def test_name(self, factor):
        assert factor.name == "VOLUME_KURTOSIS"

    def test_category(self, factor):
        assert factor.category == "高频成交分布"

    def test_repr(self, factor):
        assert "VOLUME_KURTOSIS" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "VOLUME_KURTOSIS"
        assert meta["category"] == "高频成交分布"


class TestVolumeKurtosisHandCalculated:
    def test_rolling_mean_T3(self, factor):
        """T=3 滚动均值手算验证。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        data = pd.DataFrame([3.0, 4.0, 5.0, 2.0, 6.0], index=dates, columns=stocks)

        result = factor.compute(daily_volume_kurtosis=data, T=3)

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])

        expected_2 = (3.0 + 4.0 + 5.0) / 3.0
        assert result.iloc[2, 0] == pytest.approx(expected_2, rel=1e-10)

        expected_3 = (4.0 + 5.0 + 2.0) / 3.0
        assert result.iloc[3, 0] == pytest.approx(expected_3, rel=1e-10)

        expected_4 = (5.0 + 2.0 + 6.0) / 3.0
        assert result.iloc[4, 0] == pytest.approx(expected_4, rel=1e-10)

    def test_constant_kurtosis(self, factor):
        """常数峰度时，滚动均值等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        data = pd.DataFrame([3.5] * 5, index=dates, columns=stocks)

        result = factor.compute(daily_volume_kurtosis=data, T=3)
        assert result.iloc[2, 0] == pytest.approx(3.5, rel=1e-10)


class TestVolumeKurtosisEdgeCases:
    def test_nan_propagation(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        data = pd.DataFrame([1.0, np.nan, 3.0, 4.0, 5.0], index=dates, columns=stocks)

        result = factor.compute(daily_volume_kurtosis=data, T=3)
        assert np.isnan(result.iloc[2, 0])
        assert np.isnan(result.iloc[3, 0])
        assert not np.isnan(result.iloc[4, 0])

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        data = pd.DataFrame(
            np.random.uniform(2, 5, (30, 2)), index=dates, columns=stocks
        )
        result = factor.compute(daily_volume_kurtosis=data, T=20)
        assert result.shape == data.shape
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(np.ones(10), index=dates, columns=stocks)
        T = 5
        result = factor.compute(daily_volume_kurtosis=data, T=T)
        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
