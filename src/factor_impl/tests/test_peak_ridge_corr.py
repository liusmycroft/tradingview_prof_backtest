import numpy as np
import pandas as pd
import pytest

from factors.peak_ridge_corr import PeakRidgeCorrFactor


@pytest.fixture
def factor():
    return PeakRidgeCorrFactor()


class TestPeakRidgeCorrMetadata:
    def test_name(self, factor):
        assert factor.name == "PEAK_RIDGE_CORR"

    def test_category(self, factor):
        assert factor.category == "高频成交分布"

    def test_repr(self, factor):
        assert "PEAK_RIDGE_CORR" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "PEAK_RIDGE_CORR"


class TestPeakRidgeCorrCompute:
    def test_precomputed_passthrough(self, factor):
        """预计算模式直接返回。"""
        dates = pd.date_range("2024-01-01", periods=5)
        df = pd.DataFrame({"A": [0.5, 0.6, 0.7, 0.8, 0.9]}, index=dates)
        dummy = pd.DataFrame({"A": [0.0] * 5}, index=dates)

        result = factor.compute(daily_peak_counts=df, daily_ridge_counts=dummy)
        pd.testing.assert_frame_equal(result, df)

    def test_output_range(self, factor):
        """相关系数应在 [-1, 1]。"""
        np.random.seed(42)
        dates = ["2024-01-01"]
        minutes = list(range(50))
        idx = pd.MultiIndex.from_product([dates, minutes])
        peaks = pd.DataFrame({"A": np.random.randint(0, 5, 50).astype(float)}, index=idx)
        ridges = pd.DataFrame({"A": np.random.randint(0, 5, 50).astype(float)}, index=idx)

        result = factor.compute(daily_peak_counts=peaks, daily_ridge_counts=ridges)
        val = result.iloc[0, 0]
        if not np.isnan(val):
            assert -1.0 - 1e-10 <= val <= 1.0 + 1e-10

    def test_perfect_correlation(self, factor):
        """峰数和岭数完全正相关时，相关系数=1。"""
        dates = ["2024-01-01"]
        minutes = list(range(20))
        idx = pd.MultiIndex.from_product([dates, minutes])
        vals = np.arange(20, dtype=float)
        peaks = pd.DataFrame({"A": vals}, index=idx)
        ridges = pd.DataFrame({"A": vals * 2 + 1}, index=idx)

        result = factor.compute(daily_peak_counts=peaks, daily_ridge_counts=ridges)
        assert result.iloc[0, 0] == pytest.approx(1.0, abs=1e-6)


class TestPeakRidgeCorrEdgeCases:
    def test_constant_input(self, factor):
        """常数输入时相关系数为 NaN。"""
        dates = ["2024-01-01"]
        minutes = list(range(10))
        idx = pd.MultiIndex.from_product([dates, minutes])
        peaks = pd.DataFrame({"A": [5.0] * 10}, index=idx)
        ridges = pd.DataFrame({"A": [3.0] * 10}, index=idx)

        result = factor.compute(daily_peak_counts=peaks, daily_ridge_counts=ridges)
        assert np.isnan(result.iloc[0, 0])

    def test_too_few_points(self, factor):
        """数据点不足时结果为 NaN。"""
        dates = ["2024-01-01"]
        minutes = [0, 1]
        idx = pd.MultiIndex.from_product([dates, minutes])
        peaks = pd.DataFrame({"A": [1.0, 2.0]}, index=idx)
        ridges = pd.DataFrame({"A": [3.0, np.nan]}, index=idx)

        result = factor.compute(daily_peak_counts=peaks, daily_ridge_counts=ridges)
        assert np.isnan(result.iloc[0, 0])


class TestPeakRidgeCorrOutputShape:
    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=3)
        df = pd.DataFrame({"A": [0.5, 0.6, 0.7]}, index=dates)
        result = factor.compute(daily_peak_counts=df, daily_ridge_counts=df)
        assert isinstance(result, pd.DataFrame)

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=5)
        stocks = ["A", "B"]
        df = pd.DataFrame(np.random.rand(5, 2), index=dates, columns=stocks)
        result = factor.compute(daily_peak_counts=df, daily_ridge_counts=df)
        assert result.shape == (5, 2)
