import numpy as np
import pandas as pd
import pytest

from factors.volume_surge_vol import VolumeSurgeVolFactor


@pytest.fixture
def factor():
    return VolumeSurgeVolFactor()


class TestVolumeSurgeVolMetadata:
    def test_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "VOLUME_SURGE_VOL"
        assert meta["category"] == "高频波动"
        assert "量涌" in meta["description"] or "波动" in meta["description"]

    def test_repr(self, factor):
        r = repr(factor)
        assert "VolumeSurgeVolFactor" in r
        assert "VOLUME_SURGE_VOL" in r


class TestVolumeSurgeVolCompute:
    def test_constant_input_zero_std(self, factor):
        """常数输入截面标准化后为 0，滚动 std 也为 0。"""
        dates = pd.bdate_range("2025-01-01", periods=25)
        # 多只股票，每日值相同 -> 截面标准化后全为 0
        daily = pd.DataFrame(
            {"A": [1.0] * 25, "B": [1.0] * 25, "C": [1.0] * 25},
            index=dates,
        )

        result = factor.compute(daily_segment_vol=daily, T=20)

        # 截面标准化: (1-1)/0 -> NaN (std=0)
        # 所以结果应为 NaN
        assert result.iloc[-1].isna().all()

    def test_output_shape(self, factor):
        """输出形状应与输入一致。"""
        np.random.seed(42)
        dates = pd.bdate_range("2025-01-01", periods=25)
        daily = pd.DataFrame(
            np.random.rand(25, 3) * 0.05,
            index=dates,
            columns=["A", "B", "C"],
        )

        result = factor.compute(daily_segment_vol=daily, T=20)

        assert result.shape == (25, 3)

    def test_min_periods(self, factor):
        """窗口不足 T 天时，结果应为 NaN。"""
        np.random.seed(42)
        dates = pd.bdate_range("2025-01-01", periods=10)
        daily = pd.DataFrame(
            np.random.rand(10, 3) * 0.05,
            index=dates,
            columns=["A", "B", "C"],
        )

        result = factor.compute(daily_segment_vol=daily, T=20)

        assert result.isna().all().all()

    def test_standardization(self, factor):
        """验证截面标准化的正确性。"""
        np.random.seed(42)
        dates = pd.bdate_range("2025-01-01", periods=25)
        daily = pd.DataFrame(
            np.random.rand(25, 5) * 0.05,
            index=dates,
            columns=["A", "B", "C", "D", "E"],
        )

        result = factor.compute(daily_segment_vol=daily, T=20)

        # 手动计算截面标准化
        cross_mean = daily.mean(axis=1)
        cross_std = daily.std(axis=1, ddof=1)
        standardized = daily.sub(cross_mean, axis=0).div(cross_std, axis=0)
        expected = standardized.rolling(window=20, min_periods=20).std(ddof=1)

        pd.testing.assert_frame_equal(result, expected)

    def test_non_negative_std(self, factor):
        """标准差应非负。"""
        np.random.seed(42)
        dates = pd.bdate_range("2025-01-01", periods=25)
        daily = pd.DataFrame(
            np.random.rand(25, 3) * 0.05,
            index=dates,
            columns=["A", "B", "C"],
        )

        result = factor.compute(daily_segment_vol=daily, T=20)

        valid = result.dropna()
        assert (valid >= 0).all().all()

    def test_single_stock_nan(self, factor):
        """单只股票时，截面标准差为 NaN，结果应为 NaN。"""
        dates = pd.bdate_range("2025-01-01", periods=25)
        daily = pd.DataFrame({"A": np.random.rand(25) * 0.05}, index=dates)

        result = factor.compute(daily_segment_vol=daily, T=20)

        # 单只股票截面 std = NaN -> 标准化后 NaN -> 滚动 std 也 NaN
        assert result.isna().all().all()
