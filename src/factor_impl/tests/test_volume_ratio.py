import numpy as np
import pandas as pd
import pytest

from factors.volume_ratio import VolumeRatioFactor


@pytest.fixture
def factor():
    return VolumeRatioFactor()


class TestVolumeRatioMetadata:
    def test_name(self, factor):
        assert factor.name == "VR"

    def test_category(self, factor):
        assert factor.category == "高频成交分布"

    def test_repr(self, factor):
        assert "VR" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "VR"
        assert meta["category"] == "高频成交分布"


class TestVolumeRatioHandCalculated:
    def test_constant_input(self, factor):
        """常数输入时，EMA 应等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(1.5, index=dates, columns=stocks)

        result = factor.compute(daily_volume_ratio=data, d=20)
        np.testing.assert_array_almost_equal(result["A"].values, 1.5)

    def test_ema_manual_d3(self, factor):
        """d=3, 手动验证 EMA 值。

        ewm(span=3, adjust=True), alpha = 2/(3+1) = 0.5
        data = [1.0, 2.0, 3.0, 4.0]
        ema_0 = 1.0
        ema_1 = (0.5*1 + 1.0*2) / (0.5+1.0) = 2.5/1.5 = 5/3
        ema_2 = (0.25*1 + 0.5*2 + 1.0*3) / (0.25+0.5+1.0) = 4.25/1.75 = 17/7
        ema_3 = (0.125*1 + 0.25*2 + 0.5*3 + 1.0*4) / (0.125+0.25+0.5+1.0) = 6.125/1.875 = 49/15
        """
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        stocks = ["A"]
        data = pd.DataFrame([1.0, 2.0, 3.0, 4.0], index=dates, columns=stocks)

        result = factor.compute(daily_volume_ratio=data, d=3)

        assert result.iloc[0, 0] == pytest.approx(1.0, rel=1e-6)
        assert result.iloc[1, 0] == pytest.approx(5.0 / 3.0, rel=1e-6)
        assert result.iloc[2, 0] == pytest.approx(17.0 / 7.0, rel=1e-6)
        assert result.iloc[3, 0] == pytest.approx(49.0 / 15.0, rel=1e-6)

    def test_ema_recent_weight(self, factor):
        """EMA 应赋予近期数据更高权重。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        vals = [1.0] * 5 + [3.0] * 5
        data = pd.DataFrame(vals, index=dates, columns=stocks)

        result = factor.compute(daily_volume_ratio=data, d=5)
        assert result.iloc[-1, 0] > 2.0

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(
            {"A": [1.0] * 10, "B": [2.0] * 10}, index=dates
        )

        result = factor.compute(daily_volume_ratio=data, d=5)
        np.testing.assert_array_almost_equal(result["A"].values, 1.0)
        np.testing.assert_array_almost_equal(result["B"].values, 2.0)


class TestVolumeRatioEdgeCases:
    def test_single_value(self, factor):
        """单个数据点的 EMA 应等于该值。"""
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A"]
        data = pd.DataFrame([1.8], index=dates, columns=stocks)

        result = factor.compute(daily_volume_ratio=data, d=20)
        assert result.iloc[0, 0] == pytest.approx(1.8, rel=1e-10)

    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        values = np.ones(10) * 1.5
        values[3] = np.nan
        data = pd.DataFrame(values, index=dates, columns=stocks)

        result = factor.compute(daily_volume_ratio=data, d=5)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_volume_ratio=data, d=5)
        assert result.isna().all().all()

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        data = pd.DataFrame(
            np.random.uniform(0.5, 2.0, (30, 3)), index=dates, columns=stocks
        )
        result = factor.compute(daily_volume_ratio=data, d=20)
        assert result.shape == data.shape
        assert isinstance(result, pd.DataFrame)

    def test_no_leading_nan_min_periods_1(self, factor):
        """min_periods=1, 第一行就有值。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(
            np.random.uniform(0.5, 2.0, (10, 1)), index=dates, columns=stocks
        )
        result = factor.compute(daily_volume_ratio=data, d=20)
        assert result.iloc[0].notna().all()
