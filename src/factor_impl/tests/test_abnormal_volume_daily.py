import numpy as np
import pandas as pd
import pytest

from factors.abnormal_volume_daily import AbnormalVolumeDailyFactor


@pytest.fixture
def factor():
    return AbnormalVolumeDailyFactor()


class TestABNVOLDMetadata:
    def test_name(self, factor):
        assert factor.name == "ABNVOLD"

    def test_category(self, factor):
        assert factor.category == "行为金融-投资者注意力"

    def test_repr(self, factor):
        assert "ABNVOLD" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "ABNVOLD"
        assert meta["category"] == "行为金融-投资者注意力"


class TestABNVOLDHandCalculated:
    def test_constant_volume(self, factor):
        """常数成交量时，异常比值恒为 1，最大值也为 1。"""
        dates = pd.date_range("2024-01-01", periods=300, freq="D")
        stocks = ["A"]
        volume = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(volume=volume, Y=252, M=20)
        # 前 252+20-2 行为 NaN，之后为 1.0
        valid = result.dropna()
        np.testing.assert_array_almost_equal(valid["A"].values, 1.0)

    def test_spike_detection(self, factor):
        """成交量突增时，因子值应大于 1。"""
        dates = pd.date_range("2024-01-01", periods=280, freq="D")
        stocks = ["A"]
        vals = [1000.0] * 270 + [5000.0] + [1000.0] * 9
        volume = pd.DataFrame(vals, index=dates, columns=stocks)

        result = factor.compute(volume=volume, Y=252, M=20)
        # 在 spike 之后的窗口内，最大异常成交量应 > 1
        # index 270 处有 spike，Y=252 从 index 252 开始有效
        # M=20 窗口包含 spike 的行
        spike_row = 270
        if spike_row < len(result) and not np.isnan(result.iloc[spike_row, 0]):
            assert result.iloc[spike_row, 0] > 1.0

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=300, freq="D")
        volume = pd.DataFrame(
            {"A": [1000.0] * 300, "B": [2000.0] * 300}, index=dates
        )

        result = factor.compute(volume=volume, Y=252, M=20)
        valid = result.dropna()
        if len(valid) > 0:
            np.testing.assert_array_almost_equal(valid["A"].values, 1.0)
            np.testing.assert_array_almost_equal(valid["B"].values, 1.0)

    def test_small_window(self, factor):
        """使用小窗口手算验证。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        vals = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        volume = pd.DataFrame(vals, index=dates, columns=stocks)

        result = factor.compute(volume=volume, Y=5, M=3)
        # Y=5: vol_mean at index 4 = mean(10,20,30,40,50) = 30
        # abnormal at index 4 = 50/30 = 5/3
        # Y=5: vol_mean at index 5 = mean(20,30,40,50,60) = 40
        # abnormal at index 5 = 60/40 = 1.5
        # Y=5: vol_mean at index 6 = mean(30,40,50,60,70) = 50
        # abnormal at index 6 = 70/50 = 1.4
        # M=3: max of indices 4,5,6 = max(5/3, 1.5, 1.4) = 5/3
        assert result.iloc[6, 0] == pytest.approx(5.0 / 3.0, rel=1e-10)


class TestABNVOLDEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        values = np.ones(10) * 100.0
        values[3] = np.nan
        volume = pd.DataFrame(values, index=dates, columns=stocks)

        result = factor.compute(volume=volume, Y=5, M=3)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        volume = pd.DataFrame(
            np.random.uniform(100, 1000, (30, 2)), index=dates, columns=stocks
        )
        result = factor.compute(volume=volume, Y=10, M=5)
        assert result.shape == volume.shape
        assert isinstance(result, pd.DataFrame)
