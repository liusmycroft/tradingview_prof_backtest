import numpy as np
import pandas as pd
import pytest

from factors.corrected_amplitude import CorrectedAmplitudeFactor


@pytest.fixture
def factor():
    return CorrectedAmplitudeFactor()


class TestCorrectedAmplitudeMetadata:
    def test_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "CORRECTED_AMPLITUDE"
        assert meta["category"] == "高频波动"
        assert "振幅" in meta["description"]

    def test_repr(self, factor):
        r = repr(factor)
        assert "CorrectedAmplitudeFactor" in r
        assert "CORRECTED_AMPLITUDE" in r


class TestCorrectedAmplitudeCompute:
    def test_known_values_no_flip(self, factor):
        """jump_drop 均高于截面均值时，振幅不翻转。"""
        dates = pd.bdate_range("2025-01-01", periods=2)
        high = pd.DataFrame({"A": [11.0, 12.0]}, index=dates)
        low = pd.DataFrame({"A": [9.0, 10.0]}, index=dates)
        close = pd.DataFrame({"A": [10.0, 11.0]}, index=dates)
        # 只有一只股票，jump_drop == 截面均值，不翻转（lt 为 strict <）
        jump_drop = pd.DataFrame({"A": [0.01, 0.02]}, index=dates)

        result = factor.compute(
            high=high, low=low, close=close,
            daily_jump_drop=jump_drop, T=1,
        )

        # day1: amplitude = (12-10)/10 = 0.2, jump_drop=0.02 == mean(0.02), 不翻转
        # T=1, result = 0.2
        np.testing.assert_almost_equal(result.iloc[1, 0], 0.2)

    def test_flip_sign(self, factor):
        """jump_drop 低于截面均值时，振幅符号翻转。"""
        dates = pd.bdate_range("2025-01-01", periods=2)
        high = pd.DataFrame({"A": [11.0, 12.0], "B": [11.0, 12.0]}, index=dates)
        low = pd.DataFrame({"A": [9.0, 10.0], "B": [9.0, 10.0]}, index=dates)
        close = pd.DataFrame({"A": [10.0, 11.0], "B": [10.0, 11.0]}, index=dates)
        # A 的 jump_drop 低于截面均值，B 的高于
        jump_drop = pd.DataFrame({"A": [-0.05, -0.05], "B": [0.05, 0.05]}, index=dates)

        result = factor.compute(
            high=high, low=low, close=close,
            daily_jump_drop=jump_drop, T=1,
        )

        # day1: amplitude = (12-10)/10 = 0.2 for both
        # cross_mean = mean(-0.05, 0.05) = 0.0
        # A: jump_drop=-0.05 < 0.0 -> flip -> -0.2
        # B: jump_drop=0.05 >= 0.0 -> no flip -> 0.2
        np.testing.assert_almost_equal(result.loc[dates[1], "A"], -0.2)
        np.testing.assert_almost_equal(result.loc[dates[1], "B"], 0.2)

    def test_prev_close_shift(self, factor):
        """第一行应为 NaN（无前一日收盘价）。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        high = pd.DataFrame({"A": [11.0, 12.0, 13.0]}, index=dates)
        low = pd.DataFrame({"A": [9.0, 10.0, 11.0]}, index=dates)
        close = pd.DataFrame({"A": [10.0, 11.0, 12.0]}, index=dates)
        jump_drop = pd.DataFrame({"A": [0.01, 0.02, 0.03]}, index=dates)

        result = factor.compute(
            high=high, low=low, close=close,
            daily_jump_drop=jump_drop, T=1,
        )

        # 第一行 amplitude 使用 prev_close=NaN -> NaN
        assert np.isnan(result.iloc[0, 0])

    def test_rolling_mean(self, factor):
        """验证 T 日滚动均值。"""
        dates = pd.bdate_range("2025-01-01", periods=4)
        high = pd.DataFrame({"A": [11.0, 12.0, 13.0, 14.0]}, index=dates)
        low = pd.DataFrame({"A": [9.0, 10.0, 11.0, 12.0]}, index=dates)
        close = pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0]}, index=dates)
        # 单只股票，jump_drop == 截面均值，不翻转
        jump_drop = pd.DataFrame({"A": [0.01, 0.02, 0.03, 0.04]}, index=dates)

        result = factor.compute(
            high=high, low=low, close=close,
            daily_jump_drop=jump_drop, T=2,
        )

        # day1: amp = (12-10)/10 = 0.2
        # day2: amp = (13-11)/11 ≈ 0.18182
        # day3: amp = (14-12)/12 ≈ 0.16667
        # T=2 rolling mean at day2: (0.2 + 0.18182)/2
        expected_day2 = (2 / 10 + 2 / 11) / 2
        np.testing.assert_almost_equal(result.iloc[2, 0], expected_day2)

    def test_output_shape(self, factor):
        """输出形状应与输入一致。"""
        np.random.seed(42)
        dates = pd.bdate_range("2025-01-01", periods=25)
        high = pd.DataFrame(
            np.random.uniform(11, 13, (25, 2)), index=dates, columns=["A", "B"]
        )
        low = pd.DataFrame(
            np.random.uniform(9, 11, (25, 2)), index=dates, columns=["A", "B"]
        )
        close = pd.DataFrame(
            np.random.uniform(10, 12, (25, 2)), index=dates, columns=["A", "B"]
        )
        jump_drop = pd.DataFrame(
            np.random.randn(25, 2) * 0.01, index=dates, columns=["A", "B"]
        )

        result = factor.compute(
            high=high, low=low, close=close,
            daily_jump_drop=jump_drop, T=20,
        )

        assert result.shape == (25, 2)
