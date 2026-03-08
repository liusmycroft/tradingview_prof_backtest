import numpy as np
import pandas as pd
import pytest

from factors.ideal_amplitude import IdealAmplitudeFactor


@pytest.fixture
def factor():
    return IdealAmplitudeFactor()


class TestIdealAmplitudeMetadata:
    def test_name(self, factor):
        assert factor.name == "IDEAL_AMPLITUDE"

    def test_category(self, factor):
        assert factor.category == "高频波动"

    def test_repr(self, factor):
        assert "IDEAL_AMPLITUDE" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "IDEAL_AMPLITUDE"
        assert meta["category"] == "高频波动"


class TestIdealAmplitudeHandCalculated:
    def test_simple_N4(self, factor):
        """N=4, quantile=0.25, 手算验证。

        close = [10, 20, 30, 40]
        high  = [11, 22, 33, 44]
        low   = [9,  18, 27, 36]

        amplitude = high/low - 1:
          day0: 11/9 - 1 = 0.2222...
          day1: 22/18 - 1 = 0.2222...
          day2: 33/27 - 1 = 0.2222...
          day3: 44/36 - 1 = 0.2222...

        所有振幅相同，V_high = V_low = 0.2222..., V(lambda) = 0
        """
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        stocks = ["A"]

        close = pd.DataFrame([10.0, 20.0, 30.0, 40.0], index=dates, columns=stocks)
        high = pd.DataFrame([11.0, 22.0, 33.0, 44.0], index=dates, columns=stocks)
        low = pd.DataFrame([9.0, 18.0, 27.0, 36.0], index=dates, columns=stocks)

        result = factor.compute(close=close, high=high, low=low, N=4, quantile=0.25)
        assert result.iloc[3, 0] == pytest.approx(0.0, abs=1e-10)

    def test_different_amplitudes(self, factor):
        """不同振幅时验证高低分组。

        N=4, quantile=0.25
        close = [10, 20, 30, 40]  -> 排序后 [10,20,30,40]
        25% 分位 = 17.5, 75% 分位 = 32.5
        低价日: close <= 17.5 -> day0 (close=10)
        高价日: close >= 32.5 -> day3 (close=40)

        high  = [12, 22, 33, 48]
        low   = [8,  18, 27, 32]

        amplitude:
          day0: 12/8 - 1 = 0.5
          day1: 22/18 - 1 = 0.2222...
          day2: 33/27 - 1 = 0.2222...
          day3: 48/32 - 1 = 0.5

        V_high = amp[day3] = 0.5
        V_low = amp[day0] = 0.5
        V(lambda) = 0.5 - 0.5 = 0.0
        """
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        stocks = ["A"]

        close = pd.DataFrame([10.0, 20.0, 30.0, 40.0], index=dates, columns=stocks)
        high = pd.DataFrame([12.0, 22.0, 33.0, 48.0], index=dates, columns=stocks)
        low = pd.DataFrame([8.0, 18.0, 27.0, 32.0], index=dates, columns=stocks)

        result = factor.compute(close=close, high=high, low=low, N=4, quantile=0.25)
        assert result.iloc[3, 0] == pytest.approx(0.0, abs=1e-10)

    def test_asymmetric_amplitudes(self, factor):
        """高价日振幅大于低价日振幅时，因子值为正。

        N=4, quantile=0.25
        close = [10, 20, 30, 40]
        25% 分位 = 17.5, 75% 分位 = 32.5

        低价日: day0 (close=10), amp = 15/8 - 1 = 0.875
        高价日: day3 (close=40), amp = 50/30 - 1 = 0.6667

        V_high - V_low = 0.6667 - 0.875 = -0.2083...
        """
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        stocks = ["A"]

        close = pd.DataFrame([10.0, 20.0, 30.0, 40.0], index=dates, columns=stocks)
        high = pd.DataFrame([15.0, 22.0, 33.0, 50.0], index=dates, columns=stocks)
        low = pd.DataFrame([8.0, 18.0, 27.0, 30.0], index=dates, columns=stocks)

        result = factor.compute(close=close, high=high, low=low, N=4, quantile=0.25)

        v_high = 50.0 / 30.0 - 1.0
        v_low = 15.0 / 8.0 - 1.0
        expected = v_high - v_low
        assert result.iloc[3, 0] == pytest.approx(expected, rel=1e-6)


class TestIdealAmplitudeEdgeCases:
    def test_nan_in_data(self, factor):
        """含 NaN 时不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        stocks = ["A"]

        close = pd.DataFrame([np.nan, 20.0, 30.0, 40.0], index=dates, columns=stocks)
        high = pd.DataFrame([11.0, 22.0, 33.0, 44.0], index=dates, columns=stocks)
        low = pd.DataFrame([9.0, 18.0, 27.0, 36.0], index=dates, columns=stocks)

        result = factor.compute(close=close, high=high, low=low, N=4, quantile=0.25)
        assert isinstance(result, pd.DataFrame)

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        close = pd.DataFrame(
            np.random.uniform(10, 50, (30, 2)), index=dates, columns=stocks
        )
        high = close + np.random.uniform(0.5, 2, (30, 2))
        low = close - np.random.uniform(0.5, 2, (30, 2))
        low = low.clip(lower=0.01)

        result = factor.compute(close=close, high=high, low=low, N=20)
        assert result.shape == close.shape
        assert isinstance(result, pd.DataFrame)

    def test_first_N_minus_1_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        close = pd.DataFrame(np.arange(10, 20, dtype=float), index=dates, columns=stocks)
        high = close + 1
        low = close - 1
        low = low.clip(lower=0.01)
        N = 5
        result = factor.compute(close=close, high=high, low=low, N=N)
        assert result.iloc[: N - 1].isna().all().all()
