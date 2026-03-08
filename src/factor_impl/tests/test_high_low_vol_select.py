import numpy as np
import pandas as pd
import pytest

from factors.high_low_vol_select import HighLowVolSelectFactor


@pytest.fixture
def factor():
    return HighLowVolSelectFactor()


class TestHighLowVolSelectMetadata:
    def test_name(self, factor):
        assert factor.name == "HIGH_LOW_VOL_SELECT"

    def test_category(self, factor):
        assert factor.category == "高频波动跳跃"

    def test_repr(self, factor):
        assert "HIGH_LOW_VOL_SELECT" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "HIGH_LOW_VOL_SELECT"


class TestHighLowVolSelectCompute:
    def test_precomputed_passthrough(self, factor):
        """预计算模式直接返回。"""
        dates = pd.date_range("2024-01-01", periods=5)
        df = pd.DataFrame({"A": [0.1, 0.2, 0.3, 0.4, 0.5]}, index=dates)
        dummy = pd.DataFrame({"A": [0.0] * 5}, index=dates)

        result = factor.compute(minute_close=df, minute_ret_std=dummy)
        pd.testing.assert_frame_equal(result, df)

    def test_uniform_volatility(self, factor):
        """波动率均匀时，Factor_Diff 应接近 0。"""
        np.random.seed(42)
        T = 20
        n_minutes = 48
        dates = [f"2024-01-{d+1:02d}" for d in range(T)]
        minutes = list(range(n_minutes))
        idx = pd.MultiIndex.from_product([dates, minutes])

        # 价格递增但波动率恒定
        prices = np.tile(np.linspace(10, 20, n_minutes), T)
        stds = np.ones(T * n_minutes) * 0.01

        close = pd.DataFrame({"A": prices}, index=idx)
        ret_std = pd.DataFrame({"A": stds}, index=idx)

        result = factor.compute(minute_close=close, minute_ret_std=ret_std, T=T)
        # 波动率恒定 -> avg_std_low == avg_std_high == avg_std_total -> diff = 0
        assert result.iloc[-1, 0] == pytest.approx(0.0, abs=1e-6)


class TestHighLowVolSelectEdgeCases:
    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=3)
        df = pd.DataFrame({"A": [0.1, 0.2, 0.3]}, index=dates)
        result = factor.compute(minute_close=df, minute_ret_std=df)
        assert isinstance(result, pd.DataFrame)


class TestHighLowVolSelectOutputShape:
    def test_precomputed_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=10)
        stocks = ["A", "B"]
        df = pd.DataFrame(np.random.rand(10, 2), index=dates, columns=stocks)
        result = factor.compute(minute_close=df, minute_ret_std=df)
        assert result.shape == (10, 2)
