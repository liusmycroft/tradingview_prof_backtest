import numpy as np
import pandas as pd
import pytest

from factors.bid_ask_spread import BidAskSpreadFactor


@pytest.fixture
def factor():
    return BidAskSpreadFactor()


class TestBidAskSpreadMetadata:
    def test_name(self, factor):
        assert factor.name == "BID_ASK_SPREAD"

    def test_category(self, factor):
        assert factor.category == "高频因子-流动性类"

    def test_repr(self, factor):
        assert "BID_ASK_SPREAD" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "BID_ASK_SPREAD"


class TestBidAskSpreadCompute:
    def test_constant_input(self, factor):
        """常数输入时 EMA 等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        daily_spread = pd.DataFrame(0.005, index=dates, columns=stocks)

        result = factor.compute(daily_spread=daily_spread, T=20)
        np.testing.assert_array_almost_equal(result["A"].values, 0.005)

    def test_ema_recent_weight(self, factor):
        """EMA 应偏向近期值。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        vals = [0.001] * 5 + [0.01] * 5
        daily_spread = pd.DataFrame(vals, index=dates, columns=stocks)

        result = factor.compute(daily_spread=daily_spread, T=5)
        assert result.iloc[-1, 0] > 0.005

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A", "B"]
        daily_spread = pd.DataFrame(np.random.rand(20, 2) * 0.01, index=dates, columns=stocks)

        result = factor.compute(daily_spread=daily_spread, T=20)
        assert result.shape == daily_spread.shape
        assert isinstance(result, pd.DataFrame)

    def test_single_value(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A"]
        daily_spread = pd.DataFrame([0.003], index=dates, columns=stocks)

        result = factor.compute(daily_spread=daily_spread, T=20)
        assert result.iloc[0, 0] == pytest.approx(0.003, rel=1e-10)
