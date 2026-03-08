import numpy as np
import pandas as pd
import pytest

from factors.consistent_sell_trade import ConsistentSellTradeFactor


@pytest.fixture
def factor():
    return ConsistentSellTradeFactor()


class TestConsistentSellTradeMetadata:
    def test_name(self, factor):
        assert factor.name == "CONSISTENT_SELL_TRADE"

    def test_category(self, factor):
        assert factor.category == "高频因子-成交分布类"

    def test_repr(self, factor):
        assert "CONSISTENT_SELL_TRADE" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "CONSISTENT_SELL_TRADE"


class TestConsistentSellTradeCompute:
    def test_constant_input(self, factor):
        """常数输入时均值等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(0.3, index=dates, columns=stocks)

        result = factor.compute(daily_consistent_sell_ratio=daily, d=20)
        np.testing.assert_array_almost_equal(result["A"].values, 0.3)

    def test_rolling_mean(self, factor):
        """验证滚动均值。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame([0.1, 0.2, 0.3, 0.4, 0.5], index=dates, columns=stocks)

        result = factor.compute(daily_consistent_sell_ratio=daily, d=3)
        assert result.iloc[2, 0] == pytest.approx(0.2, rel=1e-6)
        assert result.iloc[4, 0] == pytest.approx(0.4, rel=1e-6)

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        daily = pd.DataFrame(np.random.rand(30, 2), index=dates, columns=stocks)

        result = factor.compute(daily_consistent_sell_ratio=daily, d=20)
        assert result.shape == daily.shape
        assert isinstance(result, pd.DataFrame)

    def test_min_periods_1(self, factor):
        """min_periods=1, 第一行就有值。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame([0.1, 0.2, 0.3, 0.4, 0.5], index=dates, columns=stocks)

        result = factor.compute(daily_consistent_sell_ratio=daily, d=20)
        assert result.iloc[0].notna().all()
