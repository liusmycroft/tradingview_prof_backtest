import numpy as np
import pandas as pd
import pytest

from factors.consistent_buy_trade import ConsistentBuyTradeFactor


@pytest.fixture
def factor():
    return ConsistentBuyTradeFactor()


class TestConsistentBuyTradeMetadata:
    def test_name(self, factor):
        assert factor.name == "CONSISTENT_BUY_TRADE"

    def test_category(self, factor):
        assert factor.category == "高频成交分布"

    def test_repr(self, factor):
        assert "CONSISTENT_BUY_TRADE" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "CONSISTENT_BUY_TRADE"
        assert meta["category"] == "高频成交分布"


class TestConsistentBuyTradeCompute:
    def test_constant_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        daily = pd.DataFrame(0.3, index=dates, columns=["A"])

        result = factor.compute(daily_consistent_buy_ratio=daily, d=20)
        np.testing.assert_array_almost_equal(result["A"].values, 0.3)

    def test_rolling_mean(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        daily = pd.DataFrame([0.1, 0.2, 0.3, 0.4, 0.5], index=dates, columns=["A"])

        result = factor.compute(daily_consistent_buy_ratio=daily, d=3)
        assert result.iloc[0, 0] == pytest.approx(0.1)
        assert result.iloc[1, 0] == pytest.approx(0.15)
        assert result.iloc[2, 0] == pytest.approx(0.2)
        assert result.iloc[3, 0] == pytest.approx(0.3)
        assert result.iloc[4, 0] == pytest.approx(0.4)

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        daily = pd.DataFrame(np.random.rand(30, 2), index=dates, columns=stocks)

        result = factor.compute(daily_consistent_buy_ratio=daily, d=20)
        assert result.shape == daily.shape


class TestConsistentBuyTradeDaily:
    def test_all_solid_rise(self):
        """所有K线都是上涨实体K线时，占比应为1。"""
        bar_open = pd.Series([10.0] * 48)
        bar_close = pd.Series([11.0] * 48)
        bar_high = pd.Series([11.0] * 48)
        bar_low = pd.Series([10.0] * 48)
        bar_volume = pd.Series([1000.0] * 48)

        result = ConsistentBuyTradeFactor.compute_daily(
            bar_open, bar_close, bar_high, bar_low, bar_volume, alpha=0.5
        )
        assert result == pytest.approx(1.0)

    def test_no_solid_bars(self):
        """所有K线都是十字星(open==close)时，body=0，不满足body>alpha*shadow。"""
        bar_open = pd.Series([10.0] * 48)
        bar_close = pd.Series([10.0] * 48)
        bar_high = pd.Series([11.0] * 48)
        bar_low = pd.Series([9.0] * 48)
        bar_volume = pd.Series([1000.0] * 48)

        result = ConsistentBuyTradeFactor.compute_daily(
            bar_open, bar_close, bar_high, bar_low, bar_volume, alpha=0.5
        )
        assert result == pytest.approx(0.0)

    def test_zero_volume(self):
        bar_open = pd.Series([10.0] * 10)
        bar_close = pd.Series([11.0] * 10)
        bar_high = pd.Series([11.0] * 10)
        bar_low = pd.Series([10.0] * 10)
        bar_volume = pd.Series([0.0] * 10)

        result = ConsistentBuyTradeFactor.compute_daily(
            bar_open, bar_close, bar_high, bar_low, bar_volume, alpha=0.5
        )
        assert np.isnan(result)

    def test_mixed_bars(self):
        """混合K线：一半上涨实体，一半下跌实体。"""
        bar_open = pd.Series([10.0, 11.0, 10.0, 11.0])
        bar_close = pd.Series([11.0, 10.0, 11.0, 10.0])
        bar_high = pd.Series([11.0, 11.0, 11.0, 11.0])
        bar_low = pd.Series([10.0, 10.0, 10.0, 10.0])
        bar_volume = pd.Series([1000.0, 1000.0, 1000.0, 1000.0])

        result = ConsistentBuyTradeFactor.compute_daily(
            bar_open, bar_close, bar_high, bar_low, bar_volume, alpha=0.5
        )
        # 上涨实体K线: 第0和第2根, 占比 = 2000/4000 = 0.5
        assert result == pytest.approx(0.5)


class TestConsistentBuyTradeEdgeCases:
    def test_single_value(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        daily = pd.DataFrame([0.25], index=dates, columns=["A"])

        result = factor.compute(daily_consistent_buy_ratio=daily, d=20)
        assert result.iloc[0, 0] == pytest.approx(0.25)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        daily = pd.DataFrame(np.nan, index=dates, columns=["A"])

        result = factor.compute(daily_consistent_buy_ratio=daily, d=5)
        assert result.isna().all().all()
