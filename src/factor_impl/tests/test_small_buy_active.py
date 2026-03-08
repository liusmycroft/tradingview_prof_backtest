"""小买单主动成交度因子测试"""

import numpy as np
import pandas as pd
import pytest
from factors.small_buy_active import SmallBuyActiveFactor


@pytest.fixture
def factor():
    return SmallBuyActiveFactor()


@pytest.fixture
def sample_data():
    dates = pd.date_range("2024-01-01", periods=25)
    stocks = ["000001", "000002"]
    small_active_buy = pd.DataFrame(
        np.full((25, 2), 200.0), index=dates, columns=stocks
    )
    small_buy_total = pd.DataFrame(
        np.full((25, 2), 1000.0), index=dates, columns=stocks
    )
    return small_active_buy, small_buy_total


class TestMetadata:
    def test_name(self, factor):
        assert factor.name == "SMALL_BUY_ACTIVE"

    def test_category(self, factor):
        assert factor.category == "高频资金流"

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "SMALL_BUY_ACTIVE"

    def test_repr(self, factor):
        assert "SmallBuyActiveFactor" in repr(factor)


class TestCompute:
    def test_known_values(self, factor, sample_data):
        small_active_buy, small_buy_total = sample_data
        result = factor.compute(
            small_active_buy=small_active_buy, small_buy_total=small_buy_total
        )
        # 200/1000 = 0.2, rolling mean of constant = 0.2
        assert pytest.approx(result.iloc[-1, 0], rel=1e-6) == 0.2

    def test_output_shape(self, factor, sample_data):
        small_active_buy, small_buy_total = sample_data
        result = factor.compute(
            small_active_buy=small_active_buy, small_buy_total=small_buy_total
        )
        assert result.shape == small_active_buy.shape

    def test_varying_ratio(self, factor):
        dates = pd.date_range("2024-01-01", periods=5)
        stocks = ["A"]
        active = pd.DataFrame([[100], [200], [300], [400], [500]], index=dates, columns=stocks, dtype=float)
        total = pd.DataFrame(np.full((5, 1), 1000.0), index=dates, columns=stocks)
        result = factor.compute(small_active_buy=active, small_buy_total=total, T=3)
        # Last window: mean(0.3, 0.4, 0.5) = 0.4
        assert pytest.approx(result.iloc[-1, 0], rel=1e-6) == 0.4

    def test_full_active(self, factor):
        """所有买单都是主动成交时，比值为1"""
        dates = pd.date_range("2024-01-01", periods=5)
        stocks = ["A"]
        vol = pd.DataFrame(np.full((5, 1), 500.0), index=dates, columns=stocks)
        result = factor.compute(small_active_buy=vol, small_buy_total=vol)
        assert (result == 1.0).all().all()

    def test_zero_total_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=3)
        stocks = ["A"]
        active = pd.DataFrame(np.full((3, 1), 100.0), index=dates, columns=stocks)
        total = pd.DataFrame(np.zeros((3, 1)), index=dates, columns=stocks)
        result = factor.compute(small_active_buy=active, small_buy_total=total)
        assert np.isinf(result.iloc[0, 0]) or np.isnan(result.iloc[0, 0])

    def test_custom_window(self, factor):
        dates = pd.date_range("2024-01-01", periods=10)
        stocks = ["A"]
        active = pd.DataFrame(np.full((10, 1), 300.0), index=dates, columns=stocks)
        total = pd.DataFrame(np.full((10, 1), 1000.0), index=dates, columns=stocks)
        result = factor.compute(small_active_buy=active, small_buy_total=total, T=5)
        assert pytest.approx(result.iloc[-1, 0], rel=1e-6) == 0.3
