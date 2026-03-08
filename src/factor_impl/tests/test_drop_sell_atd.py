"""跌幅最大时刻-主卖笔均成交金额因子测试"""

import numpy as np
import pandas as pd
import pytest
from factors.drop_sell_atd import DropSellATDFactor


@pytest.fixture
def factor():
    return DropSellATDFactor()


@pytest.fixture
def sample_data():
    dates = pd.date_range("2024-01-01", periods=25)
    stocks = ["000001", "000002"]
    daily_satd = pd.DataFrame(
        np.full((25, 2), 0.15), index=dates, columns=stocks
    )
    return daily_satd


class TestMetadata:
    def test_name(self, factor):
        assert factor.name == "DROP_SELL_ATD"

    def test_category(self, factor):
        assert factor.category == "高频成交分布"

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "DROP_SELL_ATD"

    def test_repr(self, factor):
        assert "DropSellATDFactor" in repr(factor)


class TestCompute:
    def test_constant_values(self, factor, sample_data):
        result = factor.compute(daily_satd=sample_data)
        assert pytest.approx(result.iloc[-1, 0], rel=1e-6) == 0.15

    def test_output_shape(self, factor, sample_data):
        result = factor.compute(daily_satd=sample_data)
        assert result.shape == sample_data.shape

    def test_known_rolling_mean(self, factor):
        dates = pd.date_range("2024-01-01", periods=5)
        stocks = ["A"]
        data = pd.DataFrame([[0.1], [0.2], [0.3], [0.4], [0.5]], index=dates, columns=stocks)
        result = factor.compute(daily_satd=data, T=3)
        # Last: mean(0.3, 0.4, 0.5) = 0.4
        assert pytest.approx(result.iloc[-1, 0], rel=1e-6) == 0.4

    def test_custom_window(self, factor):
        dates = pd.date_range("2024-01-01", periods=5)
        stocks = ["A"]
        data = pd.DataFrame([[0.1], [0.2], [0.3], [0.4], [0.5]], index=dates, columns=stocks)
        result = factor.compute(daily_satd=data, T=5)
        assert pytest.approx(result.iloc[-1, 0], rel=1e-6) == 0.3

    def test_nan_handling(self, factor):
        dates = pd.date_range("2024-01-01", periods=5)
        stocks = ["A"]
        data = pd.DataFrame([[0.1], [np.nan], [0.3], [0.4], [0.5]], index=dates, columns=stocks)
        result = factor.compute(daily_satd=data, T=3)
        # Window [0.3, 0.4, 0.5] -> 0.4
        assert pytest.approx(result.iloc[-1, 0], rel=1e-6) == 0.4
