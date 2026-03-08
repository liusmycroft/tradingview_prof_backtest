"""毒流动性因子测试"""

import numpy as np
import pandas as pd
import pytest
from factors.toxic_liquidity import ToxicLiquidityFactor


@pytest.fixture
def factor():
    return ToxicLiquidityFactor()


@pytest.fixture
def sample_data():
    dates = pd.date_range("2024-01-01", periods=25)
    stocks = ["000001", "000002"]
    daily_toxic = pd.DataFrame(
        np.full((25, 2), 0.5), index=dates, columns=stocks
    )
    return daily_toxic


class TestMetadata:
    def test_name(self, factor):
        assert factor.name == "TOXIC_LIQUIDITY"

    def test_category(self, factor):
        assert factor.category == "高频流动性"

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "TOXIC_LIQUIDITY"

    def test_repr(self, factor):
        assert "ToxicLiquidityFactor" in repr(factor)


class TestCompute:
    def test_constant_values(self, factor, sample_data):
        result = factor.compute(daily_toxic=sample_data)
        # Rolling mean of constant 0.5 = 0.5
        assert pytest.approx(result.iloc[-1, 0], rel=1e-6) == 0.5

    def test_output_shape(self, factor, sample_data):
        result = factor.compute(daily_toxic=sample_data)
        assert result.shape == sample_data.shape

    def test_known_rolling_mean(self, factor):
        dates = pd.date_range("2024-01-01", periods=5)
        stocks = ["A"]
        data = pd.DataFrame([[0.2], [0.4], [0.6], [0.8], [1.0]], index=dates, columns=stocks)
        result = factor.compute(daily_toxic=data, T=3)
        # T=3: last value = mean(0.6, 0.8, 1.0) = 0.8
        assert pytest.approx(result.iloc[-1, 0], rel=1e-6) == 0.8
        # Second value: mean(0.2, 0.4) with min_periods=1 = 0.3
        assert pytest.approx(result.iloc[1, 0], rel=1e-6) == 0.3

    def test_custom_window(self, factor):
        dates = pd.date_range("2024-01-01", periods=10)
        stocks = ["A"]
        data = pd.DataFrame(np.ones((10, 1)) * 0.7, index=dates, columns=stocks)
        result = factor.compute(daily_toxic=data, T=5)
        assert pytest.approx(result.iloc[-1, 0], rel=1e-6) == 0.7

    def test_nan_handling(self, factor):
        dates = pd.date_range("2024-01-01", periods=5)
        stocks = ["A"]
        data = pd.DataFrame([[0.5], [np.nan], [0.5], [0.5], [0.5]], index=dates, columns=stocks)
        result = factor.compute(daily_toxic=data, T=3)
        # Window [nan, 0.5, 0.5] -> mean ignoring nan = 0.5
        assert not np.isnan(result.iloc[0, 0])
