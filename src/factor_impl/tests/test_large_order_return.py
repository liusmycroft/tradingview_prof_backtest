"""大单推动涨幅因子测试"""

import numpy as np
import pandas as pd
import pytest
from factors.large_order_return import LargeOrderReturnFactor


@pytest.fixture
def factor():
    return LargeOrderReturnFactor()


@pytest.fixture
def sample_data():
    dates = pd.date_range("2024-01-01", periods=25)
    stocks = ["000001", "000002"]
    daily_large_return = pd.DataFrame(
        np.full((25, 2), 0.01), index=dates, columns=stocks
    )
    return daily_large_return


class TestMetadata:
    def test_name(self, factor):
        assert factor.name == "LARGE_ORDER_RETURN"

    def test_category(self, factor):
        assert factor.category == "高频动量反转"

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "LARGE_ORDER_RETURN"

    def test_repr(self, factor):
        assert "LargeOrderReturnFactor" in repr(factor)


class TestCompute:
    def test_known_values(self, factor):
        dates = pd.date_range("2024-01-01", periods=3)
        stocks = ["A"]
        # 每天收益率0.1
        data = pd.DataFrame([[0.1], [0.1], [0.1]], index=dates, columns=stocks)
        result = factor.compute(daily_large_return=data, T=3)
        # 累计乘积: 1.1^3 = 1.331
        assert pytest.approx(result.iloc[2, 0], rel=1e-4) == 1.1 ** 3

    def test_single_day(self, factor):
        dates = pd.date_range("2024-01-01", periods=1)
        stocks = ["A"]
        data = pd.DataFrame([[0.05]], index=dates, columns=stocks)
        result = factor.compute(daily_large_return=data, T=20)
        assert pytest.approx(result.iloc[0, 0], rel=1e-6) == 1.05

    def test_output_shape(self, factor, sample_data):
        result = factor.compute(daily_large_return=sample_data)
        assert result.shape == sample_data.shape

    def test_zero_return(self, factor):
        dates = pd.date_range("2024-01-01", periods=5)
        stocks = ["A"]
        data = pd.DataFrame(np.zeros((5, 1)), index=dates, columns=stocks)
        result = factor.compute(daily_large_return=data, T=3)
        # product of (1+0) = 1.0
        assert pytest.approx(result.iloc[-1, 0], rel=1e-6) == 1.0

    def test_negative_return(self, factor):
        dates = pd.date_range("2024-01-01", periods=3)
        stocks = ["A"]
        data = pd.DataFrame([[-0.05], [-0.05], [-0.05]], index=dates, columns=stocks)
        result = factor.compute(daily_large_return=data, T=3)
        expected = 0.95 ** 3
        assert pytest.approx(result.iloc[2, 0], rel=1e-4) == expected

    def test_cumulative_product_window(self, factor):
        dates = pd.date_range("2024-01-01", periods=5)
        stocks = ["A"]
        data = pd.DataFrame([[0.1], [0.2], [0.0], [-0.1], [0.05]], index=dates, columns=stocks)
        result = factor.compute(daily_large_return=data, T=3)
        # Window [0.0, -0.1, 0.05]: product = 1.0 * 0.9 * 1.05 = 0.945
        assert pytest.approx(result.iloc[4, 0], rel=1e-4) == 1.0 * 0.9 * 1.05
