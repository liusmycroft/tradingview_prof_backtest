import numpy as np
import pandas as pd
import pytest

from factors.dazzling_return import DazzlingReturnFactor


@pytest.fixture
def factor():
    return DazzlingReturnFactor()


class TestDazzlingReturnMetadata:
    def test_name(self, factor):
        assert factor.name == "DAZZLING_RETURN"

    def test_category(self, factor):
        assert factor.category == "高频因子-动量反转类"

    def test_repr(self, factor):
        assert "DAZZLING_RETURN" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "DAZZLING_RETURN"


class TestDazzlingReturnCompute:
    def test_constant_input(self, factor):
        """常数输入时，适度日耀眼收益率为 0，因子也为 0。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A", "B"]
        daily = pd.DataFrame(0.01, index=dates, columns=stocks)

        result = factor.compute(daily_dazzling_return=daily, T=20)
        # 横截面均值 = 0.01, |0.01 - 0.01| = 0
        np.testing.assert_array_almost_equal(result.values, 0.0)

    def test_positive_output(self, factor):
        """不同股票有不同收益时，因子应为正。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A", "B"]
        daily = pd.DataFrame({"A": [0.01] * 20, "B": [0.05] * 20}, index=dates)

        result = factor.compute(daily_dazzling_return=daily, T=20)
        assert (result.iloc[-1] > 0).all()

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily = pd.DataFrame(np.random.randn(30, 3) * 0.01, index=dates, columns=stocks)

        result = factor.compute(daily_dazzling_return=daily, T=20)
        assert result.shape == daily.shape
        assert isinstance(result, pd.DataFrame)
