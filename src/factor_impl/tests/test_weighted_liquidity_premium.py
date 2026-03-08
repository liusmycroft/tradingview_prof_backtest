import numpy as np
import pandas as pd
import pytest

from factors.weighted_liquidity_premium import WeightedLiquidityPremiumFactor


@pytest.fixture
def factor():
    return WeightedLiquidityPremiumFactor()


class TestWeightedLiquidityPremiumMetadata:
    def test_name(self, factor):
        assert factor.name == "WEIGHTED_LIQUIDITY_PREMIUM"

    def test_category(self, factor):
        assert factor.category == "高频流动性"

    def test_repr(self, factor):
        assert "WEIGHTED_LIQUIDITY_PREMIUM" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "WEIGHTED_LIQUIDITY_PREMIUM"
        assert meta["category"] == "高频流动性"


class TestWeightedLiquidityPremiumCompute:
    def test_no_premium(self, factor):
        """cap_need == cap_actual 时，溢价为 0。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        cap_need = pd.DataFrame(1000.0, index=dates, columns=stocks)
        cap_actual = pd.DataFrame(1000.0, index=dates, columns=stocks)
        amount = pd.DataFrame(500.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_cap_need=cap_need, daily_cap_actual=cap_actual,
            daily_amount=amount, T=5
        )
        np.testing.assert_array_almost_equal(result["A"].values, 0.0)

    def test_positive_premium(self, factor):
        """cap_need > cap_actual 时，溢价为正。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        cap_need = pd.DataFrame(1100.0, index=dates, columns=stocks)
        cap_actual = pd.DataFrame(1000.0, index=dates, columns=stocks)
        amount = pd.DataFrame(500.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_cap_need=cap_need, daily_cap_actual=cap_actual,
            daily_amount=amount, T=5
        )
        # daily_premium = 0.1, equal weights -> result = 0.1
        assert result.iloc[-1, 0] > 0

    def test_constant_equal_weight(self, factor):
        """等额成交时，手动验证加权滚动求和。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]
        cap_need = pd.DataFrame([1100.0, 1200.0, 1050.0], index=dates, columns=stocks)
        cap_actual = pd.DataFrame(1000.0, index=dates, columns=stocks)
        amount = pd.DataFrame(100.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_cap_need=cap_need, daily_cap_actual=cap_actual,
            daily_amount=amount, T=3
        )
        # daily_premium = [0.1, 0.2, 0.05]
        # t=0: amt_sum=100, w=1.0,  wp=0.1,    roll_sum=0.1
        # t=1: amt_sum=200, w=0.5,  wp=0.1,    roll_sum=0.2
        # t=2: amt_sum=300, w=1/3,  wp=0.05/3, roll_sum=0.1+0.1+0.05/3
        expected = 0.1 + 0.1 + 0.05 / 3
        assert result.iloc[-1, 0] == pytest.approx(expected, rel=1e-4)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        cap_need = pd.DataFrame({"A": [1100.0] * 10, "B": [1200.0] * 10}, index=dates)
        cap_actual = pd.DataFrame({"A": [1000.0] * 10, "B": [1000.0] * 10}, index=dates)
        amount = pd.DataFrame({"A": [500.0] * 10, "B": [500.0] * 10}, index=dates)

        result = factor.compute(
            daily_cap_need=cap_need, daily_cap_actual=cap_actual,
            daily_amount=amount, T=5
        )
        # A premium = 0.1, B premium = 0.2
        assert result.iloc[-1, 0] == pytest.approx(0.1, rel=1e-4)
        assert result.iloc[-1, 1] == pytest.approx(0.2, rel=1e-4)


class TestWeightedLiquidityPremiumEdgeCases:
    def test_zero_cap_actual(self, factor):
        """cap_actual 为 0 时不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]
        cap_need = pd.DataFrame(1000.0, index=dates, columns=stocks)
        cap_actual = pd.DataFrame(0.0, index=dates, columns=stocks)
        amount = pd.DataFrame(500.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_cap_need=cap_need, daily_cap_actual=cap_actual,
            daily_amount=amount, T=3
        )
        assert isinstance(result, pd.DataFrame)

    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        cap_need = pd.DataFrame([1100, np.nan, 1050, 1200, 1000], index=dates, columns=stocks, dtype=float)
        cap_actual = pd.DataFrame(1000.0, index=dates, columns=stocks)
        amount = pd.DataFrame(500.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_cap_need=cap_need, daily_cap_actual=cap_actual,
            daily_amount=amount, T=3
        )
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (5, 1)


class TestWeightedLiquidityPremiumOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        cap_need = pd.DataFrame(np.random.uniform(900, 1200, (30, 3)), index=dates, columns=stocks)
        cap_actual = pd.DataFrame(np.random.uniform(800, 1100, (30, 3)), index=dates, columns=stocks)
        amount = pd.DataFrame(np.random.uniform(100, 1000, (30, 3)), index=dates, columns=stocks)

        result = factor.compute(
            daily_cap_need=cap_need, daily_cap_actual=cap_actual, daily_amount=amount
        )
        assert result.shape == cap_need.shape
        assert list(result.columns) == list(cap_need.columns)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        cap_need = pd.DataFrame(1100.0, index=dates, columns=stocks)
        cap_actual = pd.DataFrame(1000.0, index=dates, columns=stocks)
        amount = pd.DataFrame(500.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_cap_need=cap_need, daily_cap_actual=cap_actual,
            daily_amount=amount, T=3
        )
        assert isinstance(result, pd.DataFrame)
