import numpy as np
import pandas as pd
import pytest

from factors.improved_large_ratio_active import ImprovedLargeRatioActiveFactor


@pytest.fixture
def factor():
    return ImprovedLargeRatioActiveFactor()


class TestImprovedLargeRatioActiveMetadata:
    def test_name(self, factor):
        assert factor.name == "IMPROVED_LARGE_RATIO_ACTIVE"

    def test_category(self, factor):
        assert factor.category == "高频资金流"

    def test_repr(self, factor):
        assert "ImprovedLargeRatioActiveFactor" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "IMPROVED_LARGE_RATIO_ACTIVE"
        assert meta["category"] == "高频资金流"


class TestImprovedLargeRatioActiveCompute:
    def test_known_values(self, factor):
        """常数输入验证。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A", "B"]
        active_big_buy = pd.DataFrame(np.full((25, 2), 300.0), index=dates, columns=stocks)
        active_big_sell = pd.DataFrame(np.full((25, 2), 100.0), index=dates, columns=stocks)
        total_amount = pd.DataFrame(np.full((25, 2), 1000.0), index=dates, columns=stocks)

        result = factor.compute(
            active_big_buy=active_big_buy,
            active_big_sell=active_big_sell,
            total_amount=total_amount,
        )
        # (300 - 100) / 1000 = 0.2
        assert pytest.approx(result.iloc[-1, 0], rel=1e-6) == 0.2

    def test_negative_result(self, factor):
        """卖出大于买入时结果为负。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        active_big_buy = pd.DataFrame(np.full((5, 1), 100.0), index=dates, columns=stocks)
        active_big_sell = pd.DataFrame(np.full((5, 1), 400.0), index=dates, columns=stocks)
        total_amount = pd.DataFrame(np.full((5, 1), 1000.0), index=dates, columns=stocks)

        result = factor.compute(
            active_big_buy=active_big_buy,
            active_big_sell=active_big_sell,
            total_amount=total_amount,
            T=3,
        )
        # (100 - 400) / 1000 = -0.3
        assert pytest.approx(result.iloc[-1, 0], rel=1e-6) == -0.3

    def test_custom_window(self, factor):
        """自定义窗口验证。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        active_big_buy = pd.DataFrame(
            [[100], [200], [300], [400], [500]], index=dates, columns=stocks, dtype=float
        )
        active_big_sell = pd.DataFrame(np.zeros((5, 1)), index=dates, columns=stocks)
        total_amount = pd.DataFrame(np.full((5, 1), 1000.0), index=dates, columns=stocks)

        result = factor.compute(
            active_big_buy=active_big_buy,
            active_big_sell=active_big_sell,
            total_amount=total_amount,
            T=3,
        )
        # Last window: mean(0.3, 0.4, 0.5) = 0.4
        assert pytest.approx(result.iloc[-1, 0], rel=1e-6) == 0.4

    def test_balanced_buy_sell(self, factor):
        """买卖平衡时因子值为 0。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        active_big_buy = pd.DataFrame(np.full((10, 1), 200.0), index=dates, columns=stocks)
        active_big_sell = pd.DataFrame(np.full((10, 1), 200.0), index=dates, columns=stocks)
        total_amount = pd.DataFrame(np.full((10, 1), 1000.0), index=dates, columns=stocks)

        result = factor.compute(
            active_big_buy=active_big_buy,
            active_big_sell=active_big_sell,
            total_amount=total_amount,
        )
        np.testing.assert_array_almost_equal(result["A"].values, 0.0)


class TestImprovedLargeRatioActiveEdgeCases:
    def test_zero_total_amount(self, factor):
        """总成交额为 0 时不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]
        zeros = pd.DataFrame(np.zeros((3, 1)), index=dates, columns=stocks)
        result = factor.compute(
            active_big_buy=zeros, active_big_sell=zeros, total_amount=zeros
        )
        assert result.isna().all().all() or np.isinf(result.iloc[0, 0])

    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        buy = pd.DataFrame([100, np.nan, 300, 400, 500], index=dates, columns=stocks, dtype=float)
        sell = pd.DataFrame(np.full((5, 1), 100.0), index=dates, columns=stocks)
        total = pd.DataFrame(np.full((5, 1), 1000.0), index=dates, columns=stocks)

        result = factor.compute(active_big_buy=buy, active_big_sell=sell, total_amount=total, T=3)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (5, 1)


class TestImprovedLargeRatioActiveOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        buy = pd.DataFrame(np.random.uniform(100, 500, (30, 3)), index=dates, columns=stocks)
        sell = pd.DataFrame(np.random.uniform(100, 500, (30, 3)), index=dates, columns=stocks)
        total = pd.DataFrame(np.random.uniform(1000, 5000, (30, 3)), index=dates, columns=stocks)

        result = factor.compute(active_big_buy=buy, active_big_sell=sell, total_amount=total)
        assert result.shape == buy.shape
        assert list(result.columns) == list(buy.columns)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        buy = pd.DataFrame([100, 200, 300, 400, 500], index=dates, columns=stocks, dtype=float)
        sell = pd.DataFrame(np.full((5, 1), 100.0), index=dates, columns=stocks)
        total = pd.DataFrame(np.full((5, 1), 1000.0), index=dates, columns=stocks)

        result = factor.compute(active_big_buy=buy, active_big_sell=sell, total_amount=total, T=3)
        assert isinstance(result, pd.DataFrame)
