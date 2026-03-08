import numpy as np
import pandas as pd
import pytest

from factors.vwpin import VWPINFactor


@pytest.fixture
def factor():
    return VWPINFactor()


class TestVWPINMetadata:
    def test_name(self, factor):
        assert factor.name == "VWPIN"

    def test_category(self, factor):
        assert factor.category == "高频资金流"

    def test_repr(self, factor):
        assert "VWPIN" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "VWPIN"
        assert meta["category"] == "高频资金流"


class TestVWPINCompute:
    def test_balanced_order_flow(self, factor):
        """买卖平衡时，VWPIN = 0。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        buy = pd.DataFrame(500.0, index=dates, columns=stocks)
        sell = pd.DataFrame(500.0, index=dates, columns=stocks)
        total = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_buy_volume=buy, daily_sell_volume=sell, daily_total_volume=total, T=5
        )
        np.testing.assert_array_almost_equal(result["A"].values, 0.0)

    def test_all_buy(self, factor):
        """全部为买入时，VWPIN = buy / total。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        buy = pd.DataFrame(800.0, index=dates, columns=stocks)
        sell = pd.DataFrame(200.0, index=dates, columns=stocks)
        total = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_buy_volume=buy, daily_sell_volume=sell, daily_total_volume=total, T=3
        )
        # |800 - 200| / 1000 = 0.6
        np.testing.assert_array_almost_equal(result["A"].values, 0.6)

    def test_all_sell(self, factor):
        """全部为卖出时，VWPIN 同样为正（取绝对值）。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        buy = pd.DataFrame(200.0, index=dates, columns=stocks)
        sell = pd.DataFrame(800.0, index=dates, columns=stocks)
        total = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_buy_volume=buy, daily_sell_volume=sell, daily_total_volume=total, T=3
        )
        # |200 - 800| / 1000 = 0.6
        np.testing.assert_array_almost_equal(result["A"].values, 0.6)

    def test_manual_rolling_T3(self, factor):
        """T=3 手动验证滚动均值。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        buy = pd.DataFrame([800, 600, 400, 700, 500], index=dates, columns=stocks, dtype=float)
        sell = pd.DataFrame([200, 400, 600, 300, 500], index=dates, columns=stocks, dtype=float)
        total = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_buy_volume=buy, daily_sell_volume=sell, daily_total_volume=total, T=3
        )

        # imbalance: [0.6, 0.2, 0.2, 0.4, 0.0]
        # rolling mean T=3, min_periods=1:
        #   [0.6, 0.4, 1.0/3, 0.8/3, 0.6/3]
        assert result.iloc[0, 0] == pytest.approx(0.6, rel=1e-6)
        assert result.iloc[1, 0] == pytest.approx(0.4, rel=1e-6)
        assert result.iloc[2, 0] == pytest.approx(1.0 / 3, rel=1e-6)
        assert result.iloc[3, 0] == pytest.approx(0.8 / 3, rel=1e-6)
        assert result.iloc[4, 0] == pytest.approx(0.6 / 3, rel=1e-6)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        buy = pd.DataFrame({"A": [800.0] * 10, "B": [300.0] * 10}, index=dates)
        sell = pd.DataFrame({"A": [200.0] * 10, "B": [700.0] * 10}, index=dates)
        total = pd.DataFrame({"A": [1000.0] * 10, "B": [1000.0] * 10}, index=dates)

        result = factor.compute(
            daily_buy_volume=buy, daily_sell_volume=sell, daily_total_volume=total, T=5
        )
        np.testing.assert_array_almost_equal(result["A"].values, 0.6)
        np.testing.assert_array_almost_equal(result["B"].values, 0.4)

    def test_result_always_non_negative(self, factor):
        """VWPIN 应始终非负。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A", "B"]
        np.random.seed(42)
        buy = pd.DataFrame(np.random.uniform(100, 900, (20, 2)), index=dates, columns=stocks)
        sell = pd.DataFrame(np.random.uniform(100, 900, (20, 2)), index=dates, columns=stocks)
        total = pd.DataFrame(np.random.uniform(1000, 2000, (20, 2)), index=dates, columns=stocks)

        result = factor.compute(
            daily_buy_volume=buy, daily_sell_volume=sell, daily_total_volume=total, T=5
        )
        valid = result.dropna()
        assert (valid.values >= -1e-15).all()


class TestVWPINEdgeCases:
    def test_zero_total_volume(self, factor):
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]
        buy = pd.DataFrame(100.0, index=dates, columns=stocks)
        sell = pd.DataFrame(100.0, index=dates, columns=stocks)
        total = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_buy_volume=buy, daily_sell_volume=sell, daily_total_volume=total, T=3
        )
        assert isinstance(result, pd.DataFrame)

    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        buy = pd.DataFrame([800, np.nan, 400, 700, 500], index=dates, columns=stocks, dtype=float)
        sell = pd.DataFrame(200.0, index=dates, columns=stocks)
        total = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_buy_volume=buy, daily_sell_volume=sell, daily_total_volume=total, T=3
        )
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (5, 1)


class TestVWPINOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        buy = pd.DataFrame(np.random.uniform(100, 900, (30, 3)), index=dates, columns=stocks)
        sell = pd.DataFrame(np.random.uniform(100, 900, (30, 3)), index=dates, columns=stocks)
        total = pd.DataFrame(np.random.uniform(1000, 2000, (30, 3)), index=dates, columns=stocks)

        result = factor.compute(
            daily_buy_volume=buy, daily_sell_volume=sell, daily_total_volume=total
        )
        assert result.shape == buy.shape
        assert list(result.columns) == list(buy.columns)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        buy = pd.DataFrame(800.0, index=dates, columns=stocks)
        sell = pd.DataFrame(200.0, index=dates, columns=stocks)
        total = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_buy_volume=buy, daily_sell_volume=sell, daily_total_volume=total, T=3
        )
        assert isinstance(result, pd.DataFrame)
