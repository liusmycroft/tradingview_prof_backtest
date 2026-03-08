import numpy as np
import pandas as pd
import pytest

from factors.post_open_large_buy import PostOpenLargeBuyFactor


@pytest.fixture
def factor():
    return PostOpenLargeBuyFactor()


class TestPostOpenLargeBuyMetadata:
    def test_name(self, factor):
        assert factor.name == "POST_OPEN_LARGE_BUY"

    def test_category(self, factor):
        assert factor.category == "高频资金流"

    def test_repr(self, factor):
        assert "POST_OPEN_LARGE_BUY" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "POST_OPEN_LARGE_BUY"
        assert meta["category"] == "高频资金流"


class TestPostOpenLargeBuyHandCalculated:
    """手算验证 (large_net_buy / total_amount).rolling(T).mean()。"""

    def test_constant_ratio(self, factor):
        """大单净买入 200, 总成交 1000 -> 强度恒为 0.2, 均值也为 0.2。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        net_buy = pd.DataFrame(200.0, index=dates, columns=stocks)
        total = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(daily_large_net_buy=net_buy, daily_total_amount=total, T=20)
        assert result.iloc[-1, 0] == pytest.approx(0.2, rel=1e-6)

    def test_varying_ratio_T3(self, factor):
        """T=3, 手动验证。

        net_buy = [100, 200, 300, 400, 500]
        total   = [1000, 1000, 1000, 1000, 1000]
        ratio   = [0.1, 0.2, 0.3, 0.4, 0.5]
        T=3:
          row 2: mean(0.1, 0.2, 0.3) = 0.2
          row 3: mean(0.2, 0.3, 0.4) = 0.3
          row 4: mean(0.3, 0.4, 0.5) = 0.4
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        net_buy = pd.DataFrame([100.0, 200.0, 300.0, 400.0, 500.0], index=dates, columns=stocks)
        total = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(daily_large_net_buy=net_buy, daily_total_amount=total, T=3)

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(0.2, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(0.3, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(0.4, rel=1e-10)

    def test_negative_net_buy(self, factor):
        """净买入为负（净卖出）时应正常处理。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        net_buy = pd.DataFrame(-100.0, index=dates, columns=stocks)
        total = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(daily_large_net_buy=net_buy, daily_total_amount=total, T=20)
        assert result.iloc[-1, 0] == pytest.approx(-0.1, rel=1e-6)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        net_buy = pd.DataFrame({"A": [200.0] * 25, "B": [400.0] * 25}, index=dates)
        total = pd.DataFrame(1000.0, index=dates, columns=["A", "B"])

        result = factor.compute(daily_large_net_buy=net_buy, daily_total_amount=total, T=20)
        assert result.iloc[-1, 0] == pytest.approx(0.2, rel=1e-6)
        assert result.iloc[-1, 1] == pytest.approx(0.4, rel=1e-6)


class TestPostOpenLargeBuyEdgeCases:
    def test_zero_total_amount(self, factor):
        """total_amount 为 0 时产生 inf/NaN, 不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        net_buy = pd.DataFrame([100.0] * 5, index=dates, columns=stocks)
        total = pd.DataFrame([0.0, 1000.0, 1000.0, 1000.0, 1000.0], index=dates, columns=stocks)

        result = factor.compute(daily_large_net_buy=net_buy, daily_total_amount=total, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        net_buy = pd.DataFrame(np.ones(10) * 100, index=dates, columns=stocks)
        net_buy.iloc[3, 0] = np.nan
        total = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(daily_large_net_buy=net_buy, daily_total_amount=total, T=5)
        assert isinstance(result, pd.DataFrame)

    def test_insufficient_window(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        net_buy = pd.DataFrame(100.0, index=dates, columns=stocks)
        total = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(daily_large_net_buy=net_buy, daily_total_amount=total, T=20)
        assert result.isna().all().all()


class TestPostOpenLargeBuyOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        net_buy = pd.DataFrame(
            np.random.uniform(-500, 500, (30, 3)), index=dates, columns=stocks
        )
        total = pd.DataFrame(
            np.random.uniform(500, 2000, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_large_net_buy=net_buy, daily_total_amount=total, T=20)

        assert result.shape == net_buy.shape
        assert list(result.columns) == list(net_buy.columns)
        assert list(result.index) == list(net_buy.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        net_buy = pd.DataFrame(100.0, index=dates, columns=stocks)
        total = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(daily_large_net_buy=net_buy, daily_total_amount=total, T=20)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        T = 20
        net_buy = pd.DataFrame(100.0, index=dates, columns=stocks)
        total = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(daily_large_net_buy=net_buy, daily_total_amount=total, T=T)

        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
