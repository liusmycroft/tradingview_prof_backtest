import numpy as np
import pandas as pd
import pytest

from factors.buy_order_concentration import BuyOrderConcentrationFactor


@pytest.fixture
def factor():
    return BuyOrderConcentrationFactor()


class TestBuyOrderConcentrationMetadata:
    def test_name(self, factor):
        assert factor.name == "BUY_ORDER_CONCENTRATION"

    def test_category(self, factor):
        assert factor.category == "高频资金流"

    def test_repr(self, factor):
        assert "BUY_ORDER_CONCENTRATION" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "BUY_ORDER_CONCENTRATION"
        assert meta["category"] == "高频资金流"


class TestBuyOrderConcentrationHandCalculated:
    """手算验证 rolling(T, min_periods=T).mean()"""

    def test_constant_input(self, factor):
        """常数输入时，滚动均值等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily_bc = pd.DataFrame(0.05, index=dates, columns=stocks)

        result = factor.compute(daily_buy_concentration=daily_bc, T=20)
        assert result.iloc[-1, 0] == pytest.approx(0.05, rel=1e-6)

    def test_varying_T3(self, factor):
        """T=3, 手动验证。

        data = [0.1, 0.2, 0.3, 0.4, 0.5]
        T=3:
          row 0,1: NaN
          row 2: mean(0.1, 0.2, 0.3) = 0.2
          row 3: mean(0.2, 0.3, 0.4) = 0.3
          row 4: mean(0.3, 0.4, 0.5) = 0.4
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_bc = pd.DataFrame([0.1, 0.2, 0.3, 0.4, 0.5], index=dates, columns=stocks)

        result = factor.compute(daily_buy_concentration=daily_bc, T=3)

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(0.2, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(0.3, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(0.4, rel=1e-10)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A", "B"]
        daily_bc = pd.DataFrame(
            {"A": [0.03] * 25, "B": [0.08] * 25}, index=dates
        )

        result = factor.compute(daily_buy_concentration=daily_bc, T=20)
        assert result.iloc[-1, 0] == pytest.approx(0.03, rel=1e-6)
        assert result.iloc[-1, 1] == pytest.approx(0.08, rel=1e-6)


class TestBuyOrderConcentrationEdgeCases:
    def test_nan_in_input(self, factor):
        """输入含 NaN 时不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_bc = pd.DataFrame([0.1, np.nan, 0.3, 0.4, 0.5], index=dates, columns=stocks)

        result = factor.compute(daily_buy_concentration=daily_bc, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_zero_input(self, factor):
        """全零输入时结果应全为 0。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_bc = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_buy_concentration=daily_bc, T=3)
        assert result.iloc[2, 0] == pytest.approx(0.0, abs=1e-15)

    def test_insufficient_window(self, factor):
        """数据不足 T 天时应返回 NaN。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_bc = pd.DataFrame(0.05, index=dates, columns=stocks)

        result = factor.compute(daily_buy_concentration=daily_bc, T=10)
        assert result.isna().all().all()


class TestBuyOrderConcentrationOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily_bc = pd.DataFrame(
            np.random.uniform(0.01, 0.1, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_buy_concentration=daily_bc, T=20)
        assert result.shape == daily_bc.shape
        assert list(result.columns) == list(daily_bc.columns)
        assert list(result.index) == list(daily_bc.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_bc = pd.DataFrame([0.1, 0.2, 0.3, 0.4, 0.5], index=dates, columns=stocks)

        result = factor.compute(daily_buy_concentration=daily_bc, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        """前 T-1 行应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        T = 5
        daily_bc = pd.DataFrame(
            np.random.uniform(0.01, 0.1, (10, 2)), index=dates, columns=stocks
        )

        result = factor.compute(daily_buy_concentration=daily_bc, T=T)
        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
