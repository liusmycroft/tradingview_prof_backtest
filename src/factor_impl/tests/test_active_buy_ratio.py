import numpy as np
import pandas as pd
import pytest

from factors.active_buy_ratio import ActiveBuyRatioFactor


@pytest.fixture
def factor():
    return ActiveBuyRatioFactor()


class TestActiveBuyRatioMetadata:
    def test_name(self, factor):
        assert factor.name == "ACTIVE_BUY_RATIO"

    def test_category(self, factor):
        assert factor.category == "高频资金流"

    def test_repr(self, factor):
        assert "ACTIVE_BUY_RATIO" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "ACTIVE_BUY_RATIO"
        assert meta["category"] == "高频资金流"


class TestActiveBuyRatioHandCalculated:
    def test_rolling_mean_T3(self, factor):
        """T=3 滚动均值手算验证。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        data = pd.DataFrame([0.4, 0.5, 0.6, 0.3, 0.7], index=dates, columns=stocks)

        result = factor.compute(daily_active_buy_ratio=data, T=3)

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx((0.4 + 0.5 + 0.6) / 3, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx((0.5 + 0.6 + 0.3) / 3, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx((0.6 + 0.3 + 0.7) / 3, rel=1e-10)

    def test_constant_ratio(self, factor):
        """常数主买占比时，滚动均值等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(0.55, index=dates, columns=stocks)

        result = factor.compute(daily_active_buy_ratio=data, T=5)
        for i in range(4, 10):
            assert result.iloc[i, 0] == pytest.approx(0.55, rel=1e-10)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame(
            {"A": [0.4, 0.5, 0.6, 0.3, 0.7],
             "B": [0.6, 0.6, 0.6, 0.6, 0.6]},
            index=dates,
        )

        result = factor.compute(daily_active_buy_ratio=data, T=3)
        assert result.iloc[2, 0] == pytest.approx(0.5, rel=1e-10)
        assert result.iloc[2, 1] == pytest.approx(0.6, rel=1e-10)


class TestActiveBuyRatioEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        values = np.ones(10) * 0.5
        values[3] = np.nan
        data = pd.DataFrame(values, index=dates, columns=stocks)

        result = factor.compute(daily_active_buy_ratio=data, T=5)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_active_buy_ratio=data, T=5)
        assert result.isna().all().all()

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        data = pd.DataFrame(
            np.random.uniform(0.3, 0.7, (30, 3)), index=dates, columns=stocks
        )
        result = factor.compute(daily_active_buy_ratio=data, T=20)
        assert result.shape == data.shape

    def test_first_T_minus_1_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(np.ones(10) * 0.5, index=dates, columns=stocks)
        T = 5
        result = factor.compute(daily_active_buy_ratio=data, T=T)
        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
