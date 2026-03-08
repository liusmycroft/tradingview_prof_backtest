import numpy as np
import pandas as pd
import pytest

from factors.price_resiliency import PriceResiliencyFactor


@pytest.fixture
def factor():
    return PriceResiliencyFactor()


class TestPriceResiliencyMetadata:
    def test_name(self, factor):
        assert factor.name == "PRICE_RESILIENCY"

    def test_category(self, factor):
        assert factor.category == "高频流动性"

    def test_repr(self, factor):
        assert "PRICE_RESILIENCY" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "PRICE_RESILIENCY"
        assert meta["category"] == "高频流动性"
        assert "弹性" in meta["description"]


class TestPriceResiliencyHandCalculated:
    def test_constant_input(self, factor):
        """常数输入时，EMA=MA=常数，STD=0，结果为 NaN (0/0)。"""
        dates = pd.date_range("2024-01-01", periods=120, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(1.0, index=dates, columns=stocks)

        result = factor.compute(daily_resiliency=daily, T_short=20, T_long=120)
        # std=0 -> division by zero -> NaN
        assert result.iloc[-1].isna().all()

    def test_trending_input(self, factor):
        """上升趋势时，EMA_short > MA_long，因子值应为正。"""
        dates = pd.date_range("2024-01-01", periods=120, freq="D")
        stocks = ["A"]
        vals = np.linspace(1, 10, 120)
        daily = pd.DataFrame(vals, index=dates, columns=stocks)

        result = factor.compute(daily_resiliency=daily, T_short=20, T_long=120)
        last_val = result.iloc[-1, 0]
        assert not np.isnan(last_val)
        assert last_val > 0

    def test_declining_input(self, factor):
        """下降趋势时，EMA_short < MA_long，因子值应为负。"""
        dates = pd.date_range("2024-01-01", periods=120, freq="D")
        stocks = ["A"]
        vals = np.linspace(10, 1, 120)
        daily = pd.DataFrame(vals, index=dates, columns=stocks)

        result = factor.compute(daily_resiliency=daily, T_short=20, T_long=120)
        last_val = result.iloc[-1, 0]
        assert not np.isnan(last_val)
        assert last_val < 0

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=120, freq="D")
        up = np.linspace(1, 10, 120)
        down = np.linspace(10, 1, 120)
        daily = pd.DataFrame({"A": up, "B": down}, index=dates)

        result = factor.compute(daily_resiliency=daily, T_short=20, T_long=120)
        assert result.iloc[-1, 0] > 0  # A 上升
        assert result.iloc[-1, 1] < 0  # B 下降


class TestPriceResiliencyEdgeCases:
    def test_short_data(self, factor):
        """数据不足 T_long 时，结果应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(np.random.rand(10), index=dates, columns=stocks)

        result = factor.compute(daily_resiliency=daily, T_short=20, T_long=120)
        assert result.isna().all().all()

    def test_nan_in_input(self, factor):
        """输入含 NaN 时不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=120, freq="D")
        stocks = ["A"]
        vals = np.linspace(1, 10, 120)
        vals[50] = np.nan
        daily = pd.DataFrame(vals, index=dates, columns=stocks)

        result = factor.compute(daily_resiliency=daily, T_short=20, T_long=120)
        assert isinstance(result, pd.DataFrame)

    def test_all_nan(self, factor):
        """全 NaN 输入时，结果应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=120, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_resiliency=daily, T_short=20, T_long=120)
        assert result.isna().all().all()


class TestPriceResiliencyOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=120, freq="D")
        stocks = ["A", "B", "C"]
        daily = pd.DataFrame(
            np.random.uniform(0.01, 1, (120, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_resiliency=daily, T_short=20, T_long=120)
        assert result.shape == daily.shape
        assert list(result.columns) == list(daily.columns)
        assert list(result.index) == list(daily.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=120, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(np.random.rand(120), index=dates, columns=stocks)

        result = factor.compute(daily_resiliency=daily, T_short=20, T_long=120)
        assert isinstance(result, pd.DataFrame)
