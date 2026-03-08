import numpy as np
import pandas as pd
import pytest

from factors.active_buy_specificity import ActiveBuySpecificityFactor


@pytest.fixture
def factor():
    return ActiveBuySpecificityFactor()


class TestActiveBuySpecificityMetadata:
    def test_name(self, factor):
        assert factor.name == "ACTIVE_BUY_SPECIFICITY"

    def test_category(self, factor):
        assert factor.category == "高频资金流"

    def test_repr(self, factor):
        assert "ACTIVE_BUY_SPECIFICITY" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "ACTIVE_BUY_SPECIFICITY"
        assert meta["category"] == "高频资金流"


class TestActiveBuySpecificityHandCalculated:
    def test_cross_sectional_standardization(self, factor):
        """验证截面标准化: (x - mean) / std。

        day0: A=10, B=20, C=30 => mean=20, std=10
          A_std = (10-20)/10 = -1.0
          B_std = (20-20)/10 = 0.0
          C_std = (30-20)/10 = 1.0
        T=1 时均值就是自身。
        """
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A", "B", "C"]
        daily = pd.DataFrame([[10.0, 20.0, 30.0]], index=dates, columns=stocks)

        result = factor.compute(daily_active_buy_count=daily, T=1)

        assert result.iloc[0, 0] == pytest.approx(-1.0, rel=1e-6)
        assert result.iloc[0, 1] == pytest.approx(0.0, abs=1e-10)
        assert result.iloc[0, 2] == pytest.approx(1.0, rel=1e-6)

    def test_rolling_mean_T2(self, factor):
        """T=2 滚动均值验证。

        day0: A=10, B=20 => mean=15, std=~7.071
          A_std = (10-15)/7.071 = -0.7071
          B_std = (20-15)/7.071 = 0.7071
        day1: A=20, B=20 => mean=20, std=0 => NaN (std=0)
        T=2 rolling mean:
          day0: just day0 (min_periods=1)
          day1: mean(day0, day1) => day1 is NaN so mean of (-0.7071, NaN) = -0.7071 for A
        """
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A", "B"]
        daily = pd.DataFrame(
            {"A": [10.0, 30.0, 50.0], "B": [20.0, 10.0, 30.0]}, index=dates
        )

        result = factor.compute(daily_active_buy_count=daily, T=1)

        # T=1 时就是截面标准化本身
        # day0: mean=15, std=7.071 => A=-0.7071, B=0.7071
        assert result.iloc[0, 0] == pytest.approx(-0.7071, abs=0.001)
        assert result.iloc[0, 1] == pytest.approx(0.7071, abs=0.001)

    def test_constant_cross_section_gives_nan(self, factor):
        """截面全部相同时, std=0, 标准化结果为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A", "B"]
        daily = pd.DataFrame(5.0, index=dates, columns=stocks)

        result = factor.compute(daily_active_buy_count=daily, T=1)
        assert result.isna().all().all()

    def test_single_stock_gives_nan(self, factor):
        """单只股票时, std=NaN, 标准化结果为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame([10.0, 20.0, 30.0, 40.0, 50.0], index=dates, columns=stocks)

        result = factor.compute(daily_active_buy_count=daily, T=3)
        assert result.isna().all().all()


class TestActiveBuySpecificityEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A", "B", "C"]
        values = np.array([
            [10, 20, 30],
            [np.nan, 25, 35],
            [15, np.nan, 25],
            [20, 30, 40],
            [25, 35, 45],
        ], dtype=float)
        daily = pd.DataFrame(values, index=dates, columns=stocks)

        result = factor.compute(daily_active_buy_count=daily, T=3)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (5, 3)

    def test_zero_input(self, factor):
        """全零输入时, 截面 std=0, 结果为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A", "B"]
        daily = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_active_buy_count=daily, T=3)
        assert result.isna().all().all()


class TestActiveBuySpecificityOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily = pd.DataFrame(
            np.random.uniform(0, 50, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_active_buy_count=daily, T=20)

        assert result.shape == daily.shape
        assert list(result.columns) == list(daily.columns)
        assert list(result.index) == list(daily.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A", "B", "C"]
        daily = pd.DataFrame(
            np.random.uniform(0, 50, (5, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_active_buy_count=daily, T=3)
        assert isinstance(result, pd.DataFrame)
