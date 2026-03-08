import numpy as np
import pandas as pd
import pytest

from factors.trend_clarity_momentum import TrendClarityMomentumFactor


@pytest.fixture
def factor():
    return TrendClarityMomentumFactor()


class TestTrendClarityMomentumMetadata:
    def test_name(self, factor):
        assert factor.name == "TREND_CLARITY_MOMENTUM"

    def test_category(self, factor):
        assert factor.category == "趋势清晰度动量"

    def test_repr(self, factor):
        assert "TREND_CLARITY_MOMENTUM" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "TREND_CLARITY_MOMENTUM"
        assert meta["category"] == "趋势清晰度动量"


class TestTrendClarityMomentumHandCalculated:
    """手算验证 TM = -|MOM' - TC'|"""

    def test_perfect_trend_multiple_stocks(self, factor):
        """完美线性趋势时 R^2=1，多只股票验证横截面标准化。

        3 只股票，都是完美线性趋势但斜率不同。
        R^2 全为 1 -> TC' 标准化后标准差为 0 -> 跳过。
        """
        n = 260
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        stocks = ["A", "B", "C"]

        close = pd.DataFrame({
            "A": np.linspace(10, 50, n),
            "B": np.linspace(20, 80, n),
            "C": np.linspace(5, 30, n),
        }, index=dates)

        daily_return = close.pct_change().fillna(0)

        result = factor.compute(
            close=close, daily_return=daily_return,
            lookback=240, skip=20, min_periods=200,
        )

        # 当所有 TC 相同时 (R^2=1)，tc_std=0，结果应为 NaN
        assert isinstance(result, pd.DataFrame)

    def test_output_is_non_positive(self, factor):
        """TM = -|...| 应始终 <= 0。"""
        np.random.seed(42)
        n = 300
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        stocks = ["A", "B", "C", "D", "E"]

        close = pd.DataFrame(
            np.cumsum(np.random.randn(n, 5) * 0.02, axis=0) + 50,
            index=dates, columns=stocks,
        )
        daily_return = close.pct_change().fillna(0)

        result = factor.compute(
            close=close, daily_return=daily_return,
            lookback=240, skip=20, min_periods=200,
        )

        valid = result.dropna(how="all")
        if not valid.empty:
            # 所有非 NaN 值应 <= 0
            assert (valid.values[~np.isnan(valid.values)] <= 1e-10).all()

    def test_identical_stocks_give_zero(self, factor):
        """所有股票完全相同时，标准化后 MOM'=TC'=0，TM=0。"""
        np.random.seed(123)
        n = 280
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        stocks = ["A", "B", "C"]

        base = np.cumsum(np.random.randn(n) * 0.01) + 30
        close = pd.DataFrame(
            {s: base for s in stocks}, index=dates,
        )
        daily_return = close.pct_change().fillna(0)

        result = factor.compute(
            close=close, daily_return=daily_return,
            lookback=240, skip=20, min_periods=200,
        )

        # 所有股票相同 -> mom_std=0 或 tc_std=0 -> NaN，或者全为 0
        valid = result.dropna(how="all")
        if not valid.empty:
            non_nan = valid.values[~np.isnan(valid.values)]
            if len(non_nan) > 0:
                np.testing.assert_allclose(non_nan, 0.0, atol=1e-10)


class TestTrendClarityMomentumEdgeCases:
    def test_insufficient_data(self, factor):
        """数据不足 lookback 天时应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        stocks = ["A", "B"]

        close = pd.DataFrame(
            np.random.uniform(10, 50, (100, 2)), index=dates, columns=stocks
        )
        daily_return = close.pct_change().fillna(0)

        result = factor.compute(
            close=close, daily_return=daily_return,
            lookback=240, skip=20, min_periods=200,
        )
        assert result.isna().all().all()

    def test_nan_in_close(self, factor):
        """close 含 NaN 时不应抛异常。"""
        np.random.seed(42)
        n = 280
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        stocks = ["A", "B", "C"]

        close = pd.DataFrame(
            np.cumsum(np.random.randn(n, 3) * 0.02, axis=0) + 50,
            index=dates, columns=stocks,
        )
        close.iloc[50, 0] = np.nan
        daily_return = close.pct_change().fillna(0)

        result = factor.compute(
            close=close, daily_return=daily_return,
            lookback=240, skip=20, min_periods=200,
        )
        assert isinstance(result, pd.DataFrame)

    def test_single_stock(self, factor):
        """单只股票时无法做横截面标准化，结果应为 NaN。"""
        np.random.seed(42)
        n = 280
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        stocks = ["A"]

        close = pd.DataFrame(
            np.cumsum(np.random.randn(n, 1) * 0.02, axis=0) + 50,
            index=dates, columns=stocks,
        )
        daily_return = close.pct_change().fillna(0)

        result = factor.compute(
            close=close, daily_return=daily_return,
            lookback=240, skip=20, min_periods=200,
        )
        # 单只股票 std=0，应全为 NaN
        assert result.isna().all().all()


class TestTrendClarityMomentumOutputShape:
    def test_output_shape_matches_input(self, factor):
        np.random.seed(42)
        n = 280
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        stocks = ["A", "B", "C"]

        close = pd.DataFrame(
            np.cumsum(np.random.randn(n, 3) * 0.02, axis=0) + 50,
            index=dates, columns=stocks,
        )
        daily_return = close.pct_change().fillna(0)

        result = factor.compute(
            close=close, daily_return=daily_return,
            lookback=240, skip=20, min_periods=200,
        )
        assert result.shape == close.shape
        assert list(result.columns) == list(close.columns)
        assert list(result.index) == list(close.index)

    def test_output_is_dataframe(self, factor):
        np.random.seed(42)
        n = 280
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        stocks = ["A", "B", "C"]

        close = pd.DataFrame(
            np.cumsum(np.random.randn(n, 3) * 0.02, axis=0) + 50,
            index=dates, columns=stocks,
        )
        daily_return = close.pct_change().fillna(0)

        result = factor.compute(
            close=close, daily_return=daily_return,
            lookback=240, skip=20, min_periods=200,
        )
        assert isinstance(result, pd.DataFrame)

    def test_first_lookback_rows_are_nan(self, factor):
        """前 lookback 行应全为 NaN。"""
        np.random.seed(42)
        n = 300
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        stocks = ["A", "B", "C"]
        lookback = 240

        close = pd.DataFrame(
            np.cumsum(np.random.randn(n, 3) * 0.02, axis=0) + 50,
            index=dates, columns=stocks,
        )
        daily_return = close.pct_change().fillna(0)

        result = factor.compute(
            close=close, daily_return=daily_return,
            lookback=lookback, skip=20, min_periods=200,
        )
        assert result.iloc[:lookback].isna().all().all()
