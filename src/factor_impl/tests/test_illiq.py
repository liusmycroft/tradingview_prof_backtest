"""Tests for ILLIQFactor."""

import numpy as np
import pandas as pd
import pytest

from factors.illiq import ILLIQFactor


@pytest.fixture
def factor():
    return ILLIQFactor()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_daily_illiq(values, dates=None, stocks=None):
    """Build a daily_illiq DataFrame from a flat list."""
    dates = dates or pd.date_range("2024-01-01", periods=len(values))
    stocks = stocks or ["000001"]
    return pd.DataFrame(
        np.array(values).reshape(-1, 1), index=dates, columns=stocks,
    )


# ---------------------------------------------------------------------------
# Rolling mean (compute)
# ---------------------------------------------------------------------------

class TestComputeRollingMean:
    """Test the main compute method (rolling mean of daily_illiq)."""

    def test_single_day(self, factor):
        """With one row, rolling mean equals the value itself."""
        daily = _make_daily_illiq([0.5])
        result = factor.compute(daily_illiq=daily, d=20)
        assert result.iloc[0, 0] == pytest.approx(0.5)

    def test_exact_window(self, factor):
        """Rolling mean over exactly d rows should be the simple average."""
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        daily = _make_daily_illiq(vals)
        result = factor.compute(daily_illiq=daily, d=5)
        # Last row: mean of [1,2,3,4,5] = 3.0
        assert result.iloc[-1, 0] == pytest.approx(3.0)

    def test_rolling_window_shorter_than_data(self, factor):
        """d=3 on 5 rows: last value = mean(3,4,5)=4.0"""
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        daily = _make_daily_illiq(vals)
        result = factor.compute(daily_illiq=daily, d=3)
        assert result.iloc[-1, 0] == pytest.approx(4.0)
        # Row index 2: mean(1,2,3)=2.0
        assert result.iloc[2, 0] == pytest.approx(2.0)

    def test_min_periods_fills_early_rows(self, factor):
        """Early rows with fewer than d observations should still produce values (min_periods=1)."""
        vals = [2.0, 4.0, 6.0]
        daily = _make_daily_illiq(vals)
        result = factor.compute(daily_illiq=daily, d=20)
        assert result.iloc[0, 0] == pytest.approx(2.0)
        assert result.iloc[1, 0] == pytest.approx(3.0)
        assert result.iloc[2, 0] == pytest.approx(4.0)

    def test_multi_stock(self, factor):
        """Rolling mean works independently per column."""
        dates = pd.date_range("2024-01-01", periods=4)
        stocks = ["000001", "600000"]
        data = pd.DataFrame(
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
            index=dates, columns=stocks,
        )
        result = factor.compute(daily_illiq=data, d=2)
        # Last row: mean of last 2
        assert result.loc[dates[-1], "000001"] == pytest.approx(3.5)
        assert result.loc[dates[-1], "600000"] == pytest.approx(35.0)


# ---------------------------------------------------------------------------
# Shortcut formula
# ---------------------------------------------------------------------------

class TestShortcutFormula:
    """Test the shortcut = 2*(high-low) - |close-open| formula via the helper."""

    def test_basic_bar(self):
        """Single bar: open=10, high=15, low=8, close=12, value=1000."""
        # shortcut = 2*(15-8) - |12-10| = 14 - 2 = 12
        # daily_illiq = 12 / 1000 = 0.012
        result = ILLIQFactor.compute_daily_illiq(
            open_prices=pd.Series([10.0]),
            high_prices=pd.Series([15.0]),
            low_prices=pd.Series([8.0]),
            close_prices=pd.Series([12.0]),
            values=pd.Series([1000.0]),
        )
        assert result == pytest.approx(0.012)

    def test_doji_bar(self):
        """Doji: open == close, shortcut = 2*(high-low)."""
        # open=close=50, high=55, low=45 -> shortcut = 2*10 - 0 = 20
        result = ILLIQFactor.compute_daily_illiq(
            open_prices=pd.Series([50.0]),
            high_prices=pd.Series([55.0]),
            low_prices=pd.Series([45.0]),
            close_prices=pd.Series([50.0]),
            values=pd.Series([500.0]),
        )
        assert result == pytest.approx(20.0 / 500.0)

    def test_full_body_bar(self):
        """open=low, close=high (full bullish body): shortcut = 2*range - range = range."""
        # open=10, low=10, high=20, close=20 -> shortcut = 2*10 - 10 = 10
        result = ILLIQFactor.compute_daily_illiq(
            open_prices=pd.Series([10.0]),
            high_prices=pd.Series([20.0]),
            low_prices=pd.Series([10.0]),
            close_prices=pd.Series([20.0]),
            values=pd.Series([200.0]),
        )
        assert result == pytest.approx(10.0 / 200.0)

    def test_bearish_bar(self):
        """close < open: |close-open| still positive."""
        # open=20, high=22, low=15, close=16 -> shortcut = 2*7 - 4 = 10
        result = ILLIQFactor.compute_daily_illiq(
            open_prices=pd.Series([20.0]),
            high_prices=pd.Series([22.0]),
            low_prices=pd.Series([15.0]),
            close_prices=pd.Series([16.0]),
            values=pd.Series([100.0]),
        )
        assert result == pytest.approx(0.1)

    def test_multiple_bars_sum(self):
        """Two bars: result is sum of individual ratios."""
        # bar1: shortcut = 2*(15-8) - |12-10| = 12, value=1000 -> 0.012
        # bar2: shortcut = 2*(22-15) - |16-20| = 10, value=100  -> 0.1
        # total = 0.112
        result = ILLIQFactor.compute_daily_illiq(
            open_prices=pd.Series([10.0, 20.0]),
            high_prices=pd.Series([15.0, 22.0]),
            low_prices=pd.Series([8.0, 15.0]),
            close_prices=pd.Series([12.0, 16.0]),
            values=pd.Series([1000.0, 100.0]),
        )
        assert result == pytest.approx(0.112)


# ---------------------------------------------------------------------------
# Helper method with DataFrame input (multi-stock)
# ---------------------------------------------------------------------------

class TestHelperMultiStock:
    """compute_daily_illiq with DataFrame inputs returns a Series."""

    def test_dataframe_input(self):
        stocks = ["000001", "600000"]
        o = pd.DataFrame([[10.0, 50.0]], columns=stocks)
        h = pd.DataFrame([[15.0, 55.0]], columns=stocks)
        l = pd.DataFrame([[8.0, 45.0]], columns=stocks)
        c = pd.DataFrame([[12.0, 50.0]], columns=stocks)
        v = pd.DataFrame([[1000.0, 500.0]], columns=stocks)

        result = ILLIQFactor.compute_daily_illiq(o, h, l, c, v)
        # stock 000001: shortcut=12, 12/1000=0.012
        # stock 600000: shortcut=20, 20/500=0.04
        assert result["000001"] == pytest.approx(0.012)
        assert result["600000"] == pytest.approx(0.04)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Zero value, NaN handling, etc."""

    def test_zero_value_bar_skipped(self):
        """Bars with value=0 should not contribute (treated as NaN)."""
        # bar1: shortcut=12, value=0 -> NaN (skipped in sum)
        # bar2: shortcut=10, value=100 -> 0.1
        result = ILLIQFactor.compute_daily_illiq(
            open_prices=pd.Series([10.0, 20.0]),
            high_prices=pd.Series([15.0, 22.0]),
            low_prices=pd.Series([8.0, 15.0]),
            close_prices=pd.Series([12.0, 16.0]),
            values=pd.Series([0.0, 100.0]),
        )
        assert result == pytest.approx(0.1)

    def test_all_zero_values(self):
        """All bars with value=0 -> result is 0 (sum of empty = 0 via nansum behavior)."""
        result = ILLIQFactor.compute_daily_illiq(
            open_prices=pd.Series([10.0]),
            high_prices=pd.Series([15.0]),
            low_prices=pd.Series([8.0]),
            close_prices=pd.Series([12.0]),
            values=pd.Series([0.0]),
        )
        # pd.Series.sum() skips NaN by default -> 0.0
        assert result == pytest.approx(0.0)

    def test_nan_in_prices(self):
        """NaN in price data propagates to that bar's ratio."""
        result = ILLIQFactor.compute_daily_illiq(
            open_prices=pd.Series([np.nan, 20.0]),
            high_prices=pd.Series([15.0, 22.0]),
            low_prices=pd.Series([8.0, 15.0]),
            close_prices=pd.Series([12.0, 16.0]),
            values=pd.Series([1000.0, 100.0]),
        )
        # bar1 ratio is NaN (skipped), bar2 = 0.1
        assert result == pytest.approx(0.1)

    def test_nan_in_value(self):
        """NaN in value -> that bar's ratio is NaN, skipped in sum."""
        result = ILLIQFactor.compute_daily_illiq(
            open_prices=pd.Series([10.0, 20.0]),
            high_prices=pd.Series([15.0, 22.0]),
            low_prices=pd.Series([8.0, 15.0]),
            close_prices=pd.Series([12.0, 16.0]),
            values=pd.Series([np.nan, 100.0]),
        )
        assert result == pytest.approx(0.1)

    def test_negative_value_skipped(self):
        """Negative value bars should be treated like zero (skipped)."""
        result = ILLIQFactor.compute_daily_illiq(
            open_prices=pd.Series([10.0, 20.0]),
            high_prices=pd.Series([15.0, 22.0]),
            low_prices=pd.Series([8.0, 15.0]),
            close_prices=pd.Series([12.0, 16.0]),
            values=pd.Series([-500.0, 100.0]),
        )
        assert result == pytest.approx(0.1)

    def test_nan_propagation_in_rolling(self, factor):
        """NaN in daily_illiq should propagate through rolling mean correctly."""
        daily = _make_daily_illiq([1.0, np.nan, 3.0, 4.0, 5.0])
        result = factor.compute(daily_illiq=daily, d=3)
        # Row 3 (index 3): window [NaN, 3, 4] -> mean of 3,4 = 3.5
        assert result.iloc[3, 0] == pytest.approx(3.5)
        # Row 4 (index 4): window [3, 4, 5] -> 4.0
        assert result.iloc[4, 0] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

class TestMetadata:
    """Sanity check on factor metadata."""

    def test_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "illiq"
        assert meta["category"] == "liquidity"
        assert len(meta["description"]) > 0

    def test_repr(self, factor):
        assert "illiq" in repr(factor)
