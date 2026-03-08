"""Tests for AFHCloseFactor."""

import numpy as np
import pandas as pd
import pytest

from factors.afh_close import AFHCloseFactor


@pytest.fixture
def factor():
    return AFHCloseFactor()


def _make_frames(close_vals, wafh_vals, tvol_vals, dates=None, stocks=None):
    """Helper to build aligned DataFrames from flat lists."""
    dates = dates or pd.date_range("2024-01-01", periods=len(close_vals))
    stocks = stocks or ["000001"]
    kw = dict(index=dates, columns=stocks)
    return (
        pd.DataFrame(np.array(close_vals).reshape(-1, 1), **kw),
        pd.DataFrame(np.array(wafh_vals).reshape(-1, 1), **kw),
        pd.DataFrame(np.array(tvol_vals).reshape(-1, 1), **kw),
    )


class TestAFHCloseBasic:
    """Known-value tests with hand-calculated data."""

    def test_single_day(self, factor):
        """Single day, T=1: AFH = close * wafh / tvol."""
        close, wafh, tvol = _make_frames([10.0], [500.0], [10000.0])
        result = factor.compute(
            close=close, weighted_afh_volume=wafh, total_volume=tvol, T=1
        )
        # 10.0 * 500.0 / 10000.0 = 0.5
        assert result.iloc[0, 0] == pytest.approx(0.5)

    def test_two_day_rolling(self, factor):
        """Two days with T=2: rolling sums over both days."""
        close, wafh, tvol = _make_frames(
            [10.0, 12.0], [200.0, 300.0], [5000.0, 5000.0]
        )
        result = factor.compute(
            close=close, weighted_afh_volume=wafh, total_volume=tvol, T=2
        )
        # Day 1 (only 1 day available, min_periods=1):
        #   10.0 * 200.0 / 5000.0 = 0.4
        assert result.iloc[0, 0] == pytest.approx(0.4)
        # Day 2: 12.0 * (200+300) / (5000+5000) = 12.0 * 500 / 10000 = 0.6
        assert result.iloc[1, 0] == pytest.approx(0.6)

    def test_hand_calculated_three_day(self, factor):
        """Three days with T=3, verify full rolling window."""
        close, wafh, tvol = _make_frames(
            [8.0, 9.0, 10.0],
            [100.0, 200.0, 300.0],
            [1000.0, 2000.0, 3000.0],
        )
        result = factor.compute(
            close=close, weighted_afh_volume=wafh, total_volume=tvol, T=3
        )
        # Day 3: 10.0 * (100+200+300) / (1000+2000+3000) = 10.0 * 600/6000 = 1.0
        assert result.iloc[2, 0] == pytest.approx(1.0)

    def test_window_slides(self, factor):
        """With T=2 and 3 days, day 3 should only use days 2-3."""
        close, wafh, tvol = _make_frames(
            [5.0, 10.0, 20.0],
            [100.0, 200.0, 400.0],
            [1000.0, 1000.0, 1000.0],
        )
        result = factor.compute(
            close=close, weighted_afh_volume=wafh, total_volume=tvol, T=2
        )
        # Day 3: 20.0 * (200+400) / (1000+1000) = 20.0 * 600/2000 = 6.0
        assert result.iloc[2, 0] == pytest.approx(6.0)


class TestAFHCloseEdgeCases:
    """Edge case tests."""

    def test_zero_total_volume(self, factor):
        """Zero total volume should produce NaN, not inf."""
        close, wafh, tvol = _make_frames([10.0], [500.0], [0.0])
        result = factor.compute(
            close=close, weighted_afh_volume=wafh, total_volume=tvol, T=1
        )
        assert np.isnan(result.iloc[0, 0])

    def test_zero_weighted_afh(self, factor):
        """Zero weighted AFH volume should produce 0."""
        close, wafh, tvol = _make_frames([10.0], [0.0], [10000.0])
        result = factor.compute(
            close=close, weighted_afh_volume=wafh, total_volume=tvol, T=1
        )
        assert result.iloc[0, 0] == pytest.approx(0.0)

    def test_nan_in_close(self, factor):
        """NaN in close should propagate to output."""
        close, wafh, tvol = _make_frames([np.nan], [500.0], [10000.0])
        result = factor.compute(
            close=close, weighted_afh_volume=wafh, total_volume=tvol, T=1
        )
        assert np.isnan(result.iloc[0, 0])

    def test_nan_in_weighted_afh(self, factor):
        """NaN in weighted_afh_volume — rolling sum skips NaN (min_periods=1)."""
        close, wafh, tvol = _make_frames(
            [10.0, 10.0], [np.nan, 500.0], [10000.0, 10000.0]
        )
        result = factor.compute(
            close=close, weighted_afh_volume=wafh, total_volume=tvol, T=2
        )
        # rolling sum treats NaN as missing: sum = 0 + 500 = 500
        # 10.0 * 500 / 20000 = 0.25
        assert result.iloc[1, 0] == pytest.approx(0.25)
        # Day 1 only has NaN -> rolling sum = NaN -> result is NaN
        assert np.isnan(result.iloc[0, 0])

    def test_nan_in_total_volume(self, factor):
        """NaN in total_volume should propagate."""
        close, wafh, tvol = _make_frames([10.0], [500.0], [np.nan])
        result = factor.compute(
            close=close, weighted_afh_volume=wafh, total_volume=tvol, T=1
        )
        assert np.isnan(result.iloc[0, 0])

    def test_partial_zero_volume_in_window(self, factor):
        """Some zero volume days in window — should still compute if sum > 0."""
        close, wafh, tvol = _make_frames(
            [10.0, 10.0], [100.0, 200.0], [0.0, 5000.0]
        )
        result = factor.compute(
            close=close, weighted_afh_volume=wafh, total_volume=tvol, T=2
        )
        # Day 2: 10.0 * (100+200) / (0+5000) = 10.0 * 300/5000 = 0.6
        assert result.iloc[1, 0] == pytest.approx(0.6)


class TestAFHCloseOutput:
    """Output shape and type tests."""

    def test_output_is_dataframe(self, factor):
        close, wafh, tvol = _make_frames([10.0], [500.0], [10000.0])
        result = factor.compute(
            close=close, weighted_afh_volume=wafh, total_volume=tvol
        )
        assert isinstance(result, pd.DataFrame)

    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=25)
        stocks = ["000001", "000002", "600000"]
        rng = np.random.default_rng(42)
        kw = dict(index=dates, columns=stocks)
        close = pd.DataFrame(rng.uniform(5, 50, (25, 3)), **kw)
        wafh = pd.DataFrame(rng.uniform(0, 1000, (25, 3)), **kw)
        tvol = pd.DataFrame(rng.uniform(1000, 100000, (25, 3)), **kw)

        result = factor.compute(
            close=close, weighted_afh_volume=wafh, total_volume=tvol, T=20
        )
        assert result.shape == (25, 3)
        assert list(result.columns) == stocks
        assert (result.index == dates).all()

    def test_output_index_and_columns_preserved(self, factor):
        dates = pd.date_range("2024-06-01", periods=5)
        stocks = ["SH600519", "SZ000858"]
        kw = dict(index=dates, columns=stocks)
        close = pd.DataFrame(np.ones((5, 2)) * 100, **kw)
        wafh = pd.DataFrame(np.ones((5, 2)) * 10, **kw)
        tvol = pd.DataFrame(np.ones((5, 2)) * 1000, **kw)

        result = factor.compute(
            close=close, weighted_afh_volume=wafh, total_volume=tvol, T=3
        )
        assert list(result.columns) == stocks
        assert (result.index == dates).all()


class TestAFHCloseMetadata:
    """Sanity check on factor metadata."""

    def test_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "afh_close"
        assert meta["category"] == "money_flow"
        assert len(meta["description"]) > 0

    def test_repr(self, factor):
        assert "afh_close" in repr(factor)
