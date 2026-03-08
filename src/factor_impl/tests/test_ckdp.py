"""Tests for CKDPFactor."""

import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import CKDPFactor directly from the module file to avoid triggering
# factors/__init__.py which may reference not-yet-created sibling modules.
_ckdp_path = Path(__file__).resolve().parent.parent / "factors" / "ckdp.py"
_base_path = Path(__file__).resolve().parent.parent / "factors" / "base.py"

# Load base first so the relative import in ckdp.py resolves
_base_spec = importlib.util.spec_from_file_location("factors.base", _base_path)
_base_mod = importlib.util.module_from_spec(_base_spec)
sys.modules["factors.base"] = _base_mod
_base_spec.loader.exec_module(_base_mod)

_spec = importlib.util.spec_from_file_location("factors.ckdp", _ckdp_path,
                                                submodule_search_locations=[])
_mod = importlib.util.module_from_spec(_spec)
sys.modules["factors.ckdp"] = _mod
_spec.loader.exec_module(_mod)
CKDPFactor = _mod.CKDPFactor


@pytest.fixture
def factor():
    return CKDPFactor()


def _make_frames(mean_vals, high_vals, low_vals, dates=None, stocks=None):
    """Helper to build aligned DataFrames from flat lists."""
    dates = dates or pd.date_range("2024-01-01", periods=len(mean_vals))
    stocks = stocks or ["000001"]
    kw = dict(index=dates, columns=stocks)
    return (
        pd.DataFrame(np.array(mean_vals).reshape(-1, 1), **kw),
        pd.DataFrame(np.array(high_vals).reshape(-1, 1), **kw),
        pd.DataFrame(np.array(low_vals).reshape(-1, 1), **kw),
    )


class TestCKDPBasic:
    """Known-value tests."""

    def test_midpoint(self, factor):
        """mean=50, low=0, high=100 -> CKDP=0.5"""
        chip_mean, chip_highest, chip_lowest = _make_frames([50], [100], [0])
        result = factor.compute(
            chip_mean=chip_mean, chip_highest=chip_highest, chip_lowest=chip_lowest
        )
        assert result.iloc[0, 0] == pytest.approx(0.5)

    def test_at_lowest(self, factor):
        """mean equals lowest -> CKDP=0.0"""
        chip_mean, chip_highest, chip_lowest = _make_frames([10], [100], [10])
        result = factor.compute(
            chip_mean=chip_mean, chip_highest=chip_highest, chip_lowest=chip_lowest
        )
        assert result.iloc[0, 0] == pytest.approx(0.0)

    def test_at_highest(self, factor):
        """mean equals highest -> CKDP=1.0"""
        chip_mean, chip_highest, chip_lowest = _make_frames([100], [100], [0])
        result = factor.compute(
            chip_mean=chip_mean, chip_highest=chip_highest, chip_lowest=chip_lowest
        )
        assert result.iloc[0, 0] == pytest.approx(1.0)

    def test_quarter(self, factor):
        """mean=25, low=0, high=100 -> CKDP=0.25"""
        chip_mean, chip_highest, chip_lowest = _make_frames([25], [100], [0])
        result = factor.compute(
            chip_mean=chip_mean, chip_highest=chip_highest, chip_lowest=chip_lowest
        )
        assert result.iloc[0, 0] == pytest.approx(0.25)


class TestCKDPEdgeCases:
    """Edge case tests."""

    def test_highest_equals_lowest(self, factor):
        """When highest == lowest, spread is 0 -> result should be NaN/inf."""
        chip_mean, chip_highest, chip_lowest = _make_frames([50], [50], [50])
        result = factor.compute(
            chip_mean=chip_mean, chip_highest=chip_highest, chip_lowest=chip_lowest
        )
        assert np.isnan(result.iloc[0, 0]) or np.isinf(result.iloc[0, 0])

    def test_nan_propagation(self, factor):
        """NaN in any input should propagate to output."""
        chip_mean, chip_highest, chip_lowest = _make_frames(
            [np.nan, 50], [100, 100], [0, 0]
        )
        result = factor.compute(
            chip_mean=chip_mean, chip_highest=chip_highest, chip_lowest=chip_lowest
        )
        assert np.isnan(result.iloc[0, 0])
        assert result.iloc[1, 0] == pytest.approx(0.5)

    def test_nan_in_highest(self, factor):
        chip_mean, chip_highest, chip_lowest = _make_frames(
            [50], [np.nan], [0]
        )
        result = factor.compute(
            chip_mean=chip_mean, chip_highest=chip_highest, chip_lowest=chip_lowest
        )
        assert np.isnan(result.iloc[0, 0])

    def test_nan_in_lowest(self, factor):
        chip_mean, chip_highest, chip_lowest = _make_frames(
            [50], [100], [np.nan]
        )
        result = factor.compute(
            chip_mean=chip_mean, chip_highest=chip_highest, chip_lowest=chip_lowest
        )
        assert np.isnan(result.iloc[0, 0])


class TestCKDPOutputRange:
    """Verify output stays in [0, 1] for valid inputs."""

    def test_range_single(self, factor):
        means = [10, 30, 50, 70, 90]
        chip_mean, chip_highest, chip_lowest = _make_frames(
            means, [100] * 5, [0] * 5
        )
        result = factor.compute(
            chip_mean=chip_mean, chip_highest=chip_highest, chip_lowest=chip_lowest
        )
        assert (result.values >= 0).all()
        assert (result.values <= 1).all()

    def test_range_multi_stock(self, factor):
        dates = pd.date_range("2024-01-01", periods=3)
        stocks = ["000001", "000002", "600000"]
        rng = np.random.default_rng(42)
        low = pd.DataFrame(rng.uniform(5, 20, (3, 3)), index=dates, columns=stocks)
        high = low + rng.uniform(10, 50, (3, 3))
        mean = low + rng.uniform(0, 1, (3, 3)) * (high - low)

        result = factor.compute(chip_mean=mean, chip_highest=high, chip_lowest=low)
        assert result.shape == (3, 3)
        assert (result.values >= -1e-10).all()
        assert (result.values <= 1 + 1e-10).all()


class TestCKDPMetadata:
    """Sanity check on factor metadata."""

    def test_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "ckdp"
        assert meta["category"] == "chip"
        assert len(meta["description"]) > 0

    def test_repr(self, factor):
        assert "ckdp" in repr(factor)
