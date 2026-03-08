import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_factors_dir = Path(__file__).resolve().parent.parent / "factors"

_pkg_spec = importlib.util.spec_from_file_location(
    "factors", _factors_dir / "__init__.py", submodule_search_locations=[str(_factors_dir)]
)
_pkg_mod = importlib.util.module_from_spec(_pkg_spec)
sys.modules["factors"] = _pkg_mod

_base_spec = importlib.util.spec_from_file_location("factors.base", _factors_dir / "base.py")
_base_mod = importlib.util.module_from_spec(_base_spec)
sys.modules["factors.base"] = _base_mod
_base_spec.loader.exec_module(_base_mod)

_mod_spec = importlib.util.spec_from_file_location(
    "factors.upside_vol_ratio", _factors_dir / "upside_vol_ratio.py"
)
_mod = importlib.util.module_from_spec(_mod_spec)
sys.modules["factors.upside_vol_ratio"] = _mod
_mod_spec.loader.exec_module(_mod)

UpsideVolRatioFactor = _mod.UpsideVolRatioFactor


@pytest.fixture
def factor():
    return UpsideVolRatioFactor()


class TestUpsideVolRatioMetadata:
    def test_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "UPSIDE_VOL_RATIO"
        assert meta["category"] == "高频波动"
        assert meta["description"] != ""

    def test_repr(self, factor):
        r = repr(factor)
        assert "UpsideVolRatioFactor" in r


class TestUpsideVolRatioCompute:
    def test_known_rolling_mean(self, factor):
        """用已知数据验证滚动均值。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        df = pd.DataFrame({"A": [0.4, 0.5, 0.6, 0.5, 0.4]}, index=dates)

        result = factor.compute(daily_upside_ratio=df, T=3)

        expected = [0.4, 0.45, 0.5, 0.5 + 1 / 30, 0.5]
        np.testing.assert_array_almost_equal(result["A"].values, expected)

    def test_constant_half(self, factor):
        """常数 0.5 的滚动均值应为 0.5。"""
        dates = pd.bdate_range("2025-01-01", periods=20)
        df = pd.DataFrame({"X": [0.5] * 20}, index=dates)

        result = factor.compute(daily_upside_ratio=df, T=20)
        assert pytest.approx(result.iloc[-1, 0]) == 0.5

    def test_output_shape(self, factor):
        """输出形状应与输入一致。"""
        dates = pd.bdate_range("2025-01-01", periods=30)
        df = pd.DataFrame(
            {"A": np.random.rand(30), "B": np.random.rand(30)},
            index=dates,
        )

        result = factor.compute(daily_upside_ratio=df, T=20)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == df.shape

    def test_nan_handling(self, factor):
        """窗口内含 NaN 时应正确处理。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        df = pd.DataFrame({"A": [0.3, np.nan, 0.5, 0.7, 0.6]}, index=dates)

        result = factor.compute(daily_upside_ratio=df, T=3)
        # day3: mean([0.3, NaN, 0.5]) = 0.4
        assert pytest.approx(result.iloc[2, 0]) == 0.4

    def test_values_bounded(self, factor):
        """输入在 [0,1] 范围内时，输出也应在 [0,1] 范围内。"""
        np.random.seed(42)
        dates = pd.bdate_range("2025-01-01", periods=30)
        df = pd.DataFrame(
            {"A": np.random.uniform(0, 1, 30)},
            index=dates,
        )

        result = factor.compute(daily_upside_ratio=df, T=20)
        assert (result.dropna().values >= 0).all()
        assert (result.dropna().values <= 1).all()
