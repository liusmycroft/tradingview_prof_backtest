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
    "factors.main_force_strength", _factors_dir / "main_force_strength.py"
)
_mod = importlib.util.module_from_spec(_mod_spec)
sys.modules["factors.main_force_strength"] = _mod
_mod_spec.loader.exec_module(_mod)

MainForceStrengthFactor = _mod.MainForceStrengthFactor


@pytest.fixture
def factor():
    return MainForceStrengthFactor()


class TestMainForceStrengthMetadata:
    def test_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "MAIN_FORCE_STRENGTH"
        assert meta["category"] == "高频成交分布"
        assert meta["description"] != ""

    def test_repr(self, factor):
        r = repr(factor)
        assert "MainForceStrengthFactor" in r


class TestMainForceStrengthCompute:
    def test_known_rolling_mean(self, factor):
        """用已知数据验证滚动均值。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        df = pd.DataFrame({"A": [0.2, 0.4, 0.6, 0.8, 1.0]}, index=dates)

        result = factor.compute(daily_ts=df, T=3)

        expected = [0.2, 0.3, 0.4, 0.6, 0.8]
        np.testing.assert_array_almost_equal(result["A"].values, expected)

    def test_constant_values(self, factor):
        """常数序列的滚动均值应等于该常数。"""
        dates = pd.bdate_range("2025-01-01", periods=20)
        df = pd.DataFrame({"X": [0.5] * 20}, index=dates)

        result = factor.compute(daily_ts=df, T=20)
        assert pytest.approx(result.iloc[-1, 0]) == 0.5

    def test_output_shape(self, factor):
        """输出形状应与输入一致。"""
        dates = pd.bdate_range("2025-01-01", periods=30)
        df = pd.DataFrame(
            {"A": np.random.rand(30), "B": np.random.rand(30)},
            index=dates,
        )

        result = factor.compute(daily_ts=df, T=20)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == df.shape

    def test_nan_handling(self, factor):
        """窗口内含 NaN 时应正确处理。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        df = pd.DataFrame({"A": [0.1, np.nan, 0.3, 0.4, 0.5]}, index=dates)

        result = factor.compute(daily_ts=df, T=3)
        # day3: mean([0.1, NaN, 0.3]) = 0.2
        assert pytest.approx(result.iloc[2, 0]) == 0.2

    def test_multi_stock_independent(self, factor):
        """多只股票应独立计算。"""
        np.random.seed(42)
        dates = pd.bdate_range("2025-01-01", periods=25)
        df = pd.DataFrame(
            {"S1": np.random.rand(25), "S2": np.random.rand(25) + 1},
            index=dates,
        )

        result = factor.compute(daily_ts=df, T=20)
        assert result.iloc[-1, 1] > result.iloc[-1, 0]
