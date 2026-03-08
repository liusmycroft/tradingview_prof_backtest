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
    "factors.dazzling_vol", _factors_dir / "dazzling_vol.py"
)
_mod = importlib.util.module_from_spec(_mod_spec)
sys.modules["factors.dazzling_vol"] = _mod
_mod_spec.loader.exec_module(_mod)

DazzlingVolFactor = _mod.DazzlingVolFactor


@pytest.fixture
def factor():
    return DazzlingVolFactor()


class TestDazzlingVolMetadata:
    def test_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "DAZZLING_VOL"
        assert meta["category"] == "高频波动"
        assert meta["description"] != ""

    def test_repr(self, factor):
        assert "DazzlingVolFactor" in repr(factor)


class TestDazzlingVolCompute:
    def test_constant_input(self, factor):
        """常数输入: mean=c, std=0 -> result = 0.5*c"""
        dates = pd.bdate_range("2025-01-01", periods=25)
        daily = pd.DataFrame({"A": [0.03] * 25}, index=dates)

        result = factor.compute(daily_dazzling=daily, T=20)
        # std of constant = 0, mean = 0.03 -> 0.5*0.03 + 0.5*0 = 0.015
        assert pytest.approx(result.iloc[-1, 0], rel=1e-6) == 0.015

    def test_known_mean_std(self, factor):
        """手动验证均值和标准差的等权平均"""
        dates = pd.bdate_range("2025-01-01", periods=20)
        vals = list(range(1, 21))  # 1..20
        daily = pd.DataFrame({"A": [float(v) for v in vals]}, index=dates)

        result = factor.compute(daily_dazzling=daily, T=20)

        expected_mean = np.mean(vals)
        expected_std = np.std(vals, ddof=1)
        expected = 0.5 * expected_mean + 0.5 * expected_std
        assert pytest.approx(result.iloc[-1, 0], rel=1e-6) == expected

    def test_insufficient_window_returns_nan(self, factor):
        """数据不足T天时应返回NaN"""
        dates = pd.bdate_range("2025-01-01", periods=10)
        daily = pd.DataFrame({"A": [0.03] * 10}, index=dates)

        result = factor.compute(daily_dazzling=daily, T=20)
        assert result.isna().all().all()

    def test_output_shape_and_type(self, factor):
        """输出形状和类型应与输入一致"""
        np.random.seed(42)
        dates = pd.bdate_range("2025-01-01", periods=30)
        daily = pd.DataFrame(
            {"A": np.random.rand(30), "B": np.random.rand(30)}, index=dates
        )

        result = factor.compute(daily_dazzling=daily, T=20)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == daily.shape
        assert list(result.columns) == list(daily.columns)
        assert (result.index == daily.index).all()

    def test_multi_stock(self, factor):
        """多只股票应独立计算"""
        np.random.seed(42)
        dates = pd.bdate_range("2025-01-01", periods=25)
        daily = pd.DataFrame(
            {"S1": np.random.rand(25) * 0.05, "S2": np.random.rand(25) * 0.1},
            index=dates,
        )

        result = factor.compute(daily_dazzling=daily, T=20)
        assert not pd.isna(result.iloc[-1, 0])
        assert not pd.isna(result.iloc[-1, 1])
        assert result.iloc[-1, 0] != result.iloc[-1, 1]
