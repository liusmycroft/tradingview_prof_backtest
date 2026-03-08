import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Load factors.base and factors.utd directly from files, bypassing the
# package __init__.py which imports factor modules that may not exist yet.
_factors_dir = Path(__file__).resolve().parent.parent / "factors"

# 1. Register a minimal "factors" package so relative imports work.
_pkg_spec = importlib.util.spec_from_file_location(
    "factors", _factors_dir / "__init__.py", submodule_search_locations=[str(_factors_dir)]
)
_pkg_mod = importlib.util.module_from_spec(_pkg_spec)
sys.modules["factors"] = _pkg_mod  # register but do NOT exec (avoids broken imports)

# 2. Load factors.base
_base_spec = importlib.util.spec_from_file_location("factors.base", _factors_dir / "base.py")
_base_mod = importlib.util.module_from_spec(_base_spec)
sys.modules["factors.base"] = _base_mod
_base_spec.loader.exec_module(_base_mod)

# 3. Load factors.utd
_utd_spec = importlib.util.spec_from_file_location("factors.utd", _factors_dir / "utd.py")
_utd_mod = importlib.util.module_from_spec(_utd_spec)
sys.modules["factors.utd"] = _utd_mod
_utd_spec.loader.exec_module(_utd_mod)

UTDFactor = _utd_mod.UTDFactor


@pytest.fixture
def factor():
    return UTDFactor()


class TestUTDMetadata:
    def test_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "UTD"
        assert meta["category"] == "流动性"
        assert meta["description"] != ""


class TestUTDCompute:
    def test_known_cv(self, factor):
        """用可手算的数据验证变异系数计算。"""
        dates = pd.bdate_range("2025-01-01", periods=10)
        # 前 5 天: [2, 4, 4, 4, 6] -> mean=4, std(ddof=1)=sqrt(2.5), CV≈0.3953
        vals = [2.0, 4.0, 4.0, 4.0, 6.0, 3.0, 3.0, 3.0, 3.0, 3.0]
        df = pd.DataFrame({"A": vals}, index=dates)

        result = factor.compute(df, T=5)

        # 第 5 行（index=4）对应前 5 天窗口
        cv_day5 = result.iloc[4, 0]
        expected_std = np.std(vals[:5], ddof=1)
        expected_mean = np.mean(vals[:5])
        expected_cv = expected_std / expected_mean
        assert pytest.approx(cv_day5, rel=1e-6) == expected_cv

        # 后 5 天全是 3.0 -> std=0, CV=0
        cv_day10 = result.iloc[9, 0]
        assert cv_day10 == 0.0

    def test_constant_values_zero_cv(self, factor):
        """常数序列的变异系数应为 0。"""
        dates = pd.bdate_range("2025-01-01", periods=20)
        df = pd.DataFrame({"X": [5.0] * 20}, index=dates)

        result = factor.compute(df, T=20)
        assert result.iloc[-1, 0] == 0.0

    def test_nan_handling(self, factor):
        """窗口内含 NaN 时，min_periods 不满足应返回 NaN。"""
        dates = pd.bdate_range("2025-01-01", periods=20)
        vals = [1.0] * 20
        vals[10] = np.nan
        df = pd.DataFrame({"A": vals}, index=dates)

        result = factor.compute(df, T=20)
        assert pd.isna(result.iloc[-1, 0])

    def test_insufficient_window_returns_nan(self, factor):
        """数据不足 T 天时应返回 NaN。"""
        dates = pd.bdate_range("2025-01-01", periods=10)
        df = pd.DataFrame({"A": range(10)}, index=dates, dtype=float)

        result = factor.compute(df, T=20)
        assert result.isna().all().all()

    def test_output_shape_and_type(self, factor):
        """输出形状和类型应与输入一致。"""
        dates = pd.bdate_range("2025-01-01", periods=30)
        df = pd.DataFrame(
            {"A": np.random.rand(30), "B": np.random.rand(30)},
            index=dates,
        )

        result = factor.compute(df, T=20)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == df.shape
        assert list(result.columns) == list(df.columns)
        assert (result.index == df.index).all()

    def test_multi_stock(self, factor):
        """多只股票应独立计算。"""
        np.random.seed(42)
        dates = pd.bdate_range("2025-01-01", periods=25)
        df = pd.DataFrame(
            {"S1": np.random.rand(25) + 1, "S2": np.random.rand(25) + 2},
            index=dates,
        )

        result = factor.compute(df, T=20)
        assert not pd.isna(result.iloc[-1, 0])
        assert not pd.isna(result.iloc[-1, 1])
        assert result.iloc[-1, 0] != result.iloc[-1, 1]
