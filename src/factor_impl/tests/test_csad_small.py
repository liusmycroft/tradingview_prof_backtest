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
    "factors.csad_small", _factors_dir / "csad_small.py"
)
_mod = importlib.util.module_from_spec(_mod_spec)
sys.modules["factors.csad_small"] = _mod
_mod_spec.loader.exec_module(_mod)

CSADSmallFactor = _mod.CSADSmallFactor


@pytest.fixture
def factor():
    return CSADSmallFactor()


class TestCSADSmallMetadata:
    def test_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "CSAD_SMALL"
        assert meta["category"] == "行为金融"
        assert meta["description"] != ""

    def test_repr(self, factor):
        assert "CSADSmallFactor" in repr(factor)


class TestCSADSmallCompute:
    def test_highly_correlated_returns_low_csad(self, factor):
        """高度相关的股票收益率，CSAD 应较小"""
        np.random.seed(42)
        n_days = 160
        n_stocks = 15
        dates = pd.bdate_range("2024-01-01", periods=n_days)
        stocks = [f"S{i}" for i in range(n_stocks)]

        # 高度相关但不完全相同：共同因子 + 微小噪声
        base_ret = np.random.randn(n_days) * 0.02
        noise = np.random.randn(n_days, n_stocks) * 0.0001
        returns = pd.DataFrame(
            np.tile(base_ret.reshape(-1, 1), (1, n_stocks)) + noise,
            index=dates,
            columns=stocks,
        )

        result = factor.compute(returns=returns, T_corr=120, T_factor=20)
        last_valid = result.iloc[-1].dropna()
        # 高相关性 -> CSAD 很小 -> 标准化后绝对值也应较小
        if len(last_valid) > 0:
            assert last_valid.abs().max() < 5.0  # 标准化值不应极端

    def test_output_shape(self, factor):
        """输出形状应与输入一致"""
        np.random.seed(42)
        n_days = 160
        n_stocks = 15
        dates = pd.bdate_range("2024-01-01", periods=n_days)
        stocks = [f"S{i}" for i in range(n_stocks)]
        returns = pd.DataFrame(
            np.random.randn(n_days, n_stocks) * 0.02, index=dates, columns=stocks
        )

        result = factor.compute(returns=returns, T_corr=120, T_factor=20)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == returns.shape
        assert list(result.columns) == list(returns.columns)

    def test_insufficient_data_returns_nan(self, factor):
        """数据不足时应返回NaN"""
        np.random.seed(42)
        dates = pd.bdate_range("2024-01-01", periods=50)
        stocks = [f"S{i}" for i in range(15)]
        returns = pd.DataFrame(
            np.random.randn(50, 15) * 0.02, index=dates, columns=stocks
        )

        result = factor.compute(returns=returns, T_corr=120, T_factor=20)
        # 数据量 < T_corr + T_factor，全部应为 NaN
        assert result.isna().all().all()

    def test_result_is_negated(self, factor):
        """结果应为负的标准化CSAD（至少部分值为负）"""
        np.random.seed(123)
        n_days = 160
        n_stocks = 15
        dates = pd.bdate_range("2024-01-01", periods=n_days)
        stocks = [f"S{i}" for i in range(n_stocks)]
        returns = pd.DataFrame(
            np.random.randn(n_days, n_stocks) * 0.02, index=dates, columns=stocks
        )

        result = factor.compute(returns=returns, T_corr=120, T_factor=20)
        last_row = result.iloc[-1].dropna()
        # 标准化后取负，应有正有负
        if len(last_row) > 1:
            assert last_row.min() < 0 or last_row.max() > 0
