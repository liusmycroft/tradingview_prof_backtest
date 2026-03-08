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
    "factors.csad", _factors_dir / "csad.py"
)
_mod = importlib.util.module_from_spec(_mod_spec)
sys.modules["factors.csad"] = _mod
_mod_spec.loader.exec_module(_mod)

CSADFactor = _mod.CSADFactor


@pytest.fixture
def factor():
    return CSADFactor()


class TestCSADMetadata:
    def test_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "CSAD"
        assert meta["category"] == "行为金融"
        assert meta["description"] != ""

    def test_repr(self, factor):
        r = repr(factor)
        assert "CSADFactor" in r


class TestCSADCompute:
    def test_output_has_beta2_column(self, factor):
        """输出应包含 beta2 列。"""
        np.random.seed(42)
        dates = pd.bdate_range("2025-01-01", periods=30)
        stock_returns = pd.DataFrame(
            np.random.normal(0, 0.02, (30, 5)),
            index=dates, columns=[f"S{i}" for i in range(5)],
        )
        market_return = stock_returns.mean(axis=1)

        result = factor.compute(
            stock_returns=stock_returns,
            market_return=market_return,
            T=20,
        )
        assert "beta2" in result.columns

    def test_output_shape(self, factor):
        """输出行数应与输入一致。"""
        np.random.seed(42)
        dates = pd.bdate_range("2025-01-01", periods=30)
        stock_returns = pd.DataFrame(
            np.random.normal(0, 0.02, (30, 3)),
            index=dates, columns=["A", "B", "C"],
        )
        market_return = stock_returns.mean(axis=1)

        result = factor.compute(
            stock_returns=stock_returns,
            market_return=market_return,
            T=20,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 30

    def test_first_T_minus_1_are_nan(self, factor):
        """前 T-1 行应为 NaN（窗口不足）。"""
        np.random.seed(42)
        T = 10
        dates = pd.bdate_range("2025-01-01", periods=20)
        stock_returns = pd.DataFrame(
            np.random.normal(0, 0.02, (20, 3)),
            index=dates, columns=["A", "B", "C"],
        )
        market_return = stock_returns.mean(axis=1)

        result = factor.compute(
            stock_returns=stock_returns,
            market_return=market_return,
            T=T,
        )
        assert result["beta2"].iloc[: T - 1].isna().all()
        assert not result["beta2"].iloc[T - 1 :].isna().all()

    def test_beta2_is_finite(self, factor):
        """有效窗口内的 beta2 应为有限值。"""
        np.random.seed(123)
        dates = pd.bdate_range("2025-01-01", periods=40)
        stock_returns = pd.DataFrame(
            np.random.normal(0, 0.02, (40, 10)),
            index=dates, columns=[f"S{i}" for i in range(10)],
        )
        market_return = stock_returns.mean(axis=1)

        result = factor.compute(
            stock_returns=stock_returns,
            market_return=market_return,
            T=20,
        )
        valid = result["beta2"].dropna()
        assert len(valid) > 0
        assert np.isfinite(valid.values).all()

    def test_insufficient_data_all_nan(self, factor):
        """数据不足 T 天时应全部为 NaN。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        stock_returns = pd.DataFrame(
            np.random.normal(0, 0.02, (5, 3)),
            index=dates, columns=["A", "B", "C"],
        )
        market_return = stock_returns.mean(axis=1)

        result = factor.compute(
            stock_returns=stock_returns,
            market_return=market_return,
            T=20,
        )
        assert result["beta2"].isna().all()

    def test_index_preserved(self, factor):
        """输出的 index 应与输入一致。"""
        np.random.seed(42)
        dates = pd.bdate_range("2025-01-01", periods=25)
        stock_returns = pd.DataFrame(
            np.random.normal(0, 0.02, (25, 3)),
            index=dates, columns=["A", "B", "C"],
        )
        market_return = stock_returns.mean(axis=1)

        result = factor.compute(
            stock_returns=stock_returns,
            market_return=market_return,
            T=20,
        )
        assert (result.index == dates).all()
