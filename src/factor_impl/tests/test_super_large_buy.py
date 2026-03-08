import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Load factors.base and factors.super_large_buy directly from files.
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
    "factors.super_large_buy", _factors_dir / "super_large_buy.py"
)
_mod = importlib.util.module_from_spec(_mod_spec)
sys.modules["factors.super_large_buy"] = _mod
_mod_spec.loader.exec_module(_mod)

SuperLargeBuyFactor = _mod.SuperLargeBuyFactor


@pytest.fixture
def factor():
    return SuperLargeBuyFactor()


class TestSuperLargeBuyMetadata:
    def test_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "SUPER_LARGE_BUY"
        assert meta["category"] == "高频资金流"
        assert meta["description"] != ""

    def test_repr(self, factor):
        assert "SuperLargeBuyFactor" in repr(factor)


class TestSuperLargeBuyCompute:
    def test_known_values(self, factor):
        """买入100, 卖出100 -> 占比0.5"""
        dates = pd.bdate_range("2025-01-01", periods=25)
        buy = pd.DataFrame({"A": [100.0] * 25}, index=dates)
        sell = pd.DataFrame({"A": [100.0] * 25}, index=dates)

        result = factor.compute(super_big_buy=buy, super_big_sell=sell, T=20)
        assert pytest.approx(result.iloc[-1, 0], rel=1e-6) == 0.5

    def test_all_buy(self, factor):
        """卖出为0时，占比应为1.0"""
        dates = pd.bdate_range("2025-01-01", periods=25)
        buy = pd.DataFrame({"A": [100.0] * 25}, index=dates)
        sell = pd.DataFrame({"A": [0.0] * 25}, index=dates)

        result = factor.compute(super_big_buy=buy, super_big_sell=sell, T=20)
        # buy / (buy + 0) = 1.0
        assert pytest.approx(result.iloc[-1, 0], rel=1e-6) == 1.0

    def test_varying_ratio(self, factor):
        """手动验证: 前20天买入占比 = buy/(buy+sell)"""
        dates = pd.bdate_range("2025-01-01", periods=20)
        buy_vals = [80.0] * 10 + [120.0] * 10
        sell_vals = [20.0] * 10 + [80.0] * 10
        buy = pd.DataFrame({"A": buy_vals}, index=dates)
        sell = pd.DataFrame({"A": sell_vals}, index=dates)

        result = factor.compute(super_big_buy=buy, super_big_sell=sell, T=20)

        # 前10天: 80/(80+20) = 0.8, 后10天: 120/(120+80) = 0.6
        expected = (0.8 * 10 + 0.6 * 10) / 20
        assert pytest.approx(result.iloc[-1, 0], rel=1e-6) == expected

    def test_insufficient_window_returns_nan(self, factor):
        """数据不足T天时应返回NaN"""
        dates = pd.bdate_range("2025-01-01", periods=10)
        buy = pd.DataFrame({"A": [100.0] * 10}, index=dates)
        sell = pd.DataFrame({"A": [50.0] * 10}, index=dates)

        result = factor.compute(super_big_buy=buy, super_big_sell=sell, T=20)
        assert result.isna().all().all()

    def test_output_shape_and_type(self, factor):
        """输出形状和类型应与输入一致"""
        dates = pd.bdate_range("2025-01-01", periods=30)
        buy = pd.DataFrame(
            {"A": np.random.rand(30) + 1, "B": np.random.rand(30) + 1}, index=dates
        )
        sell = pd.DataFrame(
            {"A": np.random.rand(30) + 1, "B": np.random.rand(30) + 1}, index=dates
        )

        result = factor.compute(super_big_buy=buy, super_big_sell=sell, T=20)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == buy.shape
        assert list(result.columns) == list(buy.columns)
        assert (result.index == buy.index).all()

    def test_multi_stock(self, factor):
        """多只股票应独立计算"""
        np.random.seed(42)
        dates = pd.bdate_range("2025-01-01", periods=25)
        buy = pd.DataFrame(
            {"S1": np.random.rand(25) * 100 + 50, "S2": np.random.rand(25) * 50 + 10},
            index=dates,
        )
        sell = pd.DataFrame(
            {"S1": np.random.rand(25) * 100 + 50, "S2": np.random.rand(25) * 200 + 10},
            index=dates,
        )

        result = factor.compute(super_big_buy=buy, super_big_sell=sell, T=20)
        assert not pd.isna(result.iloc[-1, 0])
        assert not pd.isna(result.iloc[-1, 1])
        # S2 卖出远大于买入，占比应更低
        assert result.iloc[-1, 0] != result.iloc[-1, 1]
