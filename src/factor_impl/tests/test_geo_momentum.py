import numpy as np
import pandas as pd
import pytest

import importlib
import importlib.util
import pathlib
import sys
import types

_factors_dir = pathlib.Path(__file__).resolve().parent.parent / "factors"

_pkg = types.ModuleType("factors")
_pkg.__path__ = [str(_factors_dir)]
sys.modules["factors"] = _pkg

_base_spec = importlib.util.spec_from_file_location("factors.base", _factors_dir / "base.py")
_base_mod = importlib.util.module_from_spec(_base_spec)
sys.modules["factors.base"] = _base_mod
_base_spec.loader.exec_module(_base_mod)

_mod_spec = importlib.util.spec_from_file_location(
    "factors.geo_momentum", _factors_dir / "geo_momentum.py"
)
_mod = importlib.util.module_from_spec(_mod_spec)
sys.modules["factors.geo_momentum"] = _mod
_mod_spec.loader.exec_module(_mod)

GeoMomentumFactor = _mod.GeoMomentumFactor


@pytest.fixture
def factor():
    return GeoMomentumFactor()


# --------------------------------------------------------------------------- #
# 基本元信息
# --------------------------------------------------------------------------- #

class TestMetadata:
    def test_name(self, factor):
        assert factor.name == "GEO_MOMENTUM"

    def test_category(self, factor):
        assert factor.category == "动量溢出"

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "GEO_MOMENTUM"
        assert "description" in meta


# --------------------------------------------------------------------------- #
# 手工验算
# --------------------------------------------------------------------------- #

class TestHandCalculation:
    def test_two_stocks_same_region(self, factor):
        """同地区两只股票，互为对方的加权收益。"""
        stocks = ["A", "B"]
        returns = pd.Series([0.02, 0.04], index=stocks)
        market_cap = pd.Series([1e9, 3e9], index=stocks)
        region = pd.Series(["北京", "北京"], index=stocks)

        result = factor.compute(returns=returns, market_cap=market_cap, region=region)

        # A: peer=B, weight=3e9/3e9=1.0, RET_GEO_A = 1.0 * 0.04 = 0.04
        # B: peer=A, weight=1e9/1e9=1.0, RET_GEO_B = 1.0 * 0.02 = 0.02
        np.testing.assert_almost_equal(result["A"], 0.04)
        np.testing.assert_almost_equal(result["B"], 0.02)

    def test_three_stocks_same_region(self, factor):
        """同地区三只股票的市值加权。"""
        stocks = ["A", "B", "C"]
        returns = pd.Series([0.01, 0.02, 0.03], index=stocks)
        market_cap = pd.Series([1e9, 2e9, 3e9], index=stocks)
        region = pd.Series(["上海", "上海", "上海"], index=stocks)

        result = factor.compute(returns=returns, market_cap=market_cap, region=region)

        # A: peers=B,C, weights=[2e9/5e9, 3e9/5e9]=[0.4, 0.6]
        # RET_GEO_A = 0.4*0.02 + 0.6*0.03 = 0.008 + 0.018 = 0.026
        np.testing.assert_almost_equal(result["A"], 0.026)

        # B: peers=A,C, weights=[1e9/4e9, 3e9/4e9]=[0.25, 0.75]
        # RET_GEO_B = 0.25*0.01 + 0.75*0.03 = 0.0025 + 0.0225 = 0.025
        np.testing.assert_almost_equal(result["B"], 0.025)

    def test_different_regions(self, factor):
        """不同地区的股票互不影响。"""
        stocks = ["A", "B", "C", "D"]
        returns = pd.Series([0.01, 0.02, 0.03, 0.04], index=stocks)
        market_cap = pd.Series([1e9, 2e9, 1e9, 2e9], index=stocks)
        region = pd.Series(["北京", "北京", "上海", "上海"], index=stocks)

        result = factor.compute(returns=returns, market_cap=market_cap, region=region)

        # A: peer=B only, RET_GEO_A = 0.02
        # B: peer=A only, RET_GEO_B = 0.01
        # C: peer=D only, RET_GEO_C = 0.04
        # D: peer=C only, RET_GEO_D = 0.03
        np.testing.assert_almost_equal(result["A"], 0.02)
        np.testing.assert_almost_equal(result["B"], 0.01)
        np.testing.assert_almost_equal(result["C"], 0.04)
        np.testing.assert_almost_equal(result["D"], 0.03)


# --------------------------------------------------------------------------- #
# 边界情况
# --------------------------------------------------------------------------- #

class TestEdgeCases:
    def test_single_stock_in_region_returns_nan(self, factor):
        """地区内只有一只股票时，无同伴，返回 NaN。"""
        stocks = ["A", "B"]
        returns = pd.Series([0.01, 0.02], index=stocks)
        market_cap = pd.Series([1e9, 2e9], index=stocks)
        region = pd.Series(["北京", "上海"], index=stocks)

        result = factor.compute(returns=returns, market_cap=market_cap, region=region)

        assert np.isnan(result["A"])
        assert np.isnan(result["B"])

    def test_zero_market_cap_peers(self, factor):
        """同伴市值全为零时，返回 NaN。"""
        stocks = ["A", "B"]
        returns = pd.Series([0.01, 0.02], index=stocks)
        market_cap = pd.Series([1e9, 0.0], index=stocks)
        region = pd.Series(["北京", "北京"], index=stocks)

        result = factor.compute(returns=returns, market_cap=market_cap, region=region)

        # A: peer=B, cap=0 => total_cap=0 => NaN
        assert np.isnan(result["A"])
        # B: peer=A, cap=1e9 => weight=1.0 => 0.01
        np.testing.assert_almost_equal(result["B"], 0.01)


# --------------------------------------------------------------------------- #
# 输出形状与类型
# --------------------------------------------------------------------------- #

class TestOutputShape:
    def test_output_is_series(self, factor):
        stocks = ["A", "B", "C"]
        returns = pd.Series([0.01, 0.02, 0.03], index=stocks)
        market_cap = pd.Series([1e9, 2e9, 3e9], index=stocks)
        region = pd.Series(["北京", "北京", "上海"], index=stocks)

        result = factor.compute(returns=returns, market_cap=market_cap, region=region)
        assert isinstance(result, pd.Series)

    def test_output_length_matches_input(self, factor):
        stocks = ["A", "B", "C", "D"]
        returns = pd.Series([0.01, 0.02, 0.03, 0.04], index=stocks)
        market_cap = pd.Series([1e9, 2e9, 3e9, 4e9], index=stocks)
        region = pd.Series(["北京", "北京", "上海", "上海"], index=stocks)

        result = factor.compute(returns=returns, market_cap=market_cap, region=region)
        assert len(result) == 4

    def test_output_index_matches_input(self, factor):
        stocks = ["000001.SZ", "600000.SH", "000002.SZ"]
        returns = pd.Series([0.01, 0.02, 0.03], index=stocks)
        market_cap = pd.Series([1e9, 2e9, 3e9], index=stocks)
        region = pd.Series(["北京", "北京", "上海"], index=stocks)

        result = factor.compute(returns=returns, market_cap=market_cap, region=region)
        assert list(result.index) == stocks
