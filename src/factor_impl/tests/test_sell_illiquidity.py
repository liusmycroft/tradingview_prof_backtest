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
    "factors.sell_illiquidity", _factors_dir / "sell_illiquidity.py"
)
_mod = importlib.util.module_from_spec(_mod_spec)
sys.modules["factors.sell_illiquidity"] = _mod
_mod_spec.loader.exec_module(_mod)

SellIlliquidityFactor = _mod.SellIlliquidityFactor


@pytest.fixture
def factor():
    return SellIlliquidityFactor()


# --------------------------------------------------------------------------- #
# 基本元信息
# --------------------------------------------------------------------------- #

class TestMetadata:
    def test_name(self, factor):
        assert factor.name == "SELL_ILLIQUIDITY"

    def test_category(self, factor):
        assert factor.category == "高频流动性"

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "SELL_ILLIQUIDITY"
        assert "description" in meta


# --------------------------------------------------------------------------- #
# 手工验算
# --------------------------------------------------------------------------- #

class TestHandCalculation:
    def test_known_regression(self, factor):
        """用已知数据验证 OLS 回归系数。"""
        dates = pd.date_range("2024-01-01", periods=1, freq="B")
        stocks = ["A", "B", "C", "D", "E"]

        # S 和 B 需要线性无关，避免多重共线性
        sell = [1.0, 2.0, 3.0, 4.0, 5.0]
        buy = [2.0, 5.0, 1.0, 4.0, 3.0]
        # r = 1 + 2*S + 3*B (精确线性关系)
        ret = [1 + 2 * s + 3 * b for s, b in zip(sell, buy)]

        returns = pd.DataFrame([ret], index=dates, columns=stocks)
        sell_amount = pd.DataFrame([sell], index=dates, columns=stocks)
        buy_amount = pd.DataFrame([buy], index=dates, columns=stocks)

        result = factor.compute(returns=returns, sell_amount=sell_amount, buy_amount=buy_amount)

        # beta1 应为 2.0
        np.testing.assert_almost_equal(result.iloc[0, 0], 2.0, decimal=5)

    def test_negative_beta1(self, factor):
        """卖出金额与收益率负相关时，beta1 应为负。"""
        dates = pd.date_range("2024-01-01", periods=1, freq="B")
        stocks = ["A", "B", "C", "D", "E"]

        sell = [1.0, 2.0, 3.0, 4.0, 5.0]
        buy = [1.0, 1.0, 1.0, 1.0, 1.0]
        # r = 10 - 1.5*S + 0*B
        ret = [10 - 1.5 * s for s in sell]

        returns = pd.DataFrame([ret], index=dates, columns=stocks)
        sell_amount = pd.DataFrame([sell], index=dates, columns=stocks)
        buy_amount = pd.DataFrame([buy], index=dates, columns=stocks)

        result = factor.compute(returns=returns, sell_amount=sell_amount, buy_amount=buy_amount)

        np.testing.assert_almost_equal(result.iloc[0, 0], -1.5, decimal=5)

    def test_beta1_broadcast_to_all_stocks(self, factor):
        """beta1 应广播到所有股票列。"""
        dates = pd.date_range("2024-01-01", periods=1, freq="B")
        stocks = ["A", "B", "C", "D"]

        np.random.seed(42)
        returns = pd.DataFrame(np.random.randn(1, 4), index=dates, columns=stocks)
        sell_amount = pd.DataFrame(np.random.rand(1, 4), index=dates, columns=stocks)
        buy_amount = pd.DataFrame(np.random.rand(1, 4), index=dates, columns=stocks)

        result = factor.compute(returns=returns, sell_amount=sell_amount, buy_amount=buy_amount)

        # 同一天所有股票的 beta1 应相同
        row = result.iloc[0]
        assert row.nunique() == 1


# --------------------------------------------------------------------------- #
# 边界情况
# --------------------------------------------------------------------------- #

class TestEdgeCases:
    def test_fewer_than_3_stocks_returns_nan(self, factor):
        """少于 3 只股票时，回归不可行，返回 NaN。"""
        dates = pd.date_range("2024-01-01", periods=1, freq="B")
        stocks = ["A", "B"]

        returns = pd.DataFrame([[0.01, 0.02]], index=dates, columns=stocks)
        sell_amount = pd.DataFrame([[100.0, 200.0]], index=dates, columns=stocks)
        buy_amount = pd.DataFrame([[150.0, 250.0]], index=dates, columns=stocks)

        result = factor.compute(returns=returns, sell_amount=sell_amount, buy_amount=buy_amount)

        assert result.isna().all().all()

    def test_nan_in_input(self, factor):
        """输入含 NaN 时，有效股票数 >= 3 仍可计算。"""
        dates = pd.date_range("2024-01-01", periods=1, freq="B")
        stocks = ["A", "B", "C", "D", "E"]

        returns = pd.DataFrame([[0.01, np.nan, 0.03, 0.04, 0.05]], index=dates, columns=stocks)
        sell_amount = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 5.0]], index=dates, columns=stocks)
        buy_amount = pd.DataFrame([[5.0, 4.0, 3.0, 2.0, 1.0]], index=dates, columns=stocks)

        result = factor.compute(returns=returns, sell_amount=sell_amount, buy_amount=buy_amount)

        # 4 个有效股票 >= 3，应能计算
        assert result.shape == (1, 5)
        assert not result.isna().all().all()

    def test_multiple_dates(self, factor):
        """多个日期独立回归。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        stocks = ["A", "B", "C", "D"]

        np.random.seed(42)
        returns = pd.DataFrame(np.random.randn(3, 4) * 0.02, index=dates, columns=stocks)
        sell_amount = pd.DataFrame(np.random.rand(3, 4) * 1e6, index=dates, columns=stocks)
        buy_amount = pd.DataFrame(np.random.rand(3, 4) * 1e6, index=dates, columns=stocks)

        result = factor.compute(returns=returns, sell_amount=sell_amount, buy_amount=buy_amount)

        assert result.shape == (3, 4)
        # 不同日期的 beta1 应不同
        assert result.iloc[0, 0] != result.iloc[1, 0]


# --------------------------------------------------------------------------- #
# 输出形状与类型
# --------------------------------------------------------------------------- #

class TestOutputShape:
    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        stocks = ["A", "B", "C", "D"]

        np.random.seed(0)
        returns = pd.DataFrame(np.random.randn(5, 4), index=dates, columns=stocks)
        sell_amount = pd.DataFrame(np.random.rand(5, 4), index=dates, columns=stocks)
        buy_amount = pd.DataFrame(np.random.rand(5, 4), index=dates, columns=stocks)

        result = factor.compute(returns=returns, sell_amount=sell_amount, buy_amount=buy_amount)
        assert isinstance(result, pd.DataFrame)

    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        stocks = ["A", "B", "C", "D", "E"]

        np.random.seed(0)
        returns = pd.DataFrame(np.random.randn(10, 5), index=dates, columns=stocks)
        sell_amount = pd.DataFrame(np.random.rand(10, 5), index=dates, columns=stocks)
        buy_amount = pd.DataFrame(np.random.rand(10, 5), index=dates, columns=stocks)

        result = factor.compute(returns=returns, sell_amount=sell_amount, buy_amount=buy_amount)
        assert result.shape == returns.shape

    def test_output_columns_preserved(self, factor):
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        stocks = ["000001.SZ", "600000.SH", "000002.SZ", "600036.SH"]

        np.random.seed(0)
        returns = pd.DataFrame(np.random.randn(3, 4), index=dates, columns=stocks)
        sell_amount = pd.DataFrame(np.random.rand(3, 4), index=dates, columns=stocks)
        buy_amount = pd.DataFrame(np.random.rand(3, 4), index=dates, columns=stocks)

        result = factor.compute(returns=returns, sell_amount=sell_amount, buy_amount=buy_amount)
        assert list(result.columns) == stocks
