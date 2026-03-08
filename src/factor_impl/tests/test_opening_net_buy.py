import numpy as np
import pandas as pd
import pytest

import importlib
import importlib.util
import pathlib
import sys
import types

# 绕过 factors/__init__.py（其中导入了尚未实现的因子模块）。
_factors_dir = pathlib.Path(__file__).resolve().parent.parent / "factors"

_pkg = types.ModuleType("factors")
_pkg.__path__ = [str(_factors_dir)]
sys.modules["factors"] = _pkg

_base_spec = importlib.util.spec_from_file_location("factors.base", _factors_dir / "base.py")
_base_mod = importlib.util.module_from_spec(_base_spec)
sys.modules["factors.base"] = _base_mod
_base_spec.loader.exec_module(_base_mod)

_mod_spec = importlib.util.spec_from_file_location(
    "factors.opening_net_buy", _factors_dir / "opening_net_buy.py"
)
_mod = importlib.util.module_from_spec(_mod_spec)
sys.modules["factors.opening_net_buy"] = _mod
_mod_spec.loader.exec_module(_mod)

OpeningNetBuyFactor = _mod.OpeningNetBuyFactor


@pytest.fixture
def factor():
    return OpeningNetBuyFactor()


# --------------------------------------------------------------------------- #
# 基本元信息
# --------------------------------------------------------------------------- #

class TestMetadata:
    def test_name(self, factor):
        assert factor.name == "OPENING_NET_BUY"

    def test_category(self, factor):
        assert factor.category == "高频资金流"

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "OPENING_NET_BUY"
        assert "description" in meta


# --------------------------------------------------------------------------- #
# 手工验算
# --------------------------------------------------------------------------- #

class TestHandCalculation:
    """用极简数据手工验证公式正确性。"""

    def test_constant_inputs(self, factor):
        """常数输入：mean=100, std=50 => SNR=2.0，滚动均值=2.0"""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        stocks = ["A"]

        net_buy_mean = pd.DataFrame(100.0, index=dates, columns=stocks)
        net_buy_std = pd.DataFrame(50.0, index=dates, columns=stocks)

        result = factor.compute(net_buy_mean=net_buy_mean, net_buy_std=net_buy_std, T=3)

        expected = pd.DataFrame(2.0, index=dates, columns=stocks)
        pd.testing.assert_frame_equal(result, expected)

    def test_varying_inputs(self, factor):
        """变化输入，手工计算滚动均值。"""
        dates = pd.date_range("2024-01-01", periods=4, freq="B")
        stocks = ["A"]

        net_buy_mean = pd.DataFrame([100, 200, 300, 400], index=dates, columns=stocks, dtype=float)
        net_buy_std = pd.DataFrame([100, 100, 100, 100], index=dates, columns=stocks, dtype=float)

        result = factor.compute(net_buy_mean=net_buy_mean, net_buy_std=net_buy_std, T=3)

        # daily SNR: [1.0, 2.0, 3.0, 4.0]
        # rolling(3, min_periods=1):
        #   day0: mean([1.0])           = 1.0
        #   day1: mean([1.0, 2.0])      = 1.5
        #   day2: mean([1.0, 2.0, 3.0]) = 2.0
        #   day3: mean([2.0, 3.0, 4.0]) = 3.0
        expected_vals = [1.0, 1.5, 2.0, 3.0]
        np.testing.assert_allclose(result["A"].values, expected_vals, atol=1e-12)

    def test_multiple_stocks(self, factor):
        """多只股票并行计算，互不干扰。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        stocks = ["A", "B"]

        net_buy_mean = pd.DataFrame(
            {"A": [100.0, 200.0, 300.0], "B": [10.0, 20.0, 30.0]}, index=dates
        )
        net_buy_std = pd.DataFrame(
            {"A": [100.0, 100.0, 100.0], "B": [10.0, 10.0, 10.0]}, index=dates
        )

        result = factor.compute(net_buy_mean=net_buy_mean, net_buy_std=net_buy_std, T=2)

        # A daily SNR: [1.0, 2.0, 3.0] => rolling(2): [1.0, 1.5, 2.5]
        # B daily SNR: [1.0, 2.0, 3.0] => rolling(2): [1.0, 1.5, 2.5]
        np.testing.assert_allclose(result["A"].values, [1.0, 1.5, 2.5], atol=1e-12)
        np.testing.assert_allclose(result["B"].values, [1.0, 1.5, 2.5], atol=1e-12)


# --------------------------------------------------------------------------- #
# 边界情况
# --------------------------------------------------------------------------- #

class TestEdgeCases:
    def test_zero_std_produces_inf(self, factor):
        """标准差为零时，除法产生 inf，不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        stocks = ["A"]

        net_buy_mean = pd.DataFrame(100.0, index=dates, columns=stocks)
        net_buy_std = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(net_buy_mean=net_buy_mean, net_buy_std=net_buy_std, T=2)

        assert result.shape == (3, 1)
        assert np.all(np.isinf(result.values) | np.isnan(result.values))

    def test_nan_propagation(self, factor):
        """输入含 NaN 时，rolling mean(min_periods=1) 会跳过 NaN。"""
        dates = pd.date_range("2024-01-01", periods=4, freq="B")
        stocks = ["A"]

        net_buy_mean = pd.DataFrame([100.0, np.nan, 100.0, 100.0], index=dates, columns=stocks)
        net_buy_std = pd.DataFrame(50.0, index=dates, columns=stocks)

        result = factor.compute(net_buy_mean=net_buy_mean, net_buy_std=net_buy_std, T=2)

        # daily SNR: [2.0, NaN, 2.0, 2.0]
        # rolling(2, min_periods=1): [2.0, 2.0, 2.0, 2.0]
        expected = pd.DataFrame([2.0, 2.0, 2.0, 2.0], index=dates, columns=stocks)
        pd.testing.assert_frame_equal(result, expected)

    def test_t_equals_one(self, factor):
        """T=1 时，结果等于日度 SNR 本身。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        stocks = ["A"]

        net_buy_mean = pd.DataFrame([100.0, 200.0, 300.0], index=dates, columns=stocks)
        net_buy_std = pd.DataFrame([50.0, 100.0, 150.0], index=dates, columns=stocks)

        result = factor.compute(net_buy_mean=net_buy_mean, net_buy_std=net_buy_std, T=1)

        expected = pd.DataFrame([2.0, 2.0, 2.0], index=dates, columns=stocks)
        pd.testing.assert_frame_equal(result, expected)


# --------------------------------------------------------------------------- #
# 输出形状与类型
# --------------------------------------------------------------------------- #

class TestOutputShape:
    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        stocks = ["A", "B", "C"]

        net_buy_mean = pd.DataFrame(1.0, index=dates, columns=stocks)
        net_buy_std = pd.DataFrame(1.0, index=dates, columns=stocks)

        result = factor.compute(net_buy_mean=net_buy_mean, net_buy_std=net_buy_std)
        assert isinstance(result, pd.DataFrame)

    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        stocks = ["A", "B"]

        net_buy_mean = pd.DataFrame(1.0, index=dates, columns=stocks)
        net_buy_std = pd.DataFrame(1.0, index=dates, columns=stocks)

        result = factor.compute(net_buy_mean=net_buy_mean, net_buy_std=net_buy_std, T=5)
        assert result.shape == net_buy_mean.shape

    def test_output_index_and_columns_preserved(self, factor):
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        stocks = ["000001.SZ", "600000.SH"]

        net_buy_mean = pd.DataFrame(1.0, index=dates, columns=stocks)
        net_buy_std = pd.DataFrame(1.0, index=dates, columns=stocks)

        result = factor.compute(net_buy_mean=net_buy_mean, net_buy_std=net_buy_std)
        pd.testing.assert_index_equal(result.index, dates)
        assert list(result.columns) == stocks
