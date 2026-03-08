import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

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
    "factors.small_order_lag_corr", _factors_dir / "small_order_lag_corr.py"
)
_mod = importlib.util.module_from_spec(_mod_spec)
sys.modules["factors.small_order_lag_corr"] = _mod
_mod_spec.loader.exec_module(_mod)

SmallOrderLagCorrFactor = _mod.SmallOrderLagCorrFactor


@pytest.fixture
def factor():
    return SmallOrderLagCorrFactor()


# --------------------------------------------------------------------------- #
# 基本元信息
# --------------------------------------------------------------------------- #

class TestMetadata:
    def test_name(self, factor):
        assert factor.name == "SMALL_ORDER_LAG_CORR"

    def test_category(self, factor):
        assert factor.category == "高频资金流"

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "SMALL_ORDER_LAG_CORR"
        assert "description" in meta


# --------------------------------------------------------------------------- #
# 手工验算
# --------------------------------------------------------------------------- #

class TestHandCalculation:
    def test_perfect_positive_correlation(self, factor):
        """单调递增序列，秩相关应接近 1。"""
        dates = pd.date_range("2024-01-01", periods=22, freq="B")
        stocks = ["A"]

        # 单调递增序列 => S_t 和 S_{t+1} 高度正相关
        vals = list(range(1, 23))
        s_net_inflow = pd.DataFrame({"A": vals}, index=dates, dtype=float)

        result = factor.compute(s_net_inflow=s_net_inflow, T=20)

        # 最后一行应接近 1.0
        assert result.iloc[-1, 0] > 0.9

    def test_known_spearman(self, factor):
        """用已知数据验证 Spearman 相关系数。"""
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        stocks = ["A"]

        vals = [10.0, 20.0, 15.0, 25.0, 30.0, 5.0]
        s_net_inflow = pd.DataFrame({"A": vals}, index=dates, dtype=float)

        result = factor.compute(s_net_inflow=s_net_inflow, T=5)

        # 最后一行 (index=5): 窗口 [1..5], lagged 窗口 [0..4]
        # S_t:   [20, 15, 25, 30, 5]
        # S_{t-1}: [10, 20, 15, 25, 30]
        expected_corr, _ = spearmanr([20, 15, 25, 30, 5], [10, 20, 15, 25, 30])
        np.testing.assert_almost_equal(result.iloc[-1, 0], expected_corr, decimal=6)

    def test_multiple_stocks(self, factor):
        """多只股票独立计算。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="B")

        np.random.seed(42)
        s_net_inflow = pd.DataFrame(
            {"A": np.random.randn(10), "B": np.random.randn(10)},
            index=dates,
        )

        result = factor.compute(s_net_inflow=s_net_inflow, T=5)

        assert result.shape == (10, 2)
        # 两只股票的结果应不同
        assert not np.allclose(
            result["A"].dropna().values, result["B"].dropna().values, atol=1e-6
        )


# --------------------------------------------------------------------------- #
# 边界情况
# --------------------------------------------------------------------------- #

class TestEdgeCases:
    def test_insufficient_data_returns_nan(self, factor):
        """数据不足时（< 3 个有效配对），返回 NaN。"""
        dates = pd.date_range("2024-01-01", periods=2, freq="B")
        stocks = ["A"]

        s_net_inflow = pd.DataFrame({"A": [1.0, 2.0]}, index=dates)

        result = factor.compute(s_net_inflow=s_net_inflow, T=20)

        # 第 0 行: 只有 1 个值，lag 为 NaN => 0 个有效配对 => NaN
        # 第 1 行: 2 个值，1 个有效配对 (lag[0] 为 NaN) => < 3 => NaN
        assert result.isna().all().all()

    def test_nan_in_input(self, factor):
        """输入含 NaN 时不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        stocks = ["A"]

        vals = [1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        s_net_inflow = pd.DataFrame({"A": vals}, index=dates)

        result = factor.compute(s_net_inflow=s_net_inflow, T=5)

        assert result.shape == (10, 1)


# --------------------------------------------------------------------------- #
# 输出形状与类型
# --------------------------------------------------------------------------- #

class TestOutputShape:
    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="B")
        stocks = ["A", "B"]

        np.random.seed(0)
        s_net_inflow = pd.DataFrame(np.random.randn(25, 2), index=dates, columns=stocks)

        result = factor.compute(s_net_inflow=s_net_inflow, T=20)
        assert isinstance(result, pd.DataFrame)

    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=15, freq="B")
        stocks = ["A", "B", "C"]

        np.random.seed(0)
        s_net_inflow = pd.DataFrame(np.random.randn(15, 3), index=dates, columns=stocks)

        result = factor.compute(s_net_inflow=s_net_inflow, T=10)
        assert result.shape == s_net_inflow.shape

    def test_output_columns_preserved(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        stocks = ["000001.SZ", "600000.SH"]

        np.random.seed(0)
        s_net_inflow = pd.DataFrame(np.random.randn(10, 2), index=dates, columns=stocks)

        result = factor.compute(s_net_inflow=s_net_inflow, T=5)
        assert list(result.columns) == stocks
