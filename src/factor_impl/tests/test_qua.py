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
    "factors.qua", _factors_dir / "qua.py"
)
_mod = importlib.util.module_from_spec(_mod_spec)
sys.modules["factors.qua"] = _mod
_mod_spec.loader.exec_module(_mod)

QUAFactor = _mod.QUAFactor


@pytest.fixture
def factor():
    return QUAFactor()


# --------------------------------------------------------------------------- #
# 基本元信息
# --------------------------------------------------------------------------- #

class TestMetadata:
    def test_name(self, factor):
        assert factor.name == "QUA"

    def test_category(self, factor):
        assert factor.category == "高频成交分布"

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "QUA"
        assert "description" in meta


# --------------------------------------------------------------------------- #
# 手工验算
# --------------------------------------------------------------------------- #

class TestHandCalculation:
    def test_constant_inputs(self, factor):
        """常数输入：daily_qua=0.3，滚动均值=0.3"""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        stocks = ["A"]

        daily_qua = pd.DataFrame(0.3, index=dates, columns=stocks)

        result = factor.compute(daily_qua=daily_qua, T=3)

        expected = pd.DataFrame(0.3, index=dates, columns=stocks)
        pd.testing.assert_frame_equal(result, expected)

    def test_varying_inputs(self, factor):
        """变化输入，手工计算滚动均值。"""
        dates = pd.date_range("2024-01-01", periods=4, freq="B")
        stocks = ["A"]

        daily_qua = pd.DataFrame([0.1, 0.2, 0.3, 0.4], index=dates, columns=stocks, dtype=float)

        result = factor.compute(daily_qua=daily_qua, T=3)

        # rolling(3, min_periods=1):
        #   day0: mean([0.1])           = 0.1
        #   day1: mean([0.1, 0.2])      = 0.15
        #   day2: mean([0.1, 0.2, 0.3]) = 0.2
        #   day3: mean([0.2, 0.3, 0.4]) = 0.3
        expected_vals = [0.1, 0.15, 0.2, 0.3]
        np.testing.assert_allclose(result["A"].values, expected_vals, atol=1e-12)

    def test_multiple_stocks(self, factor):
        """多只股票并行计算。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="B")

        daily_qua = pd.DataFrame(
            {"A": [0.1, 0.2, 0.3], "B": [0.4, 0.5, 0.6]}, index=dates
        )

        result = factor.compute(daily_qua=daily_qua, T=2)

        # A: rolling(2): [0.1, 0.15, 0.25]
        # B: rolling(2): [0.4, 0.45, 0.55]
        np.testing.assert_allclose(result["A"].values, [0.1, 0.15, 0.25], atol=1e-12)
        np.testing.assert_allclose(result["B"].values, [0.4, 0.45, 0.55], atol=1e-12)


# --------------------------------------------------------------------------- #
# 边界情况
# --------------------------------------------------------------------------- #

class TestEdgeCases:
    def test_nan_propagation(self, factor):
        """输入含 NaN 时，rolling mean(min_periods=1) 会跳过 NaN。"""
        dates = pd.date_range("2024-01-01", periods=4, freq="B")
        stocks = ["A"]

        daily_qua = pd.DataFrame([0.2, np.nan, 0.2, 0.2], index=dates, columns=stocks)

        result = factor.compute(daily_qua=daily_qua, T=2)

        # daily: [0.2, NaN, 0.2, 0.2]
        # rolling(2, min_periods=1): [0.2, 0.2, 0.2, 0.2]
        expected = pd.DataFrame([0.2, 0.2, 0.2, 0.2], index=dates, columns=stocks)
        pd.testing.assert_frame_equal(result, expected)

    def test_t_equals_one(self, factor):
        """T=1 时，结果等于原始值。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        stocks = ["A"]

        daily_qua = pd.DataFrame([0.1, 0.2, 0.3], index=dates, columns=stocks, dtype=float)

        result = factor.compute(daily_qua=daily_qua, T=1)

        pd.testing.assert_frame_equal(result, daily_qua)

    def test_all_nan(self, factor):
        """全 NaN 输入不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        stocks = ["A"]

        daily_qua = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_qua=daily_qua, T=2)

        assert result.shape == (3, 1)
        assert result.isna().all().all()


# --------------------------------------------------------------------------- #
# 输出形状与类型
# --------------------------------------------------------------------------- #

class TestOutputShape:
    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        stocks = ["A", "B", "C"]

        daily_qua = pd.DataFrame(0.5, index=dates, columns=stocks)

        result = factor.compute(daily_qua=daily_qua)
        assert isinstance(result, pd.DataFrame)

    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        stocks = ["A", "B"]

        daily_qua = pd.DataFrame(0.5, index=dates, columns=stocks)

        result = factor.compute(daily_qua=daily_qua, T=5)
        assert result.shape == daily_qua.shape

    def test_output_index_and_columns_preserved(self, factor):
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        stocks = ["000001.SZ", "600000.SH"]

        daily_qua = pd.DataFrame(0.5, index=dates, columns=stocks)

        result = factor.compute(daily_qua=daily_qua)
        pd.testing.assert_index_equal(result.index, dates)
        assert list(result.columns) == stocks
