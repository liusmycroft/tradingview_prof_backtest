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
    "factors.srvj", _factors_dir / "srvj.py"
)
_mod = importlib.util.module_from_spec(_mod_spec)
sys.modules["factors.srvj"] = _mod
_mod_spec.loader.exec_module(_mod)

SRVJFactor = _mod.SRVJFactor


@pytest.fixture
def factor():
    return SRVJFactor()


# --------------------------------------------------------------------------- #
# 基本元信息
# --------------------------------------------------------------------------- #

class TestMetadata:
    def test_name(self, factor):
        assert factor.name == "SRVJ"

    def test_category(self, factor):
        assert factor.category == "高频波动"

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "SRVJ"
        assert "description" in meta


# --------------------------------------------------------------------------- #
# 手工验算
# --------------------------------------------------------------------------- #

class TestHandCalculation:
    def test_basic_subtraction(self, factor):
        """SRVJ = RVJP - RVJN，基本减法验证。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        rvjp = pd.DataFrame({"A": [0.010, 0.008, 0.012]}, index=dates)
        rvjn = pd.DataFrame({"A": [0.003, 0.005, 0.002]}, index=dates)

        result = factor.compute(rvjp=rvjp, rvjn=rvjn)

        expected = pd.DataFrame({"A": [0.007, 0.003, 0.010]}, index=dates)
        pd.testing.assert_frame_equal(result, expected)

    def test_negative_result(self, factor):
        """RVJN > RVJP 时，SRVJ 为负。"""
        dates = pd.bdate_range("2025-01-01", periods=2)
        rvjp = pd.DataFrame({"A": [0.001, 0.002]}, index=dates)
        rvjn = pd.DataFrame({"A": [0.005, 0.010]}, index=dates)

        result = factor.compute(rvjp=rvjp, rvjn=rvjn)

        expected = pd.DataFrame({"A": [-0.004, -0.008]}, index=dates)
        pd.testing.assert_frame_equal(result, expected)

    def test_multi_stock(self, factor):
        """多只股票同时计算。"""
        dates = pd.bdate_range("2025-01-01", periods=2)
        rvjp = pd.DataFrame({"A": [0.01, 0.02], "B": [0.005, 0.001]}, index=dates)
        rvjn = pd.DataFrame({"A": [0.005, 0.005], "B": [0.002, 0.010]}, index=dates)

        result = factor.compute(rvjp=rvjp, rvjn=rvjn)

        assert result.shape == (2, 2)
        np.testing.assert_almost_equal(result.loc[dates[0], "A"], 0.005)
        np.testing.assert_almost_equal(result.loc[dates[1], "A"], 0.015)
        np.testing.assert_almost_equal(result.loc[dates[0], "B"], 0.003)
        np.testing.assert_almost_equal(result.loc[dates[1], "B"], -0.009)

    def test_zero_result(self, factor):
        """RVJP == RVJN 时，SRVJ = 0。"""
        dates = pd.bdate_range("2025-01-01", periods=2)
        rvjp = pd.DataFrame({"A": [0.005, 0.010]}, index=dates)
        rvjn = pd.DataFrame({"A": [0.005, 0.010]}, index=dates)

        result = factor.compute(rvjp=rvjp, rvjn=rvjn)

        expected = pd.DataFrame({"A": [0.0, 0.0]}, index=dates)
        pd.testing.assert_frame_equal(result, expected)


# --------------------------------------------------------------------------- #
# 边界情况
# --------------------------------------------------------------------------- #

class TestEdgeCases:
    def test_nan_propagation(self, factor):
        """输入含 NaN 时，输出对应位置也应为 NaN。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        rvjp = pd.DataFrame({"A": [0.01, np.nan, 0.005]}, index=dates)
        rvjn = pd.DataFrame({"A": [0.004, 0.006, np.nan]}, index=dates)

        result = factor.compute(rvjp=rvjp, rvjn=rvjn)

        assert not np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert np.isnan(result.iloc[2, 0])

    def test_all_zeros(self, factor):
        """全零输入，结果应全为零。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        rvjp = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=dates)
        rvjn = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=dates)

        result = factor.compute(rvjp=rvjp, rvjn=rvjn)

        expected = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=dates)
        pd.testing.assert_frame_equal(result, expected)


# --------------------------------------------------------------------------- #
# 输出形状与类型
# --------------------------------------------------------------------------- #

class TestOutputShape:
    def test_output_is_dataframe(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=5)
        stocks = ["A", "B", "C"]

        rvjp = pd.DataFrame(0.01, index=dates, columns=stocks)
        rvjn = pd.DataFrame(0.005, index=dates, columns=stocks)

        result = factor.compute(rvjp=rvjp, rvjn=rvjn)
        assert isinstance(result, pd.DataFrame)

    def test_output_shape_matches_input(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=10)
        stocks = ["A", "B"]

        rvjp = pd.DataFrame(0.01, index=dates, columns=stocks)
        rvjn = pd.DataFrame(0.005, index=dates, columns=stocks)

        result = factor.compute(rvjp=rvjp, rvjn=rvjn)
        assert result.shape == rvjp.shape

    def test_output_index_and_columns_preserved(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=3)
        stocks = ["000001.SZ", "600000.SH"]

        rvjp = pd.DataFrame(0.01, index=dates, columns=stocks)
        rvjn = pd.DataFrame(0.005, index=dates, columns=stocks)

        result = factor.compute(rvjp=rvjp, rvjn=rvjn)
        pd.testing.assert_index_equal(result.index, dates)
        assert list(result.columns) == stocks
