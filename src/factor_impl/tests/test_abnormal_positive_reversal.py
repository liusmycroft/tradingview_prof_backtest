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
    "factors.abnormal_positive_reversal", _factors_dir / "abnormal_positive_reversal.py"
)
_mod = importlib.util.module_from_spec(_mod_spec)
sys.modules["factors.abnormal_positive_reversal"] = _mod
_mod_spec.loader.exec_module(_mod)

AbnormalPositiveReversalFactor = _mod.AbnormalPositiveReversalFactor


@pytest.fixture
def factor():
    return AbnormalPositiveReversalFactor()


class TestABPRMetadata:
    def test_name(self, factor):
        assert factor.name == "AB_PR"

    def test_category(self, factor):
        assert factor.category == "高频收益分布"

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "AB_PR"
        assert meta["category"] == "高频收益分布"

    def test_repr(self, factor):
        assert "AB_PR" in repr(factor)


class TestABPRCompute:
    def test_all_reversal(self, factor):
        """所有日期都发生正向逆转时，短期和长期频率都为1，AB_PR=0。"""
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        stocks = ["A"]
        ret_co = pd.DataFrame(-0.01, index=dates, columns=stocks)
        ret_oc = pd.DataFrame(0.02, index=dates, columns=stocks)

        result = factor.compute(ret_co=ret_co, ret_oc=ret_oc, T_short=20, T_long=60)

        # At day 59 (index 59), both short and long windows are full
        assert result.iloc[-1, 0] == pytest.approx(0.0, abs=1e-10)

    def test_no_reversal(self, factor):
        """无逆转时，短期和长期频率都为0，AB_PR=0。"""
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        stocks = ["A"]
        ret_co = pd.DataFrame(0.01, index=dates, columns=stocks)
        ret_oc = pd.DataFrame(0.02, index=dates, columns=stocks)

        result = factor.compute(ret_co=ret_co, ret_oc=ret_oc, T_short=20, T_long=60)
        assert result.iloc[-1, 0] == pytest.approx(0.0, abs=1e-10)

    def test_recent_burst(self, factor):
        """近期逆转频率高于长期时，AB_PR > 0。"""
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        stocks = ["A"]
        # 前40天无逆转，后20天全部逆转
        ret_co_vals = [0.01] * 40 + [-0.01] * 20
        ret_oc_vals = [0.01] * 40 + [0.02] * 20
        ret_co = pd.DataFrame(ret_co_vals, index=dates, columns=stocks)
        ret_oc = pd.DataFrame(ret_oc_vals, index=dates, columns=stocks)

        result = factor.compute(ret_co=ret_co, ret_oc=ret_oc, T_short=20, T_long=60)

        # short = 1.0, long = 20/60 = 1/3, AB_PR = 2/3
        assert result.iloc[-1, 0] == pytest.approx(2 / 3, rel=1e-6)

    def test_leading_nan(self, factor):
        """前 T_long-1 行应为 NaN（min_periods=T_long）。"""
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        stocks = ["A"]
        ret_co = pd.DataFrame(-0.01, index=dates, columns=stocks)
        ret_oc = pd.DataFrame(0.02, index=dates, columns=stocks)

        result = factor.compute(ret_co=ret_co, ret_oc=ret_oc, T_short=20, T_long=60)

        # First 59 rows should be NaN (T_long=60, min_periods=60)
        assert result.iloc[:59].isna().all().all()
        assert result.iloc[59].notna().all()

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        # Stock A: all reversal; Stock B: no reversal
        ret_co = pd.DataFrame({"A": [-0.01] * 60, "B": [0.01] * 60}, index=dates)
        ret_oc = pd.DataFrame({"A": [0.02] * 60, "B": [0.02] * 60}, index=dates)

        result = factor.compute(ret_co=ret_co, ret_oc=ret_oc, T_short=20, T_long=60)

        assert result.iloc[-1]["A"] == pytest.approx(0.0, abs=1e-10)
        assert result.iloc[-1]["B"] == pytest.approx(0.0, abs=1e-10)


class TestABPREdgeCases:
    def test_nan_propagation(self, factor):
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        ret_co_vals = [-0.01] * 60
        ret_co_vals[30] = np.nan
        ret_co = pd.DataFrame(ret_co_vals, index=dates, columns=["A"])
        ret_oc = pd.DataFrame(0.02, index=dates, columns=["A"])

        result = factor.compute(ret_co=ret_co, ret_oc=ret_oc, T_short=20, T_long=60)
        assert isinstance(result, pd.DataFrame)

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=80, freq="D")
        stocks = ["A", "B", "C"]
        ret_co = pd.DataFrame(np.random.randn(80, 3) * 0.01, index=dates, columns=stocks)
        ret_oc = pd.DataFrame(np.random.randn(80, 3) * 0.01, index=dates, columns=stocks)

        result = factor.compute(ret_co=ret_co, ret_oc=ret_oc, T_short=20, T_long=60)
        assert result.shape == ret_co.shape
        assert isinstance(result, pd.DataFrame)
