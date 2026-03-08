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
    "factors.supplier_concentration", _factors_dir / "supplier_concentration.py"
)
_mod = importlib.util.module_from_spec(_mod_spec)
sys.modules["factors.supplier_concentration"] = _mod
_mod_spec.loader.exec_module(_mod)

SupplierConcentrationFactor = _mod.SupplierConcentrationFactor


@pytest.fixture
def factor():
    return SupplierConcentrationFactor()


# --------------------------------------------------------------------------- #
# 基本元信息
# --------------------------------------------------------------------------- #

class TestMetadata:
    def test_name(self, factor):
        assert factor.name == "SUPPLIER_CONCENTRATION"

    def test_category(self, factor):
        assert factor.category == "网络结构"

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "SUPPLIER_CONCENTRATION"
        assert "description" in meta


# --------------------------------------------------------------------------- #
# 手工验算
# --------------------------------------------------------------------------- #

class TestHandCalculation:
    def test_single_company_single_industry(self, factor):
        """单公司单行业：SH = w * H"""
        weights = pd.DataFrame({"行业1": [1.0]}, index=["公司A"])
        herfindahl = pd.Series({"行业1": 0.25})

        result = factor.compute(procurement_weights=weights, herfindahl=herfindahl)

        np.testing.assert_almost_equal(result["公司A"], 0.25)

    def test_two_industries(self, factor):
        """两个行业加权求和。"""
        weights = pd.DataFrame(
            {"行业1": [0.6], "行业2": [0.4]}, index=["公司A"]
        )
        herfindahl = pd.Series({"行业1": 0.25, "行业2": 0.10})

        result = factor.compute(procurement_weights=weights, herfindahl=herfindahl)

        # SH = 0.6 * 0.25 + 0.4 * 0.10 = 0.15 + 0.04 = 0.19
        np.testing.assert_almost_equal(result["公司A"], 0.19)

    def test_multiple_companies(self, factor):
        """多家公司并行计算。"""
        weights = pd.DataFrame(
            {"行业1": [0.5, 0.1], "行业2": [0.3, 0.6], "行业3": [0.2, 0.3]},
            index=["公司A", "公司B"],
        )
        herfindahl = pd.Series({"行业1": 0.25, "行业2": 0.10, "行业3": 0.40})

        result = factor.compute(procurement_weights=weights, herfindahl=herfindahl)

        # 公司A: 0.5*0.25 + 0.3*0.10 + 0.2*0.40 = 0.125 + 0.03 + 0.08 = 0.235
        # 公司B: 0.1*0.25 + 0.6*0.10 + 0.3*0.40 = 0.025 + 0.06 + 0.12 = 0.205
        np.testing.assert_almost_equal(result["公司A"], 0.235)
        np.testing.assert_almost_equal(result["公司B"], 0.205)


# --------------------------------------------------------------------------- #
# 边界情况
# --------------------------------------------------------------------------- #

class TestEdgeCases:
    def test_zero_weights(self, factor):
        """权重全为零时，SH = 0。"""
        weights = pd.DataFrame({"行业1": [0.0], "行业2": [0.0]}, index=["公司A"])
        herfindahl = pd.Series({"行业1": 0.25, "行业2": 0.10})

        result = factor.compute(procurement_weights=weights, herfindahl=herfindahl)

        np.testing.assert_almost_equal(result["公司A"], 0.0)

    def test_partial_industry_overlap(self, factor):
        """权重矩阵和赫芬达尔指数只有部分行业重叠。"""
        weights = pd.DataFrame(
            {"行业1": [0.5], "行业X": [0.5]}, index=["公司A"]
        )
        herfindahl = pd.Series({"行业1": 0.25, "行业Y": 0.10})

        result = factor.compute(procurement_weights=weights, herfindahl=herfindahl)

        # 只有行业1重叠: SH = 0.5 * 0.25 = 0.125
        np.testing.assert_almost_equal(result["公司A"], 0.125)

    def test_zero_herfindahl(self, factor):
        """赫芬达尔指数为零时，SH = 0。"""
        weights = pd.DataFrame({"行业1": [1.0]}, index=["公司A"])
        herfindahl = pd.Series({"行业1": 0.0})

        result = factor.compute(procurement_weights=weights, herfindahl=herfindahl)

        np.testing.assert_almost_equal(result["公司A"], 0.0)


# --------------------------------------------------------------------------- #
# 输出形状与类型
# --------------------------------------------------------------------------- #

class TestOutputShape:
    def test_output_is_series(self, factor):
        weights = pd.DataFrame({"行业1": [0.5, 0.5]}, index=["公司A", "公司B"])
        herfindahl = pd.Series({"行业1": 0.25})

        result = factor.compute(procurement_weights=weights, herfindahl=herfindahl)
        assert isinstance(result, pd.Series)

    def test_output_length_matches_companies(self, factor):
        companies = ["A", "B", "C"]
        weights = pd.DataFrame({"行业1": [0.5] * 3, "行业2": [0.5] * 3}, index=companies)
        herfindahl = pd.Series({"行业1": 0.25, "行业2": 0.10})

        result = factor.compute(procurement_weights=weights, herfindahl=herfindahl)
        assert len(result) == 3

    def test_output_index_matches_companies(self, factor):
        companies = ["000001.SZ", "600000.SH"]
        weights = pd.DataFrame({"行业1": [0.5, 0.5]}, index=companies)
        herfindahl = pd.Series({"行业1": 0.25})

        result = factor.compute(procurement_weights=weights, herfindahl=herfindahl)
        assert list(result.index) == companies
