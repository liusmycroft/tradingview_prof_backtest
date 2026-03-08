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
    "factors.foreign_ops", _factors_dir / "foreign_ops.py"
)
_mod = importlib.util.module_from_spec(_mod_spec)
sys.modules["factors.foreign_ops"] = _mod
_mod_spec.loader.exec_module(_mod)

ForeignOpsFactor = _mod.ForeignOpsFactor


@pytest.fixture
def factor():
    return ForeignOpsFactor()


class TestForeignOpsMetadata:
    def test_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "FOREIGN_OPS"
        assert meta["category"] == "动量溢出"
        assert meta["description"] != ""

    def test_repr(self, factor):
        assert "ForeignOpsFactor" in repr(factor)


class TestForeignOpsCompute:
    def test_known_values(self, factor):
        """手动验证加权求和"""
        stocks = ["S1", "S2"]
        countries = ["US", "JP"]

        ratio = pd.DataFrame(
            [[0.3, 0.2], [0.5, 0.1]],
            index=stocks, columns=countries,
        )
        ret = pd.Series([0.05, -0.02], index=countries)

        result = factor.compute(foreign_sales_ratio=ratio, country_industry_return=ret)

        # S1: 0.3*0.05 + 0.2*(-0.02) = 0.015 - 0.004 = 0.011
        # S2: 0.5*0.05 + 0.1*(-0.02) = 0.025 - 0.002 = 0.023
        assert pytest.approx(result["S1"], rel=1e-6) == 0.011
        assert pytest.approx(result["S2"], rel=1e-6) == 0.023

    def test_zero_ratio(self, factor):
        """营收占比全为0时，因子值应为0"""
        stocks = ["S1"]
        countries = ["US", "JP"]

        ratio = pd.DataFrame([[0.0, 0.0]], index=stocks, columns=countries)
        ret = pd.Series([0.05, -0.02], index=countries)

        result = factor.compute(foreign_sales_ratio=ratio, country_industry_return=ret)
        assert pytest.approx(result["S1"], abs=1e-10) == 0.0

    def test_single_country(self, factor):
        """单一国家"""
        stocks = ["S1", "S2"]
        countries = ["US"]

        ratio = pd.DataFrame([[0.4], [0.6]], index=stocks, columns=countries)
        ret = pd.Series([0.10], index=countries)

        result = factor.compute(foreign_sales_ratio=ratio, country_industry_return=ret)
        assert pytest.approx(result["S1"], rel=1e-6) == 0.04
        assert pytest.approx(result["S2"], rel=1e-6) == 0.06

    def test_no_common_countries(self, factor):
        """无共同国家时应返回0"""
        stocks = ["S1"]
        ratio = pd.DataFrame([[0.5]], index=stocks, columns=["US"])
        ret = pd.Series([0.05], index=["JP"])

        result = factor.compute(foreign_sales_ratio=ratio, country_industry_return=ret)
        assert pytest.approx(result["S1"], abs=1e-10) == 0.0

    def test_output_type(self, factor):
        """输出应为 pd.Series"""
        stocks = ["S1", "S2", "S3"]
        countries = ["US", "JP"]

        ratio = pd.DataFrame(
            np.random.rand(3, 2), index=stocks, columns=countries
        )
        ret = pd.Series([0.05, -0.02], index=countries)

        result = factor.compute(foreign_sales_ratio=ratio, country_industry_return=ret)
        assert isinstance(result, pd.Series)
        assert len(result) == len(stocks)
        assert list(result.index) == stocks

    def test_dataframe_return_input(self, factor):
        """country_industry_return 为 DataFrame 时也应正常工作"""
        stocks = ["S1"]
        countries = ["US", "JP"]

        ratio = pd.DataFrame([[0.3, 0.2]], index=stocks, columns=countries)
        ret = pd.DataFrame({"return": [0.05, -0.02]}, index=countries)

        result = factor.compute(foreign_sales_ratio=ratio, country_industry_return=ret)
        expected = 0.3 * 0.05 + 0.2 * (-0.02)
        assert pytest.approx(result["S1"], rel=1e-6) == expected
