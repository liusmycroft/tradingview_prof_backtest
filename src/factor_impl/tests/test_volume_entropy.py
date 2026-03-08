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
    "factors.volume_entropy", _factors_dir / "volume_entropy.py"
)
_mod = importlib.util.module_from_spec(_mod_spec)
sys.modules["factors.volume_entropy"] = _mod
_mod_spec.loader.exec_module(_mod)

VolumeEntropyFactor = _mod.VolumeEntropyFactor


@pytest.fixture
def factor():
    return VolumeEntropyFactor()


# --------------------------------------------------------------------------- #
# 基本元信息
# --------------------------------------------------------------------------- #

class TestMetadata:
    def test_name(self, factor):
        assert factor.name == "VOLUME_ENTROPY"

    def test_category(self, factor):
        assert factor.category == "高频成交分布"

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "VOLUME_ENTROPY"
        assert "description" in meta


# --------------------------------------------------------------------------- #
# 手工验算
# --------------------------------------------------------------------------- #

class TestHandCalculation:
    def test_constant_entropy_zero_std(self, factor):
        """常数熵值，滚动标准差应为 0（或 NaN for single element）。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        stocks = ["A"]

        daily_entropy = pd.DataFrame(1.5, index=dates, columns=stocks)

        result = factor.compute(daily_entropy=daily_entropy, T=3)

        # rolling(3, min_periods=1).std():
        #   day0: std([1.5]) = NaN (single element)
        #   day1: std([1.5, 1.5]) = 0.0
        #   day2: std([1.5, 1.5, 1.5]) = 0.0
        assert np.isnan(result.iloc[0, 0]) or result.iloc[0, 0] == 0.0
        np.testing.assert_almost_equal(result.iloc[2, 0], 0.0)

    def test_varying_entropy(self, factor):
        """变化熵值，手工计算滚动标准差。"""
        dates = pd.date_range("2024-01-01", periods=4, freq="B")
        stocks = ["A"]

        daily_entropy = pd.DataFrame([1.0, 2.0, 3.0, 4.0], index=dates, columns=stocks, dtype=float)

        result = factor.compute(daily_entropy=daily_entropy, T=3)

        # rolling(3, min_periods=1).std():
        #   day0: std([1.0]) = NaN
        #   day1: std([1.0, 2.0]) = 0.7071...
        #   day2: std([1.0, 2.0, 3.0]) = 1.0
        #   day3: std([2.0, 3.0, 4.0]) = 1.0
        np.testing.assert_almost_equal(result.iloc[1, 0], np.std([1.0, 2.0], ddof=1))
        np.testing.assert_almost_equal(result.iloc[2, 0], 1.0)
        np.testing.assert_almost_equal(result.iloc[3, 0], 1.0)

    def test_multiple_stocks(self, factor):
        """多只股票独立计算。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="B")

        daily_entropy = pd.DataFrame(
            {"A": [1.0, 2.0, 3.0], "B": [10.0, 10.0, 10.0]}, index=dates
        )

        result = factor.compute(daily_entropy=daily_entropy, T=3)

        # B 常数 => std = 0 (day2 onwards)
        np.testing.assert_almost_equal(result.loc[dates[2], "B"], 0.0)
        # A 变化 => std > 0
        assert result.loc[dates[2], "A"] > 0


# --------------------------------------------------------------------------- #
# 边界情况
# --------------------------------------------------------------------------- #

class TestEdgeCases:
    def test_nan_propagation(self, factor):
        """输入含 NaN 时不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=4, freq="B")
        stocks = ["A"]

        daily_entropy = pd.DataFrame([1.0, np.nan, 3.0, 4.0], index=dates, columns=stocks)

        result = factor.compute(daily_entropy=daily_entropy, T=3)

        assert result.shape == (4, 1)

    def test_t_equals_one(self, factor):
        """T=1 时，单元素标准差为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        stocks = ["A"]

        daily_entropy = pd.DataFrame([1.0, 2.0, 3.0], index=dates, columns=stocks, dtype=float)

        result = factor.compute(daily_entropy=daily_entropy, T=1)

        # rolling(1).std() => NaN for each single element
        assert result.isna().all().all()

    def test_all_nan(self, factor):
        """全 NaN 输入不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        stocks = ["A"]

        daily_entropy = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_entropy=daily_entropy, T=2)

        assert result.shape == (3, 1)
        assert result.isna().all().all()


# --------------------------------------------------------------------------- #
# 输出形状与类型
# --------------------------------------------------------------------------- #

class TestOutputShape:
    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        stocks = ["A", "B", "C"]

        daily_entropy = pd.DataFrame(1.0, index=dates, columns=stocks)

        result = factor.compute(daily_entropy=daily_entropy)
        assert isinstance(result, pd.DataFrame)

    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        stocks = ["A", "B"]

        daily_entropy = pd.DataFrame(1.0, index=dates, columns=stocks)

        result = factor.compute(daily_entropy=daily_entropy, T=5)
        assert result.shape == daily_entropy.shape

    def test_output_index_and_columns_preserved(self, factor):
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        stocks = ["000001.SZ", "600000.SH"]

        daily_entropy = pd.DataFrame(1.0, index=dates, columns=stocks)

        result = factor.compute(daily_entropy=daily_entropy)
        pd.testing.assert_index_equal(result.index, dates)
        assert list(result.columns) == stocks
