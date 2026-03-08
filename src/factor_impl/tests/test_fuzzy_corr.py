import numpy as np
import pandas as pd
import pytest

from factors.fuzzy_corr import FuzzyCorrFactor


@pytest.fixture
def factor():
    return FuzzyCorrFactor()


class TestFuzzyCorrMetadata:
    def test_name(self, factor):
        assert factor.name == "FUZZY_CORR"

    def test_category(self, factor):
        assert factor.category == "高频因子-量价相关性类"

    def test_repr(self, factor):
        assert "FUZZY_CORR" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "FUZZY_CORR"


class TestFuzzyCorrCompute:
    def test_constant_input(self, factor):
        """常数输入时 std=0, 因子 = 0.5 * mean + 0.5 * 0 = 0.5 * const。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(0.6, index=dates, columns=stocks)

        result = factor.compute(daily_fuzzy_corr=daily, T=20)
        # rolling std of constant = 0 (fillna(0)), mean = 0.6
        # result = 0.5 * 0.6 + 0.5 * 0 = 0.3
        assert result.iloc[-1, 0] == pytest.approx(0.3, rel=1e-6)

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        daily = pd.DataFrame(np.random.randn(30, 2), index=dates, columns=stocks)

        result = factor.compute(daily_fuzzy_corr=daily, T=20)
        assert result.shape == daily.shape
        assert isinstance(result, pd.DataFrame)

    def test_min_periods_1(self, factor):
        """min_periods=1, 第一行就有值。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame([0.1, 0.2, 0.3, 0.4, 0.5], index=dates, columns=stocks)

        result = factor.compute(daily_fuzzy_corr=daily, T=20)
        assert result.iloc[0].notna().all()
