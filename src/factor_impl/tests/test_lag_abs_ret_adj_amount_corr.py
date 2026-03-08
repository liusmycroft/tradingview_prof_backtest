import numpy as np
import pandas as pd
import pytest

from factors.lag_abs_ret_adj_amount_corr import LagAbsRetAdjAmountCorrFactor


@pytest.fixture
def factor():
    return LagAbsRetAdjAmountCorrFactor()


class TestLagAbsRetAdjAmountCorrMetadata:
    def test_name(self, factor):
        assert factor.name == "LAG_ABS_RET_ADJ_AMOUNT_CORR"

    def test_category(self, factor):
        assert factor.category == "高频因子-量价相关性类"

    def test_repr(self, factor):
        assert "LAG_ABS_RET_ADJ_AMOUNT_CORR" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "LAG_ABS_RET_ADJ_AMOUNT_CORR"


class TestLagAbsRetAdjAmountCorrCompute:
    def test_constant_input(self, factor):
        """常数输入时均值等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        daily_corr = pd.DataFrame(0.5, index=dates, columns=stocks)

        result = factor.compute(daily_corr=daily_corr, T=10)
        np.testing.assert_array_almost_equal(result["A"].values, 0.5)

    def test_rolling_mean(self, factor):
        """验证滚动均值计算。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_corr = pd.DataFrame([0.1, 0.2, 0.3, 0.4, 0.5], index=dates, columns=stocks)

        result = factor.compute(daily_corr=daily_corr, T=3)
        # T=3 rolling mean: [0.1, 0.15, 0.2, 0.3, 0.4]
        assert result.iloc[2, 0] == pytest.approx(0.2, rel=1e-6)
        assert result.iloc[4, 0] == pytest.approx(0.4, rel=1e-6)

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A", "B"]
        daily_corr = pd.DataFrame(np.random.randn(20, 2), index=dates, columns=stocks)

        result = factor.compute(daily_corr=daily_corr, T=10)
        assert result.shape == daily_corr.shape
        assert isinstance(result, pd.DataFrame)

    def test_min_periods_1(self, factor):
        """min_periods=1, 第一行就有值。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_corr = pd.DataFrame([0.1, 0.2, 0.3, 0.4, 0.5], index=dates, columns=stocks)

        result = factor.compute(daily_corr=daily_corr, T=10)
        assert result.iloc[0].notna().all()
