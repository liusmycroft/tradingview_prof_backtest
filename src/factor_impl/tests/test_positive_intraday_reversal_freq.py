import numpy as np
import pandas as pd
import pytest

from factors.positive_intraday_reversal_freq import PositiveIntradayReversalFreqFactor


@pytest.fixture
def factor():
    return PositiveIntradayReversalFreqFactor()


class TestPositiveIntradayReversalFreqMetadata:
    def test_name(self, factor):
        assert factor.name == "POSITIVE_INTRADAY_REVERSAL_FREQ"

    def test_category(self, factor):
        assert factor.category == "高频收益分布"

    def test_repr(self, factor):
        assert "POSITIVE_INTRADAY_REVERSAL_FREQ" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "POSITIVE_INTRADAY_REVERSAL_FREQ"
        assert meta["category"] == "高频收益分布"


class TestPositiveIntradayReversalFreqHandCalculated:
    """手算验证 PR = rolling_mean(I{RET_CO<0} * I{RET_OC>0}, T)"""

    def test_T3_manual(self, factor):
        """T=3, 手动验证。

        ret_co = [-0.01, 0.02, -0.03, -0.01, 0.01]
        ret_oc = [ 0.02, 0.01, -0.01,  0.03, 0.02]

        indicator:
          day0: co<0 & oc>0 -> True  = 1
          day1: co>0         -> False = 0
          day2: co<0 & oc<0 -> False = 0
          day3: co<0 & oc>0 -> True  = 1
          day4: co>0         -> False = 0

        T=3:
          row 0,1: NaN
          row 2: mean(1, 0, 0) = 1/3
          row 3: mean(0, 0, 1) = 1/3
          row 4: mean(0, 1, 0) = 1/3
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        ret_co = pd.DataFrame([-0.01, 0.02, -0.03, -0.01, 0.01], index=dates, columns=stocks)
        ret_oc = pd.DataFrame([0.02, 0.01, -0.01, 0.03, 0.02], index=dates, columns=stocks)

        result = factor.compute(ret_co=ret_co, ret_oc=ret_oc, T=3)

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(1 / 3, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(1 / 3, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(1 / 3, rel=1e-10)

    def test_all_reversals(self, factor):
        """所有天都发生正向逆转时，频率应为 1.0。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        ret_co = pd.DataFrame([-0.01] * 5, index=dates, columns=stocks)
        ret_oc = pd.DataFrame([0.02] * 5, index=dates, columns=stocks)

        result = factor.compute(ret_co=ret_co, ret_oc=ret_oc, T=3)
        assert result.iloc[2, 0] == pytest.approx(1.0, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(1.0, rel=1e-10)

    def test_no_reversals(self, factor):
        """没有正向逆转时，频率应为 0.0。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        ret_co = pd.DataFrame([0.01] * 5, index=dates, columns=stocks)
        ret_oc = pd.DataFrame([0.02] * 5, index=dates, columns=stocks)

        result = factor.compute(ret_co=ret_co, ret_oc=ret_oc, T=3)
        assert result.iloc[2, 0] == pytest.approx(0.0, abs=1e-15)
        assert result.iloc[4, 0] == pytest.approx(0.0, abs=1e-15)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A", "B"]

        ret_co = pd.DataFrame(
            {"A": [-0.01] * 5, "B": [0.01] * 5}, index=dates
        )
        ret_oc = pd.DataFrame(
            {"A": [0.02] * 5, "B": [0.02] * 5}, index=dates
        )

        result = factor.compute(ret_co=ret_co, ret_oc=ret_oc, T=3)
        assert result.iloc[2, 0] == pytest.approx(1.0, rel=1e-10)  # A: all reversals
        assert result.iloc[2, 1] == pytest.approx(0.0, abs=1e-15)  # B: no reversals


class TestPositiveIntradayReversalFreqEdgeCases:
    def test_nan_in_input(self, factor):
        """输入含 NaN 时不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        ret_co = pd.DataFrame([-0.01, np.nan, -0.03, -0.01, 0.01], index=dates, columns=stocks)
        ret_oc = pd.DataFrame([0.02, 0.01, -0.01, 0.03, 0.02], index=dates, columns=stocks)

        result = factor.compute(ret_co=ret_co, ret_oc=ret_oc, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_zero_returns(self, factor):
        """收益率恰好为零时不算逆转（需严格 <0 和 >0）。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]
        ret_co = pd.DataFrame([0.0, 0.0, 0.0], index=dates, columns=stocks)
        ret_oc = pd.DataFrame([0.0, 0.0, 0.0], index=dates, columns=stocks)

        result = factor.compute(ret_co=ret_co, ret_oc=ret_oc, T=3)
        assert result.iloc[2, 0] == pytest.approx(0.0, abs=1e-15)

    def test_insufficient_window(self, factor):
        """数据不足 T 天时应返回 NaN。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]
        ret_co = pd.DataFrame([-0.01] * 3, index=dates, columns=stocks)
        ret_oc = pd.DataFrame([0.02] * 3, index=dates, columns=stocks)

        result = factor.compute(ret_co=ret_co, ret_oc=ret_oc, T=5)
        assert result.isna().all().all()


class TestPositiveIntradayReversalFreqOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        ret_co = pd.DataFrame(np.random.randn(30, 3) * 0.02, index=dates, columns=stocks)
        ret_oc = pd.DataFrame(np.random.randn(30, 3) * 0.02, index=dates, columns=stocks)

        result = factor.compute(ret_co=ret_co, ret_oc=ret_oc, T=20)
        assert result.shape == ret_co.shape
        assert list(result.columns) == list(ret_co.columns)
        assert list(result.index) == list(ret_co.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        ret_co = pd.DataFrame([-0.01] * 5, index=dates, columns=stocks)
        ret_oc = pd.DataFrame([0.02] * 5, index=dates, columns=stocks)

        result = factor.compute(ret_co=ret_co, ret_oc=ret_oc, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        """前 T-1 行应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        T = 5
        ret_co = pd.DataFrame(np.random.randn(10, 2) * 0.02, index=dates, columns=stocks)
        ret_oc = pd.DataFrame(np.random.randn(10, 2) * 0.02, index=dates, columns=stocks)

        result = factor.compute(ret_co=ret_co, ret_oc=ret_oc, T=T)
        assert result.iloc[: T - 1].isna().all().all()
