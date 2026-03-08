import numpy as np
import pandas as pd
import pytest

from factors.negative_reversal_freq import NegativeReversalFreqFactor


@pytest.fixture
def factor():
    return NegativeReversalFreqFactor()


class TestNegativeReversalFreqMetadata:
    def test_name(self, factor):
        assert factor.name == "NEGATIVE_REVERSAL_FREQ"

    def test_category(self, factor):
        assert factor.category == "高频收益分布"

    def test_repr(self, factor):
        assert "NEGATIVE_REVERSAL_FREQ" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "NEGATIVE_REVERSAL_FREQ"
        assert meta["category"] == "高频收益分布"


class TestNegativeReversalFreqHandCalculated:
    """手算验证 rolling mean of indicator(overnight<0 & intraday<0)。"""

    def test_all_negative_reversal(self, factor):
        """所有日期隔夜和日内都为负，频率应为 1.0。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        overnight = pd.DataFrame(-0.01, index=dates, columns=stocks)
        intraday = pd.DataFrame(-0.02, index=dates, columns=stocks)

        result = factor.compute(overnight_ret=overnight, intraday_ret=intraday, T=20)
        assert result.iloc[-1, 0] == pytest.approx(1.0, rel=1e-10)

    def test_no_negative_reversal(self, factor):
        """隔夜为正，频率应为 0.0。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        overnight = pd.DataFrame(0.01, index=dates, columns=stocks)
        intraday = pd.DataFrame(-0.02, index=dates, columns=stocks)

        result = factor.compute(overnight_ret=overnight, intraday_ret=intraday, T=20)
        assert result.iloc[-1, 0] == pytest.approx(0.0, abs=1e-15)

    def test_half_negative_T4(self, factor):
        """T=4, 前2天满足条件, 后2天不满足, 频率=0.5。"""
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        stocks = ["A"]
        overnight = pd.DataFrame([-0.01, -0.01, 0.01, 0.01], index=dates, columns=stocks)
        intraday = pd.DataFrame([-0.02, -0.02, -0.02, -0.02], index=dates, columns=stocks)

        result = factor.compute(overnight_ret=overnight, intraday_ret=intraday, T=4)
        assert result.iloc[-1, 0] == pytest.approx(0.5, rel=1e-10)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        overnight = pd.DataFrame(
            {"A": [-0.01] * 25, "B": [0.01] * 25}, index=dates
        )
        intraday = pd.DataFrame(
            {"A": [-0.02] * 25, "B": [-0.02] * 25}, index=dates
        )

        result = factor.compute(overnight_ret=overnight, intraday_ret=intraday, T=20)
        assert result.iloc[-1, 0] == pytest.approx(1.0, rel=1e-10)  # A: all negative
        assert result.iloc[-1, 1] == pytest.approx(0.0, abs=1e-15)  # B: overnight positive

    def test_manual_T3(self, factor):
        """T=3, 手动验证滚动均值。

        indicator = [1, 0, 1, 1, 0]
        T=3:
          row 0,1: NaN
          row 2: mean(1,0,1) = 2/3
          row 3: mean(0,1,1) = 2/3
          row 4: mean(1,1,0) = 2/3
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        overnight = pd.DataFrame([-1, 1, -1, -1, 1], index=dates, columns=stocks, dtype=float)
        intraday = pd.DataFrame([-1, -1, -1, -1, -1], index=dates, columns=stocks, dtype=float)

        result = factor.compute(overnight_ret=overnight, intraday_ret=intraday, T=3)

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(2 / 3, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(2 / 3, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(2 / 3, rel=1e-10)


class TestNegativeReversalFreqEdgeCases:
    def test_nan_in_input(self, factor):
        """输入含 NaN 时不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        overnight = pd.DataFrame(np.ones(10) * -0.01, index=dates, columns=stocks)
        overnight.iloc[3, 0] = np.nan
        intraday = pd.DataFrame(-0.02, index=dates, columns=stocks)

        result = factor.compute(overnight_ret=overnight, intraday_ret=intraday, T=5)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        """全 NaN 输入时结果应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        overnight = pd.DataFrame(np.nan, index=dates, columns=stocks)
        intraday = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(overnight_ret=overnight, intraday_ret=intraday, T=5)
        assert result.isna().all().all()

    def test_zero_returns(self, factor):
        """收益率为零时不满足 < 0 条件，频率应为 0。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        overnight = pd.DataFrame(0.0, index=dates, columns=stocks)
        intraday = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(overnight_ret=overnight, intraday_ret=intraday, T=20)
        assert result.iloc[-1, 0] == pytest.approx(0.0, abs=1e-15)

    def test_insufficient_window(self, factor):
        """数据不足 T 天时应返回 NaN。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        overnight = pd.DataFrame(-0.01, index=dates, columns=stocks)
        intraday = pd.DataFrame(-0.02, index=dates, columns=stocks)

        result = factor.compute(overnight_ret=overnight, intraday_ret=intraday, T=20)
        assert result.isna().all().all()


class TestNegativeReversalFreqOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        overnight = pd.DataFrame(
            np.random.uniform(-0.05, 0.05, (30, 3)), index=dates, columns=stocks
        )
        intraday = pd.DataFrame(
            np.random.uniform(-0.05, 0.05, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(overnight_ret=overnight, intraday_ret=intraday, T=20)

        assert result.shape == overnight.shape
        assert list(result.columns) == list(overnight.columns)
        assert list(result.index) == list(overnight.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        overnight = pd.DataFrame(-0.01, index=dates, columns=stocks)
        intraday = pd.DataFrame(-0.02, index=dates, columns=stocks)

        result = factor.compute(overnight_ret=overnight, intraday_ret=intraday, T=20)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        """前 T-1 行应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        T = 20
        overnight = pd.DataFrame(-0.01, index=dates, columns=stocks)
        intraday = pd.DataFrame(-0.02, index=dates, columns=stocks)

        result = factor.compute(overnight_ret=overnight, intraday_ret=intraday, T=T)

        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()

    def test_values_between_0_and_1(self, factor):
        """频率值应在 [0, 1] 之间。"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        overnight = pd.DataFrame(
            np.random.uniform(-0.05, 0.05, (30, 3)), index=dates, columns=stocks
        )
        intraday = pd.DataFrame(
            np.random.uniform(-0.05, 0.05, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(overnight_ret=overnight, intraday_ret=intraday, T=20)
        valid = result.dropna()
        assert (valid >= 0).all().all()
        assert (valid <= 1).all().all()
