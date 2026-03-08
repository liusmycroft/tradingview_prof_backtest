import numpy as np
import pandas as pd
import pytest

from factors.pvol_ret_corr import PvolRetCorrFactor


@pytest.fixture
def factor():
    return PvolRetCorrFactor()


class TestPvolRetCorrMetadata:
    def test_name(self, factor):
        assert factor.name == "PVOL_RET_CORR"

    def test_category(self, factor):
        assert factor.category == "高频量价"

    def test_repr(self, factor):
        assert "PVOL_RET_CORR" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "PVOL_RET_CORR"
        assert meta["category"] == "高频量价"


class TestPvolRetCorrHandCalculated:
    """用手算数据验证每笔成交量收益率相关性因子。"""

    def test_constant_corr(self, factor):
        """常数相关性 0.3, T=3 => 均值 = 0.3。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        corr = pd.DataFrame(0.3, index=dates, columns=stocks)

        result = factor.compute(daily_pvol_corr=corr, T=3)

        # min_periods=T=3, 前2行NaN
        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(0.3, rel=1e-10)

    def test_varying_corr_T3(self, factor):
        """T=3, 变化的相关性。

        corr = [0.2, 0.4, 0.6]
        rolling(3, min_periods=3):
          day2: mean([0.2, 0.4, 0.6]) = 0.4
        """
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        corr = pd.DataFrame([0.2, 0.4, 0.6], index=dates, columns=stocks)

        result = factor.compute(daily_pvol_corr=corr, T=3)

        assert result.iloc[2, 0] == pytest.approx(0.4, rel=1e-10)

    def test_negative_corr(self, factor):
        """负相关性 -0.5, T=3 => 均值 = -0.5。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        corr = pd.DataFrame(-0.5, index=dates, columns=stocks)

        result = factor.compute(daily_pvol_corr=corr, T=3)

        assert result.iloc[2, 0] == pytest.approx(-0.5, rel=1e-10)

    def test_mixed_corr_T4(self, factor):
        """T=4, 混合正负相关性。

        corr = [0.2, 0.8, -0.2, -0.8]
        mean = 0.0
        """
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        stocks = ["A"]

        corr = pd.DataFrame([0.2, 0.8, -0.2, -0.8], index=dates, columns=stocks)

        result = factor.compute(daily_pvol_corr=corr, T=4)

        assert result.iloc[3, 0] == pytest.approx(0.0, abs=1e-10)

    def test_two_stocks(self, factor):
        """两只股票并行计算。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A", "B"]

        corr = pd.DataFrame(
            {"A": [0.1, 0.2, 0.3], "B": [-0.1, -0.2, -0.3]}, index=dates
        )

        result = factor.compute(daily_pvol_corr=corr, T=3)

        assert result.loc[dates[2], "A"] == pytest.approx(0.2, rel=1e-10)
        assert result.loc[dates[2], "B"] == pytest.approx(-0.2, rel=1e-10)

    def test_rolling_window_slides(self, factor):
        """验证滚动窗口正确滑动 (T=3)。

        corr = [1, 2, 3, 4, 5]
        rolling(3):
          day2: mean(1,2,3) = 2
          day3: mean(2,3,4) = 3
          day4: mean(3,4,5) = 4
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        corr = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0], index=dates, columns=stocks)

        result = factor.compute(daily_pvol_corr=corr, T=3)

        assert result.iloc[2, 0] == pytest.approx(2.0, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(3.0, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(4.0, rel=1e-10)


class TestPvolRetCorrEdgeCases:
    def test_nan_in_corr(self, factor):
        """相关性中含 NaN 时, 窗口内有 NaN 结果为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        corr = pd.DataFrame([0.3, np.nan, 0.5], index=dates, columns=stocks)

        result = factor.compute(daily_pvol_corr=corr, T=3)
        assert np.isnan(result.iloc[2, 0])

    def test_all_zero_corr(self, factor):
        """相关性全为 0 时, 因子值应为 0。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        corr = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_pvol_corr=corr, T=3)
        assert result.iloc[2, 0] == pytest.approx(0.0, abs=1e-15)

    def test_insufficient_data(self, factor):
        """数据不足 T 天时, 全部为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        corr = pd.DataFrame(0.3, index=dates, columns=stocks)

        result = factor.compute(daily_pvol_corr=corr, T=20)
        assert result.isna().all().all()

    def test_extreme_values(self, factor):
        """极端相关性值 (-1, 1) 不应出错。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        corr = pd.DataFrame([-1.0, 1.0, -1.0], index=dates, columns=stocks)

        result = factor.compute(daily_pvol_corr=corr, T=3)
        expected = (-1.0 + 1.0 + -1.0) / 3
        assert result.iloc[2, 0] == pytest.approx(expected, rel=1e-10)


class TestPvolRetCorrOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        stocks = ["A", "B", "C"]

        corr = pd.DataFrame(
            np.random.uniform(-1, 1, (50, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_pvol_corr=corr, T=20)

        assert result.shape == corr.shape
        assert list(result.columns) == list(corr.columns)
        assert list(result.index) == list(corr.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        corr = pd.DataFrame(0.3, index=dates, columns=stocks)

        result = factor.compute(daily_pvol_corr=corr, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        """前 T-1 行应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        T = 5

        corr = pd.DataFrame(
            np.random.uniform(-1, 1, (10, 2)), index=dates, columns=stocks
        )

        result = factor.compute(daily_pvol_corr=corr, T=T)

        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
