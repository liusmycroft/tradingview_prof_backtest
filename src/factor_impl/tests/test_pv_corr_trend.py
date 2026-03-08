import numpy as np
import pandas as pd
import pytest

from factors.pv_corr_trend import PVCorrTrendFactor


@pytest.fixture
def factor():
    return PVCorrTrendFactor()


class TestPVCorrTrendMetadata:
    def test_name(self, factor):
        assert factor.name == "PV_CORR_TREND"

    def test_category(self, factor):
        assert factor.category == "高频量价相关性"

    def test_repr(self, factor):
        assert "PV_CORR_TREND" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "PV_CORR_TREND"
        assert meta["category"] == "高频量价相关性"


class TestPVCorrTrendHandCalculated:
    """用手算数据验证滚动回归斜率计算的正确性。"""

    def test_linear_increasing_corr(self, factor):
        """相关系数线性递增时, 斜率应为正常数。

        rho = [1, 2, 3, 4, 5], T=5
        t = [1, 2, 3, 4, 5]
        beta = cov(t, rho) / var(t) = 1.0
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_pv_corr = pd.DataFrame(
            [1.0, 2.0, 3.0, 4.0, 5.0], index=dates, columns=stocks
        )

        result = factor.compute(daily_pv_corr=daily_pv_corr, T=5)

        assert result.iloc[4, 0] == pytest.approx(1.0, rel=1e-6)

    def test_constant_corr_zero_slope(self, factor):
        """常数相关系数时, 斜率应为 0。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_pv_corr = pd.DataFrame(0.5, index=dates, columns=stocks)

        result = factor.compute(daily_pv_corr=daily_pv_corr, T=5)

        assert result.iloc[4, 0] == pytest.approx(0.0, abs=1e-10)

    def test_linear_decreasing_corr(self, factor):
        """相关系数线性递减时, 斜率应为负。

        rho = [5, 4, 3, 2, 1], T=5
        beta = -1.0
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_pv_corr = pd.DataFrame(
            [5.0, 4.0, 3.0, 2.0, 1.0], index=dates, columns=stocks
        )

        result = factor.compute(daily_pv_corr=daily_pv_corr, T=5)

        assert result.iloc[4, 0] == pytest.approx(-1.0, rel=1e-6)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A", "B"]
        daily_pv_corr = pd.DataFrame(
            {"A": [1.0, 2.0, 3.0, 4.0, 5.0], "B": [5.0, 4.0, 3.0, 2.0, 1.0]},
            index=dates,
        )

        result = factor.compute(daily_pv_corr=daily_pv_corr, T=5)

        assert result.iloc[4, 0] == pytest.approx(1.0, rel=1e-6)
        assert result.iloc[4, 1] == pytest.approx(-1.0, rel=1e-6)

    def test_rolling_window_T3(self, factor):
        """T=3, 验证滚动窗口。

        data = [1, 3, 2, 6, 5]
        window [1,3,2]: t=[1,2,3], rho=[1,3,2]
          t_mean=2, rho_mean=2
          beta = ((1-2)*(1-2)+(2-2)*(3-2)+(3-2)*(2-2)) / ((1-2)^2+(2-2)^2+(3-2)^2)
               = (1+0+0) / (1+0+1) = 0.5
        window [3,2,6]: t=[1,2,3], rho=[3,2,6]
          t_mean=2, rho_mean=11/3
          beta = ((1-2)*(3-11/3)+(2-2)*(2-11/3)+(3-2)*(6-11/3)) / 2
               = ((-1)*(-2/3)+0+(1)*(7/3)) / 2 = (2/3+7/3)/2 = 3/2 = 1.5
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_pv_corr = pd.DataFrame(
            [1.0, 3.0, 2.0, 6.0, 5.0], index=dates, columns=stocks
        )

        result = factor.compute(daily_pv_corr=daily_pv_corr, T=3)

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(0.5, rel=1e-6)
        assert result.iloc[3, 0] == pytest.approx(1.5, rel=1e-6)


class TestPVCorrTrendEdgeCases:
    def test_nan_in_input(self, factor):
        """输入含 NaN 时不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        values = np.ones(10) * 0.5
        values[3] = np.nan
        daily_pv_corr = pd.DataFrame(values, index=dates, columns=stocks)

        result = factor.compute(daily_pv_corr=daily_pv_corr, T=5)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        """全 NaN 输入时, 结果应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily_pv_corr = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_pv_corr=daily_pv_corr, T=5)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        """全零输入时, 斜率应为 0。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily_pv_corr = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_pv_corr=daily_pv_corr, T=5)
        valid = result.dropna()
        for val in valid["A"].values:
            assert val == pytest.approx(0.0, abs=1e-10)


class TestPVCorrTrendOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily_pv_corr = pd.DataFrame(
            np.random.uniform(-1, 1, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_pv_corr=daily_pv_corr, T=20)

        assert result.shape == daily_pv_corr.shape
        assert list(result.columns) == list(daily_pv_corr.columns)
        assert list(result.index) == list(daily_pv_corr.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily_pv_corr = pd.DataFrame(
            np.random.randn(25), index=pd.date_range("2024-01-01", periods=25, freq="D"),
            columns=stocks,
        )

        result = factor.compute(daily_pv_corr=daily_pv_corr, T=20)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        """前 T-1 行应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        T = 20
        daily_pv_corr = pd.DataFrame(
            np.random.uniform(-1, 1, (30, 2)), index=dates, columns=stocks
        )

        result = factor.compute(daily_pv_corr=daily_pv_corr, T=T)

        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
