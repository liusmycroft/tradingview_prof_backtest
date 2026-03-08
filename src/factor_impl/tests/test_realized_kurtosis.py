import numpy as np
import pandas as pd
import pytest

from factors.realized_kurtosis import RealizedKurtosisFactor


@pytest.fixture
def factor():
    return RealizedKurtosisFactor()


class TestRealizedKurtosisMetadata:
    def test_name(self, factor):
        assert factor.name == "REALIZED_KURTOSIS"

    def test_category(self, factor):
        assert factor.category == "高频收益分布"

    def test_repr(self, factor):
        assert "REALIZED_KURTOSIS" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "REALIZED_KURTOSIS"
        assert meta["category"] == "高频收益分布"


class TestRealizedKurtosisHandCalculated:
    """用手算数据验证已实现峰度因子。"""

    def test_constant_kurtosis(self, factor):
        """常数峰度 3.5, T=3 => 均值 = 3.5。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        rkurt = pd.DataFrame(3.5, index=dates, columns=stocks)

        result = factor.compute(daily_rkurt=rkurt, T=3)

        # min_periods=T=3, 前2行NaN
        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(3.5, rel=1e-10)

    def test_varying_kurtosis_T3(self, factor):
        """T=3, 变化的峰度。

        rkurt = [3.0, 4.0, 5.0]
        rolling(3, min_periods=3):
          day2: mean([3.0, 4.0, 5.0]) = 4.0
        """
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        rkurt = pd.DataFrame([3.0, 4.0, 5.0], index=dates, columns=stocks)

        result = factor.compute(daily_rkurt=rkurt, T=3)

        assert result.iloc[2, 0] == pytest.approx(4.0, rel=1e-10)

    def test_two_stocks(self, factor):
        """两只股票并行计算。

        Stock A: rkurt = [3, 4, 5] -> mean = 4
        Stock B: rkurt = [6, 6, 6] -> mean = 6
        """
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A", "B"]

        rkurt = pd.DataFrame(
            {"A": [3.0, 4.0, 5.0], "B": [6.0, 6.0, 6.0]}, index=dates
        )

        result = factor.compute(daily_rkurt=rkurt, T=3)

        assert result.loc[dates[2], "A"] == pytest.approx(4.0, rel=1e-10)
        assert result.loc[dates[2], "B"] == pytest.approx(6.0, rel=1e-10)

    def test_rolling_window_slides(self, factor):
        """验证滚动窗口正确滑动 (T=3)。

        rkurt = [1, 2, 3, 4, 5]
        rolling(3):
          day2: mean(1,2,3) = 2
          day3: mean(2,3,4) = 3
          day4: mean(3,4,5) = 4
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        rkurt = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0], index=dates, columns=stocks)

        result = factor.compute(daily_rkurt=rkurt, T=3)

        assert result.iloc[2, 0] == pytest.approx(2.0, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(3.0, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(4.0, rel=1e-10)


class TestRealizedKurtosisEdgeCases:
    def test_nan_propagation(self, factor):
        """输入含 NaN 时, 对应窗口结果也应为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        rkurt = pd.DataFrame([3.0, np.nan, 5.0], index=dates, columns=stocks)

        result = factor.compute(daily_rkurt=rkurt, T=2)

        assert np.isnan(result.iloc[1, 0])
        assert np.isnan(result.iloc[2, 0])

    def test_all_zero_kurtosis(self, factor):
        """峰度全为 0 时, 因子值应为 0。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        rkurt = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_rkurt=rkurt, T=3)
        assert result.iloc[2, 0] == pytest.approx(0.0, abs=1e-15)

    def test_insufficient_data(self, factor):
        """数据不足 T 天时, 全部为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]

        rkurt = pd.DataFrame(3.0, index=dates, columns=stocks)

        result = factor.compute(daily_rkurt=rkurt, T=20)
        assert result.isna().all().all()

    def test_large_kurtosis_values(self, factor):
        """极端峰度值不应出错。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        rkurt = pd.DataFrame([100.0, 200.0, 300.0], index=dates, columns=stocks)

        result = factor.compute(daily_rkurt=rkurt, T=3)
        assert result.iloc[2, 0] == pytest.approx(200.0, rel=1e-10)


class TestRealizedKurtosisOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        stocks = ["A", "B", "C"]

        rkurt = pd.DataFrame(
            np.random.uniform(2, 6, (50, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_rkurt=rkurt, T=20)

        assert result.shape == rkurt.shape
        assert list(result.columns) == list(rkurt.columns)
        assert list(result.index) == list(rkurt.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        rkurt = pd.DataFrame(3.0, index=dates, columns=stocks)

        result = factor.compute(daily_rkurt=rkurt, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        """前 T-1 行应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        T = 5

        rkurt = pd.DataFrame(
            np.random.uniform(2, 6, (10, 2)), index=dates, columns=stocks
        )

        result = factor.compute(daily_rkurt=rkurt, T=T)

        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
