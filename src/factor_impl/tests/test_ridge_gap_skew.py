import numpy as np
import pandas as pd
import pytest

from factors.ridge_gap_skew import RidgeGapSkewFactor


@pytest.fixture
def factor():
    return RidgeGapSkewFactor()


class TestRidgeGapSkewMetadata:
    def test_name(self, factor):
        assert factor.name == "RIDGE_GAP_SKEW"

    def test_category(self, factor):
        assert factor.category == "高频成交分布"

    def test_repr(self, factor):
        assert "RIDGE_GAP_SKEW" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "RIDGE_GAP_SKEW"
        assert meta["category"] == "高频成交分布"
        assert "量岭" in meta["description"]


class TestRidgeGapSkewHandCalculated:
    def test_T3_single_stock(self, factor):
        """T=3, 单只股票, 手动计算验证。

        data = [1.0, 2.0, 3.0]
        rolling mean(T=3) at t=2: (1.0 + 2.0 + 3.0) / 3 = 2.0
        """
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame([1.0, 2.0, 3.0], index=dates, columns=stocks)

        result = factor.compute(daily_ridge_gap_skew=daily, T=3)

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(2.0, rel=1e-10)

    def test_constant_input(self, factor):
        """常数输入时, 滚动均值等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame([0.5] * 5, index=dates, columns=stocks)

        result = factor.compute(daily_ridge_gap_skew=daily, T=3)

        assert result.iloc[2, 0] == pytest.approx(0.5, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(0.5, rel=1e-10)

    def test_rolling_window_slides(self, factor):
        """验证滚动窗口正确滑动。

        data = [0.1, 0.2, 0.3, 0.4, 0.5], T=3
        t=2: mean(0.1, 0.2, 0.3) = 0.2
        t=3: mean(0.2, 0.3, 0.4) = 0.3
        t=4: mean(0.3, 0.4, 0.5) = 0.4
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame([0.1, 0.2, 0.3, 0.4, 0.5], index=dates, columns=stocks)

        result = factor.compute(daily_ridge_gap_skew=daily, T=3)

        assert result.iloc[2, 0] == pytest.approx(0.2, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(0.3, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(0.4, rel=1e-10)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A", "B"]
        daily = pd.DataFrame(
            {"A": [1.0, 2.0, 3.0], "B": [10.0, 20.0, 30.0]}, index=dates
        )

        result = factor.compute(daily_ridge_gap_skew=daily, T=3)

        assert result.iloc[2, 0] == pytest.approx(2.0, rel=1e-10)
        assert result.iloc[2, 1] == pytest.approx(20.0, rel=1e-10)


class TestRidgeGapSkewEdgeCases:
    def test_short_data(self, factor):
        """数据不足 T 时，结果应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame([0.1] * 5, index=dates, columns=stocks)

        result = factor.compute(daily_ridge_gap_skew=daily, T=20)
        assert result.isna().all().all()

    def test_nan_in_input(self, factor):
        """输入含 NaN 时不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame([0.1, np.nan, 0.3, 0.4, 0.5], index=dates, columns=stocks)

        result = factor.compute(daily_ridge_gap_skew=daily, T=3)
        assert isinstance(result, pd.DataFrame)
        # 窗口含 NaN 的行结果也为 NaN
        assert np.isnan(result.iloc[2, 0])

    def test_all_nan(self, factor):
        """全 NaN 输入时，结果应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_ridge_gap_skew=daily, T=5)
        assert result.isna().all().all()

    def test_negative_values(self, factor):
        """负偏度值应正常处理。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame([-1.0, -2.0, -3.0], index=dates, columns=stocks)

        result = factor.compute(daily_ridge_gap_skew=daily, T=3)
        assert result.iloc[2, 0] == pytest.approx(-2.0, rel=1e-10)


class TestRidgeGapSkewOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily = pd.DataFrame(
            np.random.randn(30, 3), index=dates, columns=stocks
        )

        result = factor.compute(daily_ridge_gap_skew=daily, T=20)
        assert result.shape == daily.shape
        assert list(result.columns) == list(daily.columns)
        assert list(result.index) == list(daily.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame([0.1] * 5, index=dates, columns=stocks)

        result = factor.compute(daily_ridge_gap_skew=daily, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        """前 T-1 行应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A", "B"]
        T = 20
        daily = pd.DataFrame(
            np.random.randn(25, 2), index=dates, columns=stocks
        )

        result = factor.compute(daily_ridge_gap_skew=daily, T=T)
        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
