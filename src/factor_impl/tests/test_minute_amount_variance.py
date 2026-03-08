import numpy as np
import pandas as pd
import pytest

from factors.minute_amount_variance import MinuteAmountVarianceFactor


@pytest.fixture
def factor():
    return MinuteAmountVarianceFactor()


class TestMinuteAmountVarianceMetadata:
    def test_name(self, factor):
        assert factor.name == "VMA"

    def test_category(self, factor):
        assert factor.category == "高频成交分布"

    def test_repr(self, factor):
        assert "VMA" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "VMA"
        assert meta["category"] == "高频成交分布"


class TestMinuteAmountVarianceHandCalculated:
    def test_constant_input(self, factor):
        """常数输入时, 滚动均值应等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_vma = pd.DataFrame(1e10, index=dates, columns=stocks)

        result = factor.compute(daily_vma=daily_vma, T=3)

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(1e10, rel=1e-10)

    def test_varying_T3(self, factor):
        """T=3, 变化的方差值。

        vma = [100, 200, 300]
        rolling(3): mean = 200
        """
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]
        daily_vma = pd.DataFrame([100.0, 200.0, 300.0], index=dates, columns=stocks)

        result = factor.compute(daily_vma=daily_vma, T=3)
        assert result.iloc[2, 0] == pytest.approx(200.0, rel=1e-10)

    def test_rolling_window_slides(self, factor):
        """验证滚动窗口正确滑动 (T=3)。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_vma = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0], index=dates, columns=stocks)

        result = factor.compute(daily_vma=daily_vma, T=3)
        assert result.iloc[2, 0] == pytest.approx(2.0, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(3.0, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(4.0, rel=1e-10)

    def test_two_stocks(self, factor):
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A", "B"]
        daily_vma = pd.DataFrame(
            {"A": [10.0, 20.0, 30.0], "B": [100.0, 200.0, 300.0]}, index=dates
        )

        result = factor.compute(daily_vma=daily_vma, T=3)
        assert result.loc[dates[2], "A"] == pytest.approx(20.0, rel=1e-10)
        assert result.loc[dates[2], "B"] == pytest.approx(200.0, rel=1e-10)


class TestMinuteAmountVarianceEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]
        daily_vma = pd.DataFrame([100.0, np.nan, 300.0], index=dates, columns=stocks)

        result = factor.compute(daily_vma=daily_vma, T=3)
        assert np.isnan(result.iloc[2, 0])

    def test_all_zero(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_vma = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_vma=daily_vma, T=3)
        assert result.iloc[2, 0] == pytest.approx(0.0, abs=1e-15)

    def test_insufficient_data(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_vma = pd.DataFrame(100.0, index=dates, columns=stocks)

        result = factor.compute(daily_vma=daily_vma, T=20)
        assert result.isna().all().all()


class TestMinuteAmountVarianceOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        stocks = ["A", "B", "C"]
        daily_vma = pd.DataFrame(
            np.random.uniform(0, 1e12, (50, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_vma=daily_vma, T=20)
        assert result.shape == daily_vma.shape
        assert list(result.columns) == list(daily_vma.columns)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_vma = pd.DataFrame(100.0, index=dates, columns=stocks)

        result = factor.compute(daily_vma=daily_vma, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        T = 5
        daily_vma = pd.DataFrame(
            np.random.uniform(0, 1e12, (10, 2)), index=dates, columns=stocks
        )

        result = factor.compute(daily_vma=daily_vma, T=T)
        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
