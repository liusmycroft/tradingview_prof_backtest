import numpy as np
import pandas as pd
import pytest

from factors.co_min import COMinFactor


@pytest.fixture
def factor():
    return COMinFactor()


class TestMetadata:
    def test_name(self, factor):
        assert factor.name == "CO_MIN"

    def test_category(self, factor):
        assert factor.category == "高频动量反转"

    def test_repr(self, factor):
        assert "CO_MIN" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "CO_MIN"
        assert meta["category"] == "高频动量反转"


class TestHandCalculated:
    def test_constant_ratio(self, factor):
        """rm_min=0.002, std_min=0.001 => ratio=2.0, rolling mean=2.0"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        rm = pd.DataFrame(0.002, index=dates, columns=stocks)
        std = pd.DataFrame(0.001, index=dates, columns=stocks)

        result = factor.compute(daily_rm_min=rm, daily_std_min=std, T=20)
        assert result.iloc[-1, 0] == pytest.approx(2.0, rel=1e-6)

    def test_varying_ratio_T3(self, factor):
        """T=3, ratio=[1, 2, 3] => mean=2.0"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]
        rm = pd.DataFrame([1.0, 2.0, 3.0], index=dates, columns=stocks)
        std = pd.DataFrame([1.0, 1.0, 1.0], index=dates, columns=stocks)

        result = factor.compute(daily_rm_min=rm, daily_std_min=std, T=3)
        assert result.iloc[2, 0] == pytest.approx(2.0, rel=1e-10)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A", "B"]
        rm = pd.DataFrame({"A": [0.01] * 25, "B": [0.03] * 25}, index=dates)
        std = pd.DataFrame({"A": [0.01] * 25, "B": [0.01] * 25}, index=dates)

        result = factor.compute(daily_rm_min=rm, daily_std_min=std, T=20)
        assert result.iloc[-1, 0] == pytest.approx(1.0, rel=1e-6)
        assert result.iloc[-1, 1] == pytest.approx(3.0, rel=1e-6)

    def test_rolling_window_slides(self, factor):
        """验证滚动窗口正确滑动 (T=3)。

        ratio = [2, 4, 6, 8, 10]
        rolling(3): day2=4, day3=6, day4=8
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        rm = pd.DataFrame([2.0, 4.0, 6.0, 8.0, 10.0], index=dates, columns=stocks)
        std = pd.DataFrame(1.0, index=dates, columns=stocks)

        result = factor.compute(daily_rm_min=rm, daily_std_min=std, T=3)
        assert result.iloc[2, 0] == pytest.approx(4.0, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(6.0, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(8.0, rel=1e-10)


class TestEdgeCases:
    def test_nan_in_input(self, factor):
        """输入含 NaN 时, 不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        rm = pd.DataFrame([1.0, np.nan, 3.0, 4.0, 5.0], index=dates, columns=stocks)
        std = pd.DataFrame(1.0, index=dates, columns=stocks)

        result = factor.compute(daily_rm_min=rm, daily_std_min=std, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_zero_std_min(self, factor):
        """std_min 为零时, 结果应为 inf/NaN, 不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        rm = pd.DataFrame(1.0, index=dates, columns=stocks)
        std = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_rm_min=rm, daily_std_min=std, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_insufficient_data(self, factor):
        """数据不足 T 天时, 全部为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        rm = pd.DataFrame(1.0, index=dates, columns=stocks)
        std = pd.DataFrame(1.0, index=dates, columns=stocks)

        result = factor.compute(daily_rm_min=rm, daily_std_min=std, T=20)
        assert result.isna().all().all()


class TestOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        rm = pd.DataFrame(np.random.rand(30, 3) * 0.01, index=dates, columns=stocks)
        std = pd.DataFrame(np.random.rand(30, 3) * 0.01 + 0.001, index=dates, columns=stocks)

        result = factor.compute(daily_rm_min=rm, daily_std_min=std, T=20)
        assert result.shape == rm.shape
        assert list(result.columns) == list(rm.columns)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        rm = pd.DataFrame(1.0, index=dates, columns=stocks)
        std = pd.DataFrame(1.0, index=dates, columns=stocks)

        result = factor.compute(daily_rm_min=rm, daily_std_min=std, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        """前 T-1 行应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        T = 5
        rm = pd.DataFrame(np.random.rand(10, 2) * 0.01, index=dates, columns=stocks)
        std = pd.DataFrame(np.random.rand(10, 2) * 0.01 + 0.001, index=dates, columns=stocks)

        result = factor.compute(daily_rm_min=rm, daily_std_min=std, T=T)
        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
