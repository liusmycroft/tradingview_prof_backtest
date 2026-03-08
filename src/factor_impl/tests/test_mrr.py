import numpy as np
import pandas as pd
import pytest

from factors.mrr import MRRFactor


@pytest.fixture
def factor():
    return MRRFactor()


class TestMetadata:
    def test_name(self, factor):
        assert factor.name == "MRR"

    def test_category(self, factor):
        assert factor.category == "图谱网络-动量溢出"

    def test_repr(self, factor):
        assert "MRR" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "MRR"
        assert meta["category"] == "图谱网络-动量溢出"


class TestHandCalculated:
    def test_constant_input(self, factor):
        """常数输入时, rolling mean 应等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily_mrr = pd.DataFrame(0.01, index=dates, columns=stocks)

        result = factor.compute(daily_mrr=daily_mrr, T=20)
        assert result.iloc[-1, 0] == pytest.approx(0.01, rel=1e-6)

    def test_varying_T3(self, factor):
        """T=3, data=[1, 2, 3] => mean=2.0"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]
        daily_mrr = pd.DataFrame([1.0, 2.0, 3.0], index=dates, columns=stocks)

        result = factor.compute(daily_mrr=daily_mrr, T=3)
        assert result.iloc[2, 0] == pytest.approx(2.0, rel=1e-10)

    def test_rolling_window_slides(self, factor):
        """验证滚动窗口正确滑动 (T=3)。

        data = [1, 2, 3, 4, 5]
        rolling(3): day2=2, day3=3, day4=4
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_mrr = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0], index=dates, columns=stocks)

        result = factor.compute(daily_mrr=daily_mrr, T=3)
        assert result.iloc[2, 0] == pytest.approx(2.0, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(3.0, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(4.0, rel=1e-10)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A", "B"]
        daily_mrr = pd.DataFrame(
            {"A": [0.01] * 25, "B": [-0.02] * 25}, index=dates
        )

        result = factor.compute(daily_mrr=daily_mrr, T=20)
        assert result.iloc[-1, 0] == pytest.approx(0.01, rel=1e-6)
        assert result.iloc[-1, 1] == pytest.approx(-0.02, rel=1e-6)

    def test_symmetric_cancels(self, factor):
        """前半正后半负, 对称时均值为 0。"""
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        stocks = ["A"]
        daily_mrr = pd.DataFrame([1.0, 1.0, -1.0, -1.0], index=dates, columns=stocks)

        result = factor.compute(daily_mrr=daily_mrr, T=4)
        assert result.iloc[3, 0] == pytest.approx(0.0, abs=1e-10)


class TestEdgeCases:
    def test_nan_in_input(self, factor):
        """输入含 NaN 时, 不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_mrr = pd.DataFrame([1.0, np.nan, 3.0, 4.0, 5.0], index=dates, columns=stocks)

        result = factor.compute(daily_mrr=daily_mrr, T=3)
        assert isinstance(result, pd.DataFrame)
        assert np.isnan(result.iloc[2, 0])

    def test_all_nan(self, factor):
        """全 NaN 输入时, 结果应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily_mrr = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_mrr=daily_mrr, T=5)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        """全零输入时, rolling mean 应全为 0。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily_mrr = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_mrr=daily_mrr, T=20)
        assert result.iloc[-1, 0] == pytest.approx(0.0, abs=1e-15)

    def test_insufficient_data(self, factor):
        """数据不足 T 天时, 全部为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_mrr = pd.DataFrame(1.0, index=dates, columns=stocks)

        result = factor.compute(daily_mrr=daily_mrr, T=20)
        assert result.isna().all().all()


class TestOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily_mrr = pd.DataFrame(
            np.random.randn(30, 3) * 0.01, index=dates, columns=stocks
        )

        result = factor.compute(daily_mrr=daily_mrr, T=20)
        assert result.shape == daily_mrr.shape
        assert list(result.columns) == list(daily_mrr.columns)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_mrr = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0], index=dates, columns=stocks)

        result = factor.compute(daily_mrr=daily_mrr, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        """前 T-1 行应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        T = 5
        daily_mrr = pd.DataFrame(
            np.random.randn(10, 2) * 0.01, index=dates, columns=stocks
        )

        result = factor.compute(daily_mrr=daily_mrr, T=T)
        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
