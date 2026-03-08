import numpy as np
import pandas as pd
import pytest

from factors.lcps import LCPSFactor


@pytest.fixture
def factor():
    return LCPSFactor()


class TestMetadata:
    def test_name(self, factor):
        assert factor.name == "LCPS"

    def test_category(self, factor):
        assert factor.category == "行为金融-遗憾规避"

    def test_repr(self, factor):
        assert "LCPS" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "LCPS"
        assert meta["category"] == "行为金融-遗憾规避"


class TestHandCalculated:
    def test_constant_input(self, factor):
        """常数输入时, rolling mean 应等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily_lcps = pd.DataFrame(0.05, index=dates, columns=stocks)

        result = factor.compute(daily_lcps=daily_lcps, T=20)
        assert result.iloc[-1, 0] == pytest.approx(0.05, rel=1e-6)

    def test_varying_T3(self, factor):
        """T=3, data=[1, 2, 3] => mean=2.0"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]
        daily_lcps = pd.DataFrame([1.0, 2.0, 3.0], index=dates, columns=stocks)

        result = factor.compute(daily_lcps=daily_lcps, T=3)
        assert result.iloc[2, 0] == pytest.approx(2.0, rel=1e-10)

    def test_rolling_window_slides(self, factor):
        """验证滚动窗口正确滑动 (T=3)。

        data = [1, 2, 3, 4, 5]
        rolling(3): day2=2, day3=3, day4=4
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_lcps = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0], index=dates, columns=stocks)

        result = factor.compute(daily_lcps=daily_lcps, T=3)
        assert result.iloc[2, 0] == pytest.approx(2.0, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(3.0, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(4.0, rel=1e-10)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A", "B"]
        daily_lcps = pd.DataFrame(
            {"A": [0.01] * 25, "B": [0.05] * 25}, index=dates
        )

        result = factor.compute(daily_lcps=daily_lcps, T=20)
        assert result.iloc[-1, 0] == pytest.approx(0.01, rel=1e-6)
        assert result.iloc[-1, 1] == pytest.approx(0.05, rel=1e-6)


class TestEdgeCases:
    def test_nan_in_input(self, factor):
        """输入含 NaN 时, 不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_lcps = pd.DataFrame([1.0, np.nan, 3.0, 4.0, 5.0], index=dates, columns=stocks)

        result = factor.compute(daily_lcps=daily_lcps, T=3)
        assert isinstance(result, pd.DataFrame)
        # 窗口内有 NaN, rolling mean 结果为 NaN
        assert np.isnan(result.iloc[2, 0])

    def test_all_nan(self, factor):
        """全 NaN 输入时, 结果应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily_lcps = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_lcps=daily_lcps, T=5)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        """全零输入时, rolling mean 应全为 0。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily_lcps = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_lcps=daily_lcps, T=20)
        assert result.iloc[-1, 0] == pytest.approx(0.0, abs=1e-15)

    def test_insufficient_data(self, factor):
        """数据不足 T 天时, 全部为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_lcps = pd.DataFrame(1.0, index=dates, columns=stocks)

        result = factor.compute(daily_lcps=daily_lcps, T=20)
        assert result.isna().all().all()


class TestOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily_lcps = pd.DataFrame(
            np.random.randn(30, 3) * 0.01, index=dates, columns=stocks
        )

        result = factor.compute(daily_lcps=daily_lcps, T=20)
        assert result.shape == daily_lcps.shape
        assert list(result.columns) == list(daily_lcps.columns)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_lcps = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0], index=dates, columns=stocks)

        result = factor.compute(daily_lcps=daily_lcps, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        """前 T-1 行应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        T = 5
        daily_lcps = pd.DataFrame(
            np.random.randn(10, 2) * 0.01, index=dates, columns=stocks
        )

        result = factor.compute(daily_lcps=daily_lcps, T=T)
        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
