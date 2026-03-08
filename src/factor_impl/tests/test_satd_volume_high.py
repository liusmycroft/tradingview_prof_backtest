import numpy as np
import pandas as pd
import pytest

from factors.satd_volume_high import SATDVolumeHighFactor


@pytest.fixture
def factor():
    return SATDVolumeHighFactor()


class TestMetadata:
    def test_name(self, factor):
        assert factor.name == "SATD_VolumeHigh"

    def test_category(self, factor):
        assert factor.category == "高频成交分布"

    def test_repr(self, factor):
        assert "SATD_VolumeHigh" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "SATD_VolumeHigh"
        assert meta["category"] == "高频成交分布"


class TestHandCalculated:
    def test_constant_ratio(self, factor):
        """atd_high=5000, atd_all=2500 => satd=2.0, rolling mean=2.0"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        atd_high = pd.DataFrame(5000.0, index=dates, columns=stocks)
        atd_all = pd.DataFrame(2500.0, index=dates, columns=stocks)

        result = factor.compute(daily_atd_high=atd_high, daily_atd_all=atd_all, T=20)
        assert result.iloc[-1, 0] == pytest.approx(2.0, rel=1e-6)

    def test_equal_atd(self, factor):
        """atd_high == atd_all => satd=1.0"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        atd = pd.DataFrame(3000.0, index=dates, columns=stocks)

        result = factor.compute(daily_atd_high=atd, daily_atd_all=atd, T=20)
        assert result.iloc[-1, 0] == pytest.approx(1.0, rel=1e-6)

    def test_varying_ratio_T3(self, factor):
        """T=3, satd=[1, 2, 3] => mean=2.0"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]
        atd_high = pd.DataFrame([100.0, 200.0, 300.0], index=dates, columns=stocks)
        atd_all = pd.DataFrame(100.0, index=dates, columns=stocks)

        result = factor.compute(daily_atd_high=atd_high, daily_atd_all=atd_all, T=3)
        assert result.iloc[2, 0] == pytest.approx(2.0, rel=1e-10)

    def test_rolling_window_slides(self, factor):
        """验证滚动窗口正确滑动 (T=3)。

        satd = [2, 4, 6, 8, 10]
        rolling(3): day2=4, day3=6, day4=8
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        atd_high = pd.DataFrame([2.0, 4.0, 6.0, 8.0, 10.0], index=dates, columns=stocks)
        atd_all = pd.DataFrame(1.0, index=dates, columns=stocks)

        result = factor.compute(daily_atd_high=atd_high, daily_atd_all=atd_all, T=3)
        assert result.iloc[2, 0] == pytest.approx(4.0, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(6.0, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(8.0, rel=1e-10)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        atd_high = pd.DataFrame({"A": [6000.0] * 25, "B": [4000.0] * 25}, index=dates)
        atd_all = pd.DataFrame({"A": [3000.0] * 25, "B": [2000.0] * 25}, index=dates)

        result = factor.compute(daily_atd_high=atd_high, daily_atd_all=atd_all, T=20)
        assert result.iloc[-1, 0] == pytest.approx(2.0, rel=1e-6)
        assert result.iloc[-1, 1] == pytest.approx(2.0, rel=1e-6)


class TestEdgeCases:
    def test_nan_in_input(self, factor):
        """输入含 NaN 时, 不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        atd_high = pd.DataFrame([1.0, np.nan, 3.0, 4.0, 5.0], index=dates, columns=stocks)
        atd_all = pd.DataFrame(1.0, index=dates, columns=stocks)

        result = factor.compute(daily_atd_high=atd_high, daily_atd_all=atd_all, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_zero_atd_all(self, factor):
        """atd_all 为零时, 结果应为 inf/NaN, 不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        atd_high = pd.DataFrame(1.0, index=dates, columns=stocks)
        atd_all = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_atd_high=atd_high, daily_atd_all=atd_all, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_insufficient_data(self, factor):
        """数据不足 T 天时, 全部为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        atd_high = pd.DataFrame(1.0, index=dates, columns=stocks)
        atd_all = pd.DataFrame(1.0, index=dates, columns=stocks)

        result = factor.compute(daily_atd_high=atd_high, daily_atd_all=atd_all, T=20)
        assert result.isna().all().all()

    def test_all_nan(self, factor):
        """全 NaN 输入时, 结果应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        atd_high = pd.DataFrame(np.nan, index=dates, columns=stocks)
        atd_all = pd.DataFrame(1.0, index=dates, columns=stocks)

        result = factor.compute(daily_atd_high=atd_high, daily_atd_all=atd_all, T=5)
        assert result.isna().all().all()


class TestOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        atd_high = pd.DataFrame(np.random.rand(30, 3) * 5000 + 3000, index=dates, columns=stocks)
        atd_all = pd.DataFrame(np.random.rand(30, 3) * 3000 + 2000, index=dates, columns=stocks)

        result = factor.compute(daily_atd_high=atd_high, daily_atd_all=atd_all, T=20)
        assert result.shape == atd_high.shape
        assert list(result.columns) == list(atd_high.columns)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        atd_high = pd.DataFrame(1.0, index=dates, columns=stocks)
        atd_all = pd.DataFrame(1.0, index=dates, columns=stocks)

        result = factor.compute(daily_atd_high=atd_high, daily_atd_all=atd_all, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        """前 T-1 行应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        T = 5
        atd_high = pd.DataFrame(np.random.rand(10, 2) * 5000 + 3000, index=dates, columns=stocks)
        atd_all = pd.DataFrame(np.random.rand(10, 2) * 3000 + 2000, index=dates, columns=stocks)

        result = factor.compute(daily_atd_high=atd_high, daily_atd_all=atd_all, T=T)
        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
