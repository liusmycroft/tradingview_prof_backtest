import numpy as np
import pandas as pd
import pytest

from factors.normal_big_ret import NormalBigRetFactor


@pytest.fixture
def factor():
    return NormalBigRetFactor()


class TestNormalBigRetMetadata:
    def test_name(self, factor):
        assert factor.name == "NORMAL_BIG_RET"

    def test_category(self, factor):
        assert factor.category == "高频动量反转"

    def test_repr(self, factor):
        assert "NORMAL_BIG_RET" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "NORMAL_BIG_RET"
        assert meta["category"] == "高频动量反转"


class TestNormalBigRetHandCalculated:
    def test_constant_input(self, factor):
        """常数输入时, rolling sum 应等于 T * 常数。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame(0.01, index=dates, columns=["A"])

        result = factor.compute(daily_normal_big_ret=data, T=3)
        # rolling(3, min_periods=1): day0=0.01, day1=0.02, day2=0.03, day3=0.03, day4=0.03
        assert result.iloc[0, 0] == pytest.approx(0.01, rel=1e-10)
        assert result.iloc[1, 0] == pytest.approx(0.02, rel=1e-10)
        assert result.iloc[2, 0] == pytest.approx(0.03, rel=1e-10)

    def test_varying_T3(self, factor):
        """T=3, data=[0.01, 0.02, 0.03] => sum=0.06。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        data = pd.DataFrame([0.01, 0.02, 0.03], index=dates, columns=["A"])

        result = factor.compute(daily_normal_big_ret=data, T=3)
        assert result.iloc[2, 0] == pytest.approx(0.06, rel=1e-10)

    def test_rolling_window_slides(self, factor):
        """验证滚动窗口正确滑动 (T=3)。

        data = [1, 2, 3, 4, 5]
        rolling(3) sum: day2=6, day3=9, day4=12
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0], index=dates, columns=["A"])

        result = factor.compute(daily_normal_big_ret=data, T=3)
        assert result.iloc[2, 0] == pytest.approx(6.0, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(9.0, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(12.0, rel=1e-10)


class TestNormalBigRetEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        values = np.ones(10) * 0.01
        values[3] = np.nan
        data = pd.DataFrame(values, index=dates, columns=["A"])

        result = factor.compute(daily_normal_big_ret=data, T=5)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(np.nan, index=dates, columns=["A"])

        result = factor.compute(daily_normal_big_ret=data, T=5)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(0.0, index=dates, columns=["A"])

        result = factor.compute(daily_normal_big_ret=data, T=5)
        for val in result["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)


class TestNormalBigRetOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        data = pd.DataFrame(
            np.random.randn(30, 3) * 0.01, index=dates, columns=stocks
        )

        result = factor.compute(daily_normal_big_ret=data, T=20)
        assert result.shape == data.shape
        assert list(result.columns) == list(data.columns)
        assert list(result.index) == list(data.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame([0.01, 0.02, 0.03, 0.04, 0.05], index=dates, columns=["A"])

        result = factor.compute(daily_normal_big_ret=data, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_no_leading_nan_min_periods_1(self, factor):
        """min_periods=1, 第一行就有值。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(
            np.random.randn(10, 2) * 0.01, index=dates, columns=["A", "B"]
        )

        result = factor.compute(daily_normal_big_ret=data, T=20)
        assert result.iloc[0].notna().all()
        assert result.notna().all().all()
