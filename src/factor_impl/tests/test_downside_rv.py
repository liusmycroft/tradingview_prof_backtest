import numpy as np
import pandas as pd
import pytest

from factors.downside_rv import DownsideRVFactor


@pytest.fixture
def factor():
    return DownsideRVFactor()


class TestDownsideRVMetadata:
    def test_name(self, factor):
        assert factor.name == "DOWNSIDE_RV"

    def test_category(self, factor):
        assert factor.category == "高频波动跳跃"

    def test_repr(self, factor):
        assert "DOWNSIDE_RV" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "DOWNSIDE_RV"
        assert meta["category"] == "高频波动跳跃"


class TestDownsideRVHandCalculated:
    """用手算数据验证 rolling mean 计算的正确性。"""

    def test_constant_input(self, factor):
        """常数输入时, rolling mean 应等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily_downside_rv = pd.DataFrame(0.005, index=dates, columns=stocks)

        result = factor.compute(daily_downside_rv=daily_downside_rv, T=20)

        valid = result.dropna()
        np.testing.assert_array_almost_equal(valid["A"].values, 0.005)

    def test_rolling_mean_manual_T3(self, factor):
        """T=3, 手动验证 rolling mean。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_downside_rv = pd.DataFrame(
            [1.0, 2.0, 3.0, 4.0, 5.0], index=dates, columns=stocks
        )

        result = factor.compute(daily_downside_rv=daily_downside_rv, T=3)

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(2.0, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(3.0, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(4.0, rel=1e-10)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A", "B"]
        daily_downside_rv = pd.DataFrame(
            {"A": [0.001] * 25, "B": [0.01] * 25}, index=dates
        )

        result = factor.compute(daily_downside_rv=daily_downside_rv, T=20)

        valid = result.dropna()
        np.testing.assert_array_almost_equal(valid["A"].values, 0.001)
        np.testing.assert_array_almost_equal(valid["B"].values, 0.01)

    def test_rolling_window_moves(self, factor):
        """验证滚动窗口正确移动。"""
        dates = pd.date_range("2024-01-01", periods=6, freq="D")
        stocks = ["A"]
        daily_downside_rv = pd.DataFrame(
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0], index=dates, columns=stocks
        )

        result = factor.compute(daily_downside_rv=daily_downside_rv, T=4)

        assert np.isnan(result.iloc[2, 0])
        assert result.iloc[3, 0] == pytest.approx(25.0, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(35.0, rel=1e-10)
        assert result.iloc[5, 0] == pytest.approx(45.0, rel=1e-10)


class TestDownsideRVEdgeCases:
    def test_nan_in_input(self, factor):
        """输入含 NaN 时不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        values = np.ones(25) * 0.005
        values[10] = np.nan
        daily_downside_rv = pd.DataFrame(values, index=dates, columns=stocks)

        result = factor.compute(daily_downside_rv=daily_downside_rv, T=20)
        assert result.shape == (25, 1)

    def test_all_nan(self, factor):
        """全 NaN 输入时, 结果应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily_downside_rv = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_downside_rv=daily_downside_rv, T=20)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        """全零输入时, rolling mean 应全为 0。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily_downside_rv = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_downside_rv=daily_downside_rv, T=20)
        valid = result.dropna()
        for val in valid["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)


class TestDownsideRVOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily_downside_rv = pd.DataFrame(
            np.random.uniform(0, 0.01, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_downside_rv=daily_downside_rv, T=20)

        assert result.shape == daily_downside_rv.shape
        assert list(result.columns) == list(daily_downside_rv.columns)
        assert list(result.index) == list(daily_downside_rv.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily_downside_rv = pd.DataFrame([0.001] * 25, index=dates, columns=stocks)

        result = factor.compute(daily_downside_rv=daily_downside_rv, T=20)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        """min_periods=T, 前 T-1 行应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        T = 20
        daily_downside_rv = pd.DataFrame(
            np.random.uniform(0, 0.01, (30, 2)), index=dates, columns=stocks
        )

        result = factor.compute(daily_downside_rv=daily_downside_rv, T=T)

        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
