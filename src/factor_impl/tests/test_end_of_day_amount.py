import numpy as np
import pandas as pd
import pytest

from factors.end_of_day_amount import EndOfDayAmountFactor


@pytest.fixture
def factor():
    return EndOfDayAmountFactor()


class TestEndOfDayAmountMetadata:
    def test_name(self, factor):
        assert factor.name == "END_OF_DAY_AMOUNT"

    def test_category(self, factor):
        assert factor.category == "高频成交分布"

    def test_repr(self, factor):
        assert "END_OF_DAY_AMOUNT" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "END_OF_DAY_AMOUNT"
        assert meta["category"] == "高频成交分布"


class TestEndOfDayAmountHandCalculated:
    def test_constant_input(self, factor):
        """常数输入时，滚动均值应等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        daily_apl = pd.DataFrame(0.15, index=dates, columns=stocks)

        result = factor.compute(daily_apl=daily_apl, N=20)
        np.testing.assert_array_almost_equal(result["A"].values, 0.15)

    def test_rolling_mean_manual(self, factor):
        """手动验证滚动均值 N=3。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_apl = pd.DataFrame(
            [0.10, 0.20, 0.30, 0.40, 0.50], index=dates, columns=stocks
        )

        result = factor.compute(daily_apl=daily_apl, N=3)

        # min_periods=1, so:
        # day0: 0.10
        # day1: (0.10+0.20)/2 = 0.15
        # day2: (0.10+0.20+0.30)/3 = 0.20
        # day3: (0.20+0.30+0.40)/3 = 0.30
        # day4: (0.30+0.40+0.50)/3 = 0.40
        assert result.iloc[0, 0] == pytest.approx(0.10, rel=1e-6)
        assert result.iloc[1, 0] == pytest.approx(0.15, rel=1e-6)
        assert result.iloc[2, 0] == pytest.approx(0.20, rel=1e-6)
        assert result.iloc[3, 0] == pytest.approx(0.30, rel=1e-6)
        assert result.iloc[4, 0] == pytest.approx(0.40, rel=1e-6)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        daily_apl = pd.DataFrame(
            {"A": [0.10] * 10, "B": [0.20] * 10}, index=dates
        )

        result = factor.compute(daily_apl=daily_apl, N=5)
        np.testing.assert_array_almost_equal(result["A"].values, 0.10)
        np.testing.assert_array_almost_equal(result["B"].values, 0.20)


class TestEndOfDayAmountEdgeCases:
    def test_single_value(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A"]
        daily_apl = pd.DataFrame([0.12], index=dates, columns=stocks)

        result = factor.compute(daily_apl=daily_apl, N=20)
        assert result.iloc[0, 0] == pytest.approx(0.12, rel=1e-10)

    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        values = np.ones(10) * 0.15
        values[3] = np.nan
        daily_apl = pd.DataFrame(values, index=dates, columns=stocks)

        result = factor.compute(daily_apl=daily_apl, N=5)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily_apl = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_apl=daily_apl, N=5)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily_apl = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_apl=daily_apl, N=5)
        for val in result["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)


class TestEndOfDayAmountOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily_apl = pd.DataFrame(
            np.random.uniform(0.05, 0.25, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_apl=daily_apl, N=20)
        assert result.shape == daily_apl.shape
        assert list(result.columns) == list(daily_apl.columns)
        assert list(result.index) == list(daily_apl.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_apl = pd.DataFrame([0.1, 0.2, 0.3, 0.4, 0.5], index=dates, columns=stocks)

        result = factor.compute(daily_apl=daily_apl, N=3)
        assert isinstance(result, pd.DataFrame)

    def test_no_leading_nan_min_periods_1(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        daily_apl = pd.DataFrame(
            np.random.uniform(0.05, 0.25, (10, 2)), index=dates, columns=stocks
        )

        result = factor.compute(daily_apl=daily_apl, N=20)
        assert result.iloc[0].notna().all()
        assert result.notna().all().all()
