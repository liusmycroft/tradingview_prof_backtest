import numpy as np
import pandas as pd
import pytest

from factors.avg_positive_jump_return import AvgPositiveJumpReturnFactor


@pytest.fixture
def factor():
    return AvgPositiveJumpReturnFactor()


class TestAvgPositiveJumpReturnMetadata:
    def test_name(self, factor):
        assert factor.name == "AVG_POSITIVE_JUMP_RETURN"

    def test_category(self, factor):
        assert factor.category == "高频波动跳跃"

    def test_repr(self, factor):
        assert "AVG_POSITIVE_JUMP_RETURN" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "AVG_POSITIVE_JUMP_RETURN"
        assert meta["category"] == "高频波动跳跃"


class TestAvgPositiveJumpReturnHandCalculated:
    def test_constant_input(self, factor):
        """常数输入时, 滚动均值等于该常数."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(0.005, index=dates, columns=stocks)

        result = factor.compute(daily_avg_pos_jump_ret=daily, T=5)

        # min_periods=1, all rows have values
        np.testing.assert_array_almost_equal(result["A"].values, 0.005)

    def test_rolling_mean_T3(self, factor):
        """T=3 手算验证滚动均值 (min_periods=1)."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        vals = [0.01, 0.02, 0.03, 0.04, 0.05]
        daily = pd.DataFrame(vals, index=dates, columns=stocks)

        result = factor.compute(daily_avg_pos_jump_ret=daily, T=3)

        # min_periods=1:
        #   row 0: mean(0.01) = 0.01
        #   row 1: mean(0.01, 0.02) = 0.015
        #   row 2: mean(0.01, 0.02, 0.03) = 0.02
        #   row 3: mean(0.02, 0.03, 0.04) = 0.03
        #   row 4: mean(0.03, 0.04, 0.05) = 0.04
        assert result.iloc[0, 0] == pytest.approx(0.01, rel=1e-10)
        assert result.iloc[1, 0] == pytest.approx(0.015, rel=1e-10)
        assert result.iloc[2, 0] == pytest.approx(0.02, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(0.03, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(0.04, rel=1e-10)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        daily = pd.DataFrame(
            {"A": [0.01, 0.02, 0.03, 0.04, 0.05],
             "B": [0.10, 0.10, 0.10, 0.10, 0.10]},
            index=dates,
        )

        result = factor.compute(daily_avg_pos_jump_ret=daily, T=3)

        assert result.iloc[2, 0] == pytest.approx(0.02, rel=1e-10)
        assert result.iloc[2, 1] == pytest.approx(0.10, rel=1e-10)


class TestAvgPositiveJumpReturnEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        values = np.ones(10) * 0.005
        values[3] = np.nan
        daily = pd.DataFrame(values, index=dates, columns=stocks)

        result = factor.compute(daily_avg_pos_jump_ret=daily, T=5)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_avg_pos_jump_ret=daily, T=5)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_avg_pos_jump_ret=daily, T=5)
        np.testing.assert_array_almost_equal(result["A"].values, 0.0)

    def test_no_leading_nan_min_periods_1(self, factor):
        """min_periods=1, 第一行就有值."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(0.005, index=dates, columns=stocks)

        result = factor.compute(daily_avg_pos_jump_ret=daily, T=20)
        assert result.iloc[0].notna().all()
        assert result.notna().all().all()


class TestAvgPositiveJumpReturnOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily = pd.DataFrame(
            np.random.uniform(0, 0.01, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_avg_pos_jump_ret=daily, T=20)
        assert result.shape == daily.shape
        assert list(result.columns) == list(daily.columns)
        assert list(result.index) == list(daily.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(0.005, index=dates, columns=stocks)

        result = factor.compute(daily_avg_pos_jump_ret=daily, T=3)
        assert isinstance(result, pd.DataFrame)
