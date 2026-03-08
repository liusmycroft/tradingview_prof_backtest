import numpy as np
import pandas as pd
import pytest

from factors.intraday_jump_drop import IntradayJumpDropFactor


@pytest.fixture
def factor():
    return IntradayJumpDropFactor()


class TestIntradayJumpDropMetadata:
    def test_name(self, factor):
        assert factor.name == "INTRADAY_JUMP_DROP"

    def test_category(self, factor):
        assert factor.category == "高频波动跳跃"

    def test_repr(self, factor):
        assert "INTRADAY_JUMP_DROP" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "INTRADAY_JUMP_DROP"
        assert meta["category"] == "高频波动跳跃"


class TestIntradayJumpDropHandCalculated:
    def test_constant_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        mean_data = pd.DataFrame(0.001, index=dates, columns=["A"])
        std_data = pd.DataFrame(0.002, index=dates, columns=["A"])
        result = factor.compute(
            daily_jump_drop_mean=mean_data, daily_jump_drop_std=std_data, T=20
        )
        expected = 0.5 * 0.001 + 0.5 * 0.002
        assert result.iloc[19, 0] == pytest.approx(expected)

    def test_rolling_mean_T3(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        mean_data = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0], index=dates, columns=["A"])
        std_data = pd.DataFrame([0.1, 0.2, 0.3, 0.4, 0.5], index=dates, columns=["A"])
        result = factor.compute(
            daily_jump_drop_mean=mean_data, daily_jump_drop_std=std_data, T=3
        )
        expected = 0.5 * 2.0 + 0.5 * 0.2  # mean of [1,2,3]=2, mean of [0.1,0.2,0.3]=0.2
        assert result.iloc[2, 0] == pytest.approx(expected)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        mean_data = pd.DataFrame({"A": [1.0] * 5, "B": [2.0] * 5}, index=dates)
        std_data = pd.DataFrame({"A": [0.5] * 5, "B": [1.0] * 5}, index=dates)
        result = factor.compute(
            daily_jump_drop_mean=mean_data, daily_jump_drop_std=std_data, T=3
        )
        assert result.iloc[2, 0] == pytest.approx(0.5 * 1.0 + 0.5 * 0.5)
        assert result.iloc[2, 1] == pytest.approx(0.5 * 2.0 + 0.5 * 1.0)


class TestIntradayJumpDropEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        mean_data = pd.DataFrame([1.0, np.nan, 3.0, 4.0, 5.0], index=dates, columns=["A"])
        std_data = pd.DataFrame([0.1, 0.2, 0.3, 0.4, 0.5], index=dates, columns=["A"])
        result = factor.compute(
            daily_jump_drop_mean=mean_data, daily_jump_drop_std=std_data, T=3
        )
        assert isinstance(result, pd.DataFrame)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        mean_data = pd.DataFrame(np.nan, index=dates, columns=["A"])
        std_data = pd.DataFrame(np.nan, index=dates, columns=["A"])
        result = factor.compute(
            daily_jump_drop_mean=mean_data, daily_jump_drop_std=std_data, T=5
        )
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        mean_data = pd.DataFrame(0.0, index=dates, columns=["A"])
        std_data = pd.DataFrame(0.0, index=dates, columns=["A"])
        result = factor.compute(
            daily_jump_drop_mean=mean_data, daily_jump_drop_std=std_data, T=3
        )
        assert result.iloc[2, 0] == pytest.approx(0.0, abs=1e-15)


class TestIntradayJumpDropOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        mean_data = pd.DataFrame(
            np.random.uniform(0.001, 0.01, (30, 3)), index=dates, columns=stocks
        )
        std_data = pd.DataFrame(
            np.random.uniform(0.001, 0.01, (30, 3)), index=dates, columns=stocks
        )
        result = factor.compute(
            daily_jump_drop_mean=mean_data, daily_jump_drop_std=std_data, T=20
        )
        assert result.shape == mean_data.shape

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        mean_data = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0], index=dates, columns=["A"])
        std_data = pd.DataFrame([0.1, 0.2, 0.3, 0.4, 0.5], index=dates, columns=["A"])
        result = factor.compute(
            daily_jump_drop_mean=mean_data, daily_jump_drop_std=std_data, T=3
        )
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        mean_data = pd.DataFrame(
            np.random.uniform(0.001, 0.01, (25, 2)), index=dates, columns=["A", "B"]
        )
        std_data = pd.DataFrame(
            np.random.uniform(0.001, 0.01, (25, 2)), index=dates, columns=["A", "B"]
        )
        result = factor.compute(
            daily_jump_drop_mean=mean_data, daily_jump_drop_std=std_data, T=20
        )
        assert result.iloc[:19].isna().all().all()
        assert result.iloc[19:].notna().all().all()
