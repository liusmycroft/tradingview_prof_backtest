import numpy as np
import pandas as pd
import pytest

from factors.super_big_ret import SuperBigRetFactor


@pytest.fixture
def factor():
    return SuperBigRetFactor()


class TestSuperBigRetMetadata:
    def test_name(self, factor):
        assert factor.name == "SUPER_BIG_RET"

    def test_category(self, factor):
        assert factor.category == "高频动量反转"

    def test_repr(self, factor):
        assert "SUPER_BIG_RET" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "SUPER_BIG_RET"
        assert meta["category"] == "高频动量反转"


class TestSuperBigRetHandCalculated:
    def test_zero_return(self, factor):
        """Zero returns: log1p(0)=0, rolling sum=0."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(0.0, index=dates, columns=["A"])

        result = factor.compute(daily_super_big_return=data, T=5)
        for val in result["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)

    def test_rolling_sum_T3(self, factor):
        """T=3 rolling sum of log1p hand-calculated (min_periods=1)."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        vals = [0.01, 0.02, 0.03, 0.04, 0.05]
        data = pd.DataFrame(vals, index=dates, columns=["A"])

        result = factor.compute(daily_super_big_return=data, T=3)

        log_vals = [np.log1p(v) for v in vals]
        assert result.iloc[0, 0] == pytest.approx(log_vals[0], rel=1e-10)
        assert result.iloc[1, 0] == pytest.approx(log_vals[0] + log_vals[1], rel=1e-10)
        assert result.iloc[2, 0] == pytest.approx(sum(log_vals[0:3]), rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(sum(log_vals[1:4]), rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(sum(log_vals[2:5]), rel=1e-10)

    def test_negative_returns(self, factor):
        """Negative returns should produce negative log1p sums."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame([-0.01, -0.02, -0.03, -0.04, -0.05], index=dates, columns=["A"])

        result = factor.compute(daily_super_big_return=data, T=5)
        assert result.iloc[4, 0] < 0

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(
            {"A": [0.01] * 10, "B": [-0.01] * 10}, index=dates
        )

        result = factor.compute(daily_super_big_return=data, T=5)
        # A should be positive, B should be negative
        assert result.iloc[-1, 0] > 0
        assert result.iloc[-1, 1] < 0


class TestSuperBigRetEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        values = np.ones(10) * 0.01
        values[3] = np.nan
        data = pd.DataFrame(values, index=dates, columns=["A"])

        result = factor.compute(daily_super_big_return=data, T=5)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(np.nan, index=dates, columns=["A"])

        result = factor.compute(daily_super_big_return=data, T=5)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(0.0, index=dates, columns=["A"])

        result = factor.compute(daily_super_big_return=data, T=5)
        for val in result["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)


class TestSuperBigRetOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        data = pd.DataFrame(
            np.random.uniform(-0.05, 0.05, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_super_big_return=data, T=20)
        assert result.shape == data.shape
        assert list(result.columns) == list(data.columns)
        assert list(result.index) == list(data.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame(
            [0.01, 0.02, 0.03, 0.04, 0.05], index=dates, columns=["A"]
        )

        result = factor.compute(daily_super_big_return=data, T=3)
        assert isinstance(result, pd.DataFrame)
