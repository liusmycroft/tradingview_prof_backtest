import numpy as np
import pandas as pd
import pytest

from factors.jump_absolute_return import JumpAbsoluteReturnFactor


@pytest.fixture
def factor():
    return JumpAbsoluteReturnFactor()


class TestJumpAbsoluteReturnMetadata:
    def test_name(self, factor):
        assert factor.name == "JAR"

    def test_category(self, factor):
        assert factor.category == "高频动量反转"

    def test_repr(self, factor):
        assert "JAR" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "JAR"
        assert meta["category"] == "高频动量反转"


class TestJumpAbsoluteReturnHandCalculated:
    def test_constant_input(self, factor):
        """Rolling sum of constant c over T days = c*T (min_periods=1)."""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(0.01, index=dates, columns=stocks)

        result = factor.compute(daily_jump_abs_return=data, T=20)
        assert result.iloc[-1, 0] == pytest.approx(0.01 * 20, rel=1e-10)

    def test_simple_sum_T3(self, factor):
        """T=3 rolling sum with min_periods=1."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(
            [1.0, 2.0, 3.0, 4.0, 5.0], index=dates, columns=stocks
        )

        result = factor.compute(daily_jump_abs_return=data, T=3)
        assert result.iloc[0, 0] == pytest.approx(1.0, rel=1e-10)
        assert result.iloc[1, 0] == pytest.approx(3.0, rel=1e-10)
        assert result.iloc[2, 0] == pytest.approx(6.0, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(9.0, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(12.0, rel=1e-10)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(
            {"A": [0.01] * 10, "B": [0.05] * 10}, index=dates
        )

        result = factor.compute(daily_jump_abs_return=data, T=5)
        np.testing.assert_allclose(result["A"].iloc[-1], 0.05, atol=1e-10)
        np.testing.assert_allclose(result["B"].iloc[-1], 0.25, atol=1e-10)


class TestJumpAbsoluteReturnEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        values = np.ones(10) * 0.01
        values[4] = np.nan
        data = pd.DataFrame(values, index=dates, columns=stocks)

        result = factor.compute(daily_jump_abs_return=data, T=5)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_jump_abs_return=data, T=5)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_jump_abs_return=data, T=5)
        np.testing.assert_allclose(result["A"].values, 0.0, atol=1e-15)


class TestJumpAbsoluteReturnOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        data = pd.DataFrame(
            np.random.uniform(0.0, 0.05, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_jump_abs_return=data, T=20)
        assert result.shape == data.shape
        assert list(result.columns) == list(data.columns)
        assert list(result.index) == list(data.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(
            [0.01, 0.02, 0.03, 0.02, 0.01], index=dates, columns=stocks
        )

        result = factor.compute(daily_jump_abs_return=data, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_no_leading_nan_min_periods_1(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        data = pd.DataFrame(
            np.random.uniform(0.0, 0.05, (10, 2)), index=dates, columns=stocks
        )

        result = factor.compute(daily_jump_abs_return=data, T=20)
        assert result.iloc[0].notna().all()
