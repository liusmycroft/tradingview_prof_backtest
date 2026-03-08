import numpy as np
import pandas as pd
import pytest

from factors.salience_return import SalienceReturnFactor


@pytest.fixture
def factor():
    return SalienceReturnFactor()


class TestSalienceReturnMetadata:
    def test_name(self, factor):
        assert factor.name == "SALIENCE_RETURN"

    def test_category(self, factor):
        assert factor.category == "高频动量反转"

    def test_repr(self, factor):
        assert "SALIENCE_RETURN" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "SALIENCE_RETURN"
        assert meta["category"] == "高频动量反转"


class TestSalienceReturnHandCalculated:
    def test_basic_output(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        daily_return = pd.DataFrame(
            {"A": [0.01, -0.02, 0.03, -0.01, 0.02]}, index=dates
        )
        market_med = pd.DataFrame(
            {"med": [0.005, -0.01, 0.015, -0.005, 0.01]}, index=dates
        )
        result = factor.compute(
            daily_return=daily_return, market_median_return=market_med, T=3
        )
        assert isinstance(result, pd.DataFrame)
        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert not np.isnan(result.iloc[2, 0])

    def test_two_stocks(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        daily_return = pd.DataFrame(
            {"A": [0.01, -0.02, 0.03, -0.01, 0.02],
             "B": [-0.01, 0.02, -0.03, 0.01, -0.02]},
            index=dates,
        )
        market_med = pd.DataFrame({"med": [0.0] * 5}, index=dates)
        result = factor.compute(
            daily_return=daily_return, market_median_return=market_med, T=3
        )
        assert result.shape == daily_return.shape
        assert result.iloc[2:].notna().all().all()


class TestSalienceReturnEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        daily_return = pd.DataFrame(
            {"A": [0.01, np.nan, 0.03, -0.01, 0.02]}, index=dates
        )
        market_med = pd.DataFrame({"med": [0.0] * 5}, index=dates)
        result = factor.compute(
            daily_return=daily_return, market_median_return=market_med, T=3
        )
        assert isinstance(result, pd.DataFrame)

    def test_all_zero_returns(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        daily_return = pd.DataFrame({"A": [0.0] * 5}, index=dates)
        market_med = pd.DataFrame({"med": [0.0] * 5}, index=dates)
        result = factor.compute(
            daily_return=daily_return, market_median_return=market_med, T=3
        )
        assert isinstance(result, pd.DataFrame)
        # cov of zeros with anything is 0
        assert result.iloc[2, 0] == pytest.approx(0.0, abs=1e-10)


class TestSalienceReturnOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily_return = pd.DataFrame(
            np.random.uniform(-0.05, 0.05, (30, 3)), index=dates, columns=stocks
        )
        market_med = pd.DataFrame(
            {"med": np.random.uniform(-0.02, 0.02, 30)}, index=dates
        )
        result = factor.compute(
            daily_return=daily_return, market_median_return=market_med, T=20
        )
        assert result.shape == daily_return.shape
        assert list(result.columns) == list(daily_return.columns)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        daily_return = pd.DataFrame(
            {"A": [0.01, -0.02, 0.03, -0.01, 0.02]}, index=dates
        )
        market_med = pd.DataFrame({"med": [0.0] * 5}, index=dates)
        result = factor.compute(
            daily_return=daily_return, market_median_return=market_med, T=3
        )
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        daily_return = pd.DataFrame(
            np.random.uniform(-0.05, 0.05, (25, 2)), index=dates, columns=["A", "B"]
        )
        market_med = pd.DataFrame(
            {"med": np.random.uniform(-0.02, 0.02, 25)}, index=dates
        )
        result = factor.compute(
            daily_return=daily_return, market_median_return=market_med, T=20
        )
        assert result.iloc[:19].isna().all().all()
        assert result.iloc[19:].notna().all().all()
