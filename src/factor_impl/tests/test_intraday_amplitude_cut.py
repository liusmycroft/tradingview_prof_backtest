import numpy as np
import pandas as pd
import pytest

from factors.intraday_amplitude_cut import IntradayAmplitudeCutFactor


@pytest.fixture
def factor():
    return IntradayAmplitudeCutFactor()


class TestIntradayAmplitudeCutMetadata:
    def test_name(self, factor):
        assert factor.name == "INTRADAY_AMPLITUDE_CUT"

    def test_category(self, factor):
        assert factor.category == "高频波动跳跃"

    def test_repr(self, factor):
        assert "INTRADAY_AMPLITUDE_CUT" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "INTRADAY_AMPLITUDE_CUT"
        assert meta["category"] == "高频波动跳跃"


class TestIntradayAmplitudeCutHandCalculated:
    def test_constant_input_all_stocks_same(self, factor):
        """When all stocks have the same constant value, z-scores are NaN
        (std=0 -> replaced by NaN)."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B", "C"]
        daily_amplitude_cut = pd.DataFrame(0.05, index=dates, columns=stocks)

        result = factor.compute(daily_amplitude_cut=daily_amplitude_cut, N=10)
        # row_std = 0 -> replaced by NaN -> z-scores are NaN
        valid = result.dropna(how="all")
        assert len(valid) == 0

    def test_different_stocks_zscore(self, factor):
        """With varying values per stock, z-scores should be computable."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B", "C"]
        # Use varying data so rolling std is non-zero
        daily_amplitude_cut = pd.DataFrame(
            {
                "A": np.random.uniform(0.01, 0.02, 10),
                "B": np.random.uniform(0.04, 0.06, 10),
                "C": np.random.uniform(0.08, 0.10, 10),
            },
            index=dates,
        )

        result = factor.compute(daily_amplitude_cut=daily_amplitude_cut, N=10)
        # Last row should have values (N=10, 10 rows available)
        row = result.iloc[-1]
        assert row.notna().all()
        # A has lowest mean -> lowest z-score, C has highest
        assert row["A"] < row["C"]

    def test_min_periods_produces_nan(self, factor):
        """With N=10 and only 5 rows, all output should be NaN."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A", "B"]
        daily_amplitude_cut = pd.DataFrame(
            np.random.rand(5, 2), index=dates, columns=stocks
        )

        result = factor.compute(daily_amplitude_cut=daily_amplitude_cut, N=10)
        assert result.isna().all().all()


class TestIntradayAmplitudeCutEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=15, freq="D")
        stocks = ["A", "B"]
        data = pd.DataFrame(
            np.random.rand(15, 2), index=dates, columns=stocks
        )
        data.iloc[3, 0] = np.nan

        result = factor.compute(daily_amplitude_cut=data, N=10)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (15, 2)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=15, freq="D")
        stocks = ["A", "B"]
        data = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_amplitude_cut=data, N=10)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=15, freq="D")
        stocks = ["A", "B"]
        data = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_amplitude_cut=data, N=10)
        assert isinstance(result, pd.DataFrame)


class TestIntradayAmplitudeCutOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        data = pd.DataFrame(
            np.random.rand(30, 3), index=dates, columns=stocks
        )

        result = factor.compute(daily_amplitude_cut=data, N=10)
        assert result.shape == data.shape
        assert list(result.columns) == list(data.columns)
        assert list(result.index) == list(data.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=15, freq="D")
        stocks = ["A", "B"]
        data = pd.DataFrame(
            np.random.rand(15, 2), index=dates, columns=stocks
        )

        result = factor.compute(daily_amplitude_cut=data, N=10)
        assert isinstance(result, pd.DataFrame)
