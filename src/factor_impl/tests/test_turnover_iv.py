import numpy as np
import pandas as pd
import pytest

from factors.turnover_iv import TurnoverIVFactor


@pytest.fixture
def factor():
    return TurnoverIVFactor()


class TestTurnoverIVMetadata:
    def test_name(self, factor):
        assert factor.name == "TURNOVER_IV"

    def test_category(self, factor):
        assert factor.category == "高频波动跳跃"

    def test_repr(self, factor):
        assert "TURNOVER_IV" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "TURNOVER_IV"
        assert meta["category"] == "高频波动跳跃"


class TestTurnoverIVHandCalculated:
    def test_constant_turnover(self, factor):
        """Constant turnover: residual after PCA should be ~0, std ~0."""
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        stocks = ["A", "B", "C", "D", "E"]
        turnover = pd.DataFrame(0.05, index=dates, columns=stocks)

        result = factor.compute(turnover=turnover, T=20, n_components=3)
        # Constant data: centered = 0, residual = 0, std = 0 or NaN
        valid = result.dropna()
        if len(valid) > 0:
            for col in stocks:
                vals = valid[col].values
                assert all(v == pytest.approx(0.0, abs=1e-10) or np.isnan(v) for v in vals)

    def test_output_non_negative(self, factor):
        """Standard deviation should be non-negative."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        stocks = ["A", "B", "C", "D", "E"]
        turnover = pd.DataFrame(
            np.random.uniform(0.01, 0.1, (60, 5)), index=dates, columns=stocks
        )

        result = factor.compute(turnover=turnover, T=20, n_components=3)
        valid = result.values[~np.isnan(result.values)]
        assert (valid >= -1e-10).all()

    def test_insufficient_columns(self, factor):
        """With fewer columns than n_components+1, result should be all NaN."""
        dates = pd.date_range("2024-01-01", periods=40, freq="D")
        stocks = ["A", "B"]
        turnover = pd.DataFrame(
            np.random.uniform(0.01, 0.1, (40, 2)), index=dates, columns=stocks
        )

        result = factor.compute(turnover=turnover, T=20, n_components=3)
        assert result.isna().all().all()


class TestTurnoverIVEdgeCases:
    def test_nan_in_input(self, factor):
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        stocks = ["A", "B", "C", "D", "E"]
        turnover = pd.DataFrame(
            np.random.uniform(0.01, 0.1, (60, 5)), index=dates, columns=stocks
        )
        turnover.iloc[10, 0] = np.nan

        result = factor.compute(turnover=turnover, T=20, n_components=3)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (60, 5)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=40, freq="D")
        stocks = ["A", "B", "C", "D", "E"]
        turnover = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(turnover=turnover, T=20, n_components=3)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        stocks = ["A", "B", "C", "D", "E"]
        turnover = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(turnover=turnover, T=20, n_components=3)
        assert isinstance(result, pd.DataFrame)


class TestTurnoverIVOutputShape:
    def test_output_shape_matches_input(self, factor):
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        stocks = ["A", "B", "C", "D", "E"]
        turnover = pd.DataFrame(
            np.random.uniform(0.01, 0.1, (60, 5)), index=dates, columns=stocks
        )

        result = factor.compute(turnover=turnover, T=20, n_components=3)
        assert result.shape == turnover.shape
        assert list(result.columns) == list(turnover.columns)
        assert list(result.index) == list(turnover.index)

    def test_output_is_dataframe(self, factor):
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        stocks = ["A", "B", "C", "D", "E"]
        turnover = pd.DataFrame(
            np.random.uniform(0.01, 0.1, (60, 5)), index=dates, columns=stocks
        )

        result = factor.compute(turnover=turnover, T=20, n_components=3)
        assert isinstance(result, pd.DataFrame)
