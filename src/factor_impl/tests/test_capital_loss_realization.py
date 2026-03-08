import numpy as np
import pandas as pd
import pytest

from factors.capital_loss_realization import CapitalLossRealizationFactor


@pytest.fixture
def factor():
    return CapitalLossRealizationFactor()


class TestCPLRMetadata:
    def test_name(self, factor):
        assert factor.name == "CPLR"

    def test_category(self, factor):
        assert factor.category == "行为金融-处置效应"

    def test_repr(self, factor):
        assert "CPLR" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "CPLR"
        assert meta["category"] == "行为金融-处置效应"


class TestCPLRHandCalculated:
    def test_equal_losses(self, factor):
        """realized_loss == paper_loss => daily_cplr = 0.5, rolling mean = 0.5."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        realized = pd.DataFrame(100.0, index=dates, columns=stocks)
        paper = pd.DataFrame(100.0, index=dates, columns=stocks)

        result = factor.compute(realized_loss=realized, paper_loss=paper, T=5)
        np.testing.assert_array_almost_equal(result["A"].values, 0.5)

    def test_manual_calculation(self, factor):
        """Hand-calculated CPLR with T=3, min_periods=1."""
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        stocks = ["A"]
        # daily_cplr: [100/400=0.25, 200/400=0.5, 300/400=0.75, 400/400=1.0]
        realized = pd.DataFrame([100.0, 200.0, 300.0, 400.0], index=dates, columns=stocks)
        paper = pd.DataFrame([300.0, 200.0, 100.0, 0.0], index=dates, columns=stocks)

        result = factor.compute(realized_loss=realized, paper_loss=paper, T=3)
        # rolling(3, min_periods=1):
        #   day0: mean([0.25]) = 0.25
        #   day1: mean([0.25, 0.5]) = 0.375
        #   day2: mean([0.25, 0.5, 0.75]) = 0.5
        #   day3: mean([0.5, 0.75, 1.0]) = 0.75
        expected = [0.25, 0.375, 0.5, 0.75]
        np.testing.assert_array_almost_equal(result["A"].values, expected)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        realized = pd.DataFrame({"A": [50.0] * 10, "B": [25.0] * 10}, index=dates)
        paper = pd.DataFrame({"A": [50.0] * 10, "B": [75.0] * 10}, index=dates)

        result = factor.compute(realized_loss=realized, paper_loss=paper, T=5)
        np.testing.assert_array_almost_equal(result["A"].values, 0.5)
        np.testing.assert_array_almost_equal(result["B"].values, 0.25)


class TestCPLREdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        realized_vals = np.ones(10) * 50.0
        realized_vals[3] = np.nan
        realized = pd.DataFrame(realized_vals, index=dates, columns=stocks)
        paper = pd.DataFrame(50.0, index=dates, columns=stocks)

        result = factor.compute(realized_loss=realized, paper_loss=paper, T=5)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        realized = pd.DataFrame(np.nan, index=dates, columns=stocks)
        paper = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(realized_loss=realized, paper_loss=paper, T=5)
        assert result.isna().all().all()

    def test_zero_both(self, factor):
        """Both zero => 0/0 = NaN."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        realized = pd.DataFrame(0.0, index=dates, columns=stocks)
        paper = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(realized_loss=realized, paper_loss=paper, T=3)
        assert result.isna().all().all()


class TestCPLROutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        realized = pd.DataFrame(
            np.random.uniform(0, 100, (30, 3)), index=dates, columns=stocks
        )
        paper = pd.DataFrame(
            np.random.uniform(0, 100, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(realized_loss=realized, paper_loss=paper, T=20)
        assert result.shape == realized.shape
        assert list(result.columns) == list(realized.columns)
        assert list(result.index) == list(realized.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        realized = pd.DataFrame(50.0, index=dates, columns=stocks)
        paper = pd.DataFrame(50.0, index=dates, columns=stocks)

        result = factor.compute(realized_loss=realized, paper_loss=paper, T=5)
        assert isinstance(result, pd.DataFrame)
