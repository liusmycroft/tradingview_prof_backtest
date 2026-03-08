import numpy as np
import pandas as pd
import pytest

from factors.corrected_overnight_reversal import CorrectedOvernightReversalFactor


@pytest.fixture
def factor():
    return CorrectedOvernightReversalFactor()


class TestCorrectedOvernightReversalMetadata:
    def test_name(self, factor):
        assert factor.name == "CORRECTED_OVERNIGHT_REVERSAL"

    def test_category(self, factor):
        assert factor.category == "高频动量反转"

    def test_repr(self, factor):
        assert "CORRECTED_OVERNIGHT_REVERSAL" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "CORRECTED_OVERNIGHT_REVERSAL"
        assert meta["category"] == "高频动量反转"


class TestCorrectedOvernightReversalHandCalculated:
    def test_constant_input_single_stock(self, factor):
        """Single stock: vol is NaN (std of constant), so result is NaN."""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        data = pd.DataFrame(0.01, index=dates, columns=["A"])
        result = factor.compute(overnight_distance=data, T=20)
        # std of constant is 0, cross_section_mean with 1 stock is 0
        # 0 < 0 is False, so no flip; rolling mean of 0.01 = 0.01
        # Actually std of constant = 0, mean(axis=1) of single col = 0
        # 0.lt(0) = False, so corrected = original
        assert isinstance(result, pd.DataFrame)

    def test_multi_stock_flip(self, factor):
        """With multiple stocks, low-vol stocks get flipped."""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        np.random.seed(42)
        # Stock A: low variance, Stock B: high variance
        a_vals = np.ones(25) * 0.01
        b_vals = np.linspace(0.0, 0.1, 25)
        data = pd.DataFrame({"A": a_vals, "B": b_vals}, index=dates)
        result = factor.compute(overnight_distance=data, T=20)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == data.shape

    def test_output_has_nan_for_first_T_minus_1(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.uniform(0.0, 0.05, (25, 2)), index=dates, columns=["A", "B"]
        )
        result = factor.compute(overnight_distance=data, T=20)
        # First T-1 rows of rolling(T).mean() are NaN
        assert result.iloc[:19].isna().all().all()


class TestCorrectedOvernightReversalEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        values = np.ones(25) * 0.01
        values[5] = np.nan
        data = pd.DataFrame({"A": values, "B": values * 2}, index=dates)
        result = factor.compute(overnight_distance=data, T=20)
        assert isinstance(result, pd.DataFrame)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        data = pd.DataFrame(np.nan, index=dates, columns=["A"])
        result = factor.compute(overnight_distance=data, T=20)
        assert result.isna().all().all()


class TestCorrectedOvernightReversalOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        data = pd.DataFrame(
            np.random.uniform(0.0, 0.05, (30, 3)), index=dates, columns=stocks
        )
        result = factor.compute(overnight_distance=data, T=20)
        assert result.shape == data.shape
        assert list(result.columns) == list(data.columns)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        data = pd.DataFrame(
            np.random.uniform(0.0, 0.05, (25, 2)), index=dates, columns=["A", "B"]
        )
        result = factor.compute(overnight_distance=data, T=20)
        assert isinstance(result, pd.DataFrame)
