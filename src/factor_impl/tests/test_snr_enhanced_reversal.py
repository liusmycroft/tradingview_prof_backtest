import numpy as np
import pandas as pd
import pytest

from factors.snr_enhanced_reversal import SNREnhancedReversalFactor


@pytest.fixture
def factor():
    return SNREnhancedReversalFactor()


class TestSNREnhancedReversalMetadata:
    def test_name(self, factor):
        assert factor.name == "SNR_ENHANCED_REVERSAL"

    def test_category(self, factor):
        assert factor.category == "高频动量反转"

    def test_repr(self, factor):
        assert "SNR_ENHANCED_REVERSAL" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "SNR_ENHANCED_REVERSAL"
        assert meta["category"] == "高频动量反转"


class TestSNREnhancedReversalHandCalculated:
    def test_basic_calculation(self, factor):
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        snr = pd.DataFrame({"A": [0.0, 0.5, 1.0], "B": [1.0, 0.5, 0.0]}, index=dates)
        reversal = pd.DataFrame({"A": [-0.02, -0.01, -0.03], "B": [-0.01, -0.02, -0.01]}, index=dates)
        result = factor.compute(snr=snr, reversal=reversal)
        # Day 0: snr_min=0, snr_max=1, range=1
        # A: weight=(0-0)/1=0, B: weight=(1-0)/1=1
        # A: 0*(-0.02)=0, B: 1*(-0.01)=-0.01
        assert result.iloc[0, 0] == pytest.approx(0.0, abs=1e-10)
        assert result.iloc[0, 1] == pytest.approx(-0.01)

    def test_uniform_snr_produces_nan(self, factor):
        """When all SNR values are equal, range=0, weight=NaN."""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        snr = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]}, index=dates)
        reversal = pd.DataFrame({"A": [-0.01, -0.02, -0.03], "B": [-0.01, -0.02, -0.03]}, index=dates)
        result = factor.compute(snr=snr, reversal=reversal)
        assert result.isna().all().all()

    def test_single_stock(self, factor):
        """Single stock: min=max, range=0, result is NaN."""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        snr = pd.DataFrame({"A": [0.3, 0.5, 0.7]}, index=dates)
        reversal = pd.DataFrame({"A": [-0.01, -0.02, -0.03]}, index=dates)
        result = factor.compute(snr=snr, reversal=reversal)
        # Single stock: min=max per row, so range=0 -> NaN
        assert result.isna().all().all()


class TestSNREnhancedReversalEdgeCases:
    def test_nan_in_snr(self, factor):
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        snr = pd.DataFrame({"A": [0.0, np.nan, 1.0], "B": [1.0, 0.5, 0.0]}, index=dates)
        reversal = pd.DataFrame({"A": [-0.01, -0.02, -0.03], "B": [-0.01, -0.02, -0.01]}, index=dates)
        result = factor.compute(snr=snr, reversal=reversal)
        assert isinstance(result, pd.DataFrame)

    def test_nan_in_reversal(self, factor):
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        snr = pd.DataFrame({"A": [0.0, 0.5, 1.0], "B": [1.0, 0.5, 0.0]}, index=dates)
        reversal = pd.DataFrame({"A": [-0.01, np.nan, -0.03], "B": [-0.01, -0.02, -0.01]}, index=dates)
        result = factor.compute(snr=snr, reversal=reversal)
        assert isinstance(result, pd.DataFrame)
        assert np.isnan(result.iloc[1, 0])


class TestSNREnhancedReversalOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        snr = pd.DataFrame(np.random.uniform(-1, 1, (30, 3)), index=dates, columns=stocks)
        reversal = pd.DataFrame(np.random.uniform(-0.05, 0.05, (30, 3)), index=dates, columns=stocks)
        result = factor.compute(snr=snr, reversal=reversal)
        assert result.shape == snr.shape
        assert list(result.columns) == list(snr.columns)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        snr = pd.DataFrame({"A": [0.1, 0.2, 0.3, 0.4, 0.5], "B": [0.5, 0.4, 0.3, 0.2, 0.1]}, index=dates)
        reversal = pd.DataFrame({"A": [-0.01] * 5, "B": [-0.02] * 5}, index=dates)
        result = factor.compute(snr=snr, reversal=reversal)
        assert isinstance(result, pd.DataFrame)
