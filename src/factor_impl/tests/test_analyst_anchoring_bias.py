import numpy as np
import pandas as pd
import pytest

from factors.analyst_anchoring_bias import AnalystAnchoringBiasFactor


@pytest.fixture
def factor():
    return AnalystAnchoringBiasFactor()


class TestCAFEPMetadata:
    def test_name(self, factor):
        assert factor.name == "CAF_EP"

    def test_category(self, factor):
        assert factor.category == "行为金融-锚定效应"

    def test_repr(self, factor):
        assert "CAF_EP" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "CAF_EP"
        assert meta["category"] == "行为金融-锚定效应"


class TestCAFEPHandCalculated:
    def test_same_pe_in_industry(self, factor):
        """同行业所有股票 PE 相同时, FEP 相同, IFEP=FEP, CAF_EP=0."""
        dates = pd.date_range("2024-01-01", periods=2, freq="ME")
        stocks = ["A", "B", "C"]
        forecast_pe = pd.DataFrame(10.0, index=dates, columns=stocks)
        industry = pd.Series(["ind1", "ind1", "ind1"], index=stocks)

        result = factor.compute(forecast_pe=forecast_pe, industry=industry)

        for i in range(2):
            for j in range(3):
                assert result.iloc[i, j] == pytest.approx(0.0, abs=1e-10)

    def test_manual_two_industries(self, factor):
        """手算两个行业的 CAF_EP.

        Industry 1: A(PE=10), B(PE=20)
          FEP: A=0.1, B=0.05
          IFEP = median(0.1, 0.05) = 0.075
          CAF_EP_A = (0.1 - 0.075) / 0.075 = 1/3
          CAF_EP_B = (0.05 - 0.075) / 0.075 = -1/3

        Industry 2: C(PE=5)
          FEP: C=0.2
          IFEP = 0.2 (single stock)
          CAF_EP_C = 0.0
        """
        dates = pd.date_range("2024-01-01", periods=1, freq="ME")
        stocks = ["A", "B", "C"]
        forecast_pe = pd.DataFrame([[10.0, 20.0, 5.0]], index=dates, columns=stocks)
        industry = pd.Series(["ind1", "ind1", "ind2"], index=stocks)

        result = factor.compute(forecast_pe=forecast_pe, industry=industry)

        assert result.iloc[0, 0] == pytest.approx(1.0 / 3.0, rel=1e-6)
        assert result.iloc[0, 1] == pytest.approx(-1.0 / 3.0, rel=1e-6)
        assert result.iloc[0, 2] == pytest.approx(0.0, abs=1e-10)

    def test_higher_pe_gives_negative_caf(self, factor):
        """PE 越高 => FEP 越低 => CAF_EP 越低."""
        dates = pd.date_range("2024-01-01", periods=1, freq="ME")
        stocks = ["A", "B"]
        forecast_pe = pd.DataFrame([[5.0, 20.0]], index=dates, columns=stocks)
        industry = pd.Series(["ind1", "ind1"], index=stocks)

        result = factor.compute(forecast_pe=forecast_pe, industry=industry)

        assert result.iloc[0, 0] > 0  # A: high FEP, above median
        assert result.iloc[0, 1] < 0  # B: low FEP, below median


class TestCAFEPEdgeCases:
    def test_nan_in_pe(self, factor):
        dates = pd.date_range("2024-01-01", periods=2, freq="ME")
        stocks = ["A", "B", "C"]
        vals = [[10.0, np.nan, 5.0], [10.0, 20.0, 5.0]]
        forecast_pe = pd.DataFrame(vals, index=dates, columns=stocks)
        industry = pd.Series(["ind1", "ind1", "ind1"], index=stocks)

        result = factor.compute(forecast_pe=forecast_pe, industry=industry)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 3)

    def test_single_stock_industry(self, factor):
        """单只股票的行业, IFEP = FEP, CAF_EP = 0."""
        dates = pd.date_range("2024-01-01", periods=1, freq="ME")
        stocks = ["A"]
        forecast_pe = pd.DataFrame([[10.0]], index=dates, columns=stocks)
        industry = pd.Series(["ind1"], index=stocks)

        result = factor.compute(forecast_pe=forecast_pe, industry=industry)
        assert result.iloc[0, 0] == pytest.approx(0.0, abs=1e-10)


class TestCAFEPOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=3, freq="ME")
        stocks = ["A", "B", "C", "D"]
        forecast_pe = pd.DataFrame(
            np.random.uniform(5, 30, (3, 4)), index=dates, columns=stocks
        )
        industry = pd.Series(["ind1", "ind1", "ind2", "ind2"], index=stocks)

        result = factor.compute(forecast_pe=forecast_pe, industry=industry)
        assert result.shape == forecast_pe.shape
        assert list(result.columns) == list(forecast_pe.columns)
        assert list(result.index) == list(forecast_pe.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="ME")
        stocks = ["A", "B"]
        forecast_pe = pd.DataFrame([[10.0, 20.0]], index=dates, columns=stocks)
        industry = pd.Series(["ind1", "ind1"], index=stocks)

        result = factor.compute(forecast_pe=forecast_pe, industry=industry)
        assert isinstance(result, pd.DataFrame)
