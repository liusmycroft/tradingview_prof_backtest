import numpy as np
import pandas as pd
import pytest

from factors.hcvol import HCVOLFactor


@pytest.fixture
def factor():
    return HCVOLFactor()


class TestHCVOLMetadata:
    def test_name(self, factor):
        assert factor.name == "HCVOL"

    def test_category(self, factor):
        assert factor.category == "高频成交分布"

    def test_repr(self, factor):
        assert "HCVOL" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "HCVOL"
        assert meta["category"] == "高频成交分布"


class TestHCVOLHandCalculated:
    def test_constant_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        daily_hcvol = pd.DataFrame(0.3, index=dates, columns=stocks)

        result = factor.compute(daily_hcvol=daily_hcvol, T=20)
        np.testing.assert_allclose(result["A"].values, 0.3, atol=1e-10)

    def test_simple_mean_T3(self, factor):
        """T=3 rolling mean with min_periods=1."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_hcvol = pd.DataFrame(
            [0.1, 0.2, 0.3, 0.4, 0.5], index=dates, columns=stocks
        )

        result = factor.compute(daily_hcvol=daily_hcvol, T=3)
        assert result.iloc[0, 0] == pytest.approx(0.1, rel=1e-10)
        assert result.iloc[1, 0] == pytest.approx(0.15, rel=1e-10)
        assert result.iloc[2, 0] == pytest.approx(0.2, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(0.3, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(0.4, rel=1e-10)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        daily_hcvol = pd.DataFrame(
            {"A": [0.2] * 10, "B": [0.6] * 10}, index=dates
        )

        result = factor.compute(daily_hcvol=daily_hcvol, T=5)
        np.testing.assert_allclose(result["A"].values, 0.2, atol=1e-10)
        np.testing.assert_allclose(result["B"].values, 0.6, atol=1e-10)


class TestHCVOLEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        values = np.ones(10) * 0.3
        values[4] = np.nan
        daily_hcvol = pd.DataFrame(values, index=dates, columns=stocks)

        result = factor.compute(daily_hcvol=daily_hcvol, T=5)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily_hcvol = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_hcvol=daily_hcvol, T=5)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily_hcvol = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_hcvol=daily_hcvol, T=5)
        np.testing.assert_allclose(result["A"].values, 0.0, atol=1e-15)


class TestHCVOLOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily_hcvol = pd.DataFrame(
            np.random.uniform(0.0, 0.5, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_hcvol=daily_hcvol, T=20)
        assert result.shape == daily_hcvol.shape
        assert list(result.columns) == list(daily_hcvol.columns)
        assert list(result.index) == list(daily_hcvol.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_hcvol = pd.DataFrame(
            [0.1, 0.2, 0.3, 0.2, 0.1], index=dates, columns=stocks
        )

        result = factor.compute(daily_hcvol=daily_hcvol, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_no_leading_nan_min_periods_1(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        daily_hcvol = pd.DataFrame(
            np.random.uniform(0.0, 0.5, (10, 2)), index=dates, columns=stocks
        )

        result = factor.compute(daily_hcvol=daily_hcvol, T=20)
        assert result.iloc[0].notna().all()
