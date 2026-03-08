import numpy as np
import pandas as pd
import pytest

from factors.high_low_spread_intensity import HighLowSpreadIntensityFactor


@pytest.fixture
def factor():
    return HighLowSpreadIntensityFactor()


class TestHighLowSpreadIntensityMetadata:
    def test_name(self, factor):
        assert factor.name == "HLI"

    def test_category(self, factor):
        assert factor.category == "高频流动性"

    def test_repr(self, factor):
        assert "HLI" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "HLI"
        assert meta["category"] == "高频流动性"


class TestHighLowSpreadIntensityHandCalculated:
    def test_constant_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        daily_hli = pd.DataFrame(0.05, index=dates, columns=stocks)

        result = factor.compute(daily_hli=daily_hli, T=20)
        np.testing.assert_allclose(result["A"].values, 0.05, atol=1e-10)

    def test_simple_mean_T3(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_hli = pd.DataFrame(
            [1.0, 2.0, 3.0, 4.0, 5.0], index=dates, columns=stocks
        )

        result = factor.compute(daily_hli=daily_hli, T=3)
        assert result.iloc[0, 0] == pytest.approx(1.0, rel=1e-10)
        assert result.iloc[1, 0] == pytest.approx(1.5, rel=1e-10)
        assert result.iloc[2, 0] == pytest.approx(2.0, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(3.0, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(4.0, rel=1e-10)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        daily_hli = pd.DataFrame(
            {"A": [0.01] * 10, "B": [0.10] * 10}, index=dates
        )

        result = factor.compute(daily_hli=daily_hli, T=5)
        np.testing.assert_allclose(result["A"].values, 0.01, atol=1e-10)
        np.testing.assert_allclose(result["B"].values, 0.10, atol=1e-10)


class TestHighLowSpreadIntensityEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        values = np.ones(10) * 0.05
        values[3] = np.nan
        daily_hli = pd.DataFrame(values, index=dates, columns=stocks)

        result = factor.compute(daily_hli=daily_hli, T=5)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily_hli = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_hli=daily_hli, T=5)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily_hli = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_hli=daily_hli, T=5)
        np.testing.assert_allclose(result["A"].values, 0.0, atol=1e-15)


class TestHighLowSpreadIntensityOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily_hli = pd.DataFrame(
            np.random.uniform(0.0, 0.1, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_hli=daily_hli, T=20)
        assert result.shape == daily_hli.shape
        assert list(result.columns) == list(daily_hli.columns)
        assert list(result.index) == list(daily_hli.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_hli = pd.DataFrame(
            [0.01, 0.02, 0.03, 0.02, 0.01], index=dates, columns=stocks
        )

        result = factor.compute(daily_hli=daily_hli, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_no_leading_nan_min_periods_1(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        daily_hli = pd.DataFrame(
            np.random.uniform(0.0, 0.1, (10, 2)), index=dates, columns=stocks
        )

        result = factor.compute(daily_hli=daily_hli, T=20)
        assert result.iloc[0].notna().all()
