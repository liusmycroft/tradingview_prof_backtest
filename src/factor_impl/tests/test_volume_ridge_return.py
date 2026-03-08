import numpy as np
import pandas as pd
import pytest

from factors.volume_ridge_return import VolumeRidgeReturnFactor


@pytest.fixture
def factor():
    return VolumeRidgeReturnFactor()


class TestVolumeRidgeReturnMetadata:
    def test_name(self, factor):
        assert factor.name == "VOLUME_RIDGE_RETURN"

    def test_category(self, factor):
        assert factor.category == "高频量价相关性"

    def test_repr(self, factor):
        assert "VOLUME_RIDGE_RETURN" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "VOLUME_RIDGE_RETURN"
        assert meta["category"] == "高频量价相关性"


class TestVolumeRidgeReturnHandCalculated:
    def test_basic_rolling_mean_T3(self, factor):
        """T=3, min_periods=T, hand-calculated rolling mean."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame({"A": [0.01, 0.02, 0.03, 0.04, 0.05]}, index=dates)

        result = factor.compute(daily_ridge_return=data, T=3)

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(0.02, rel=1e-10)  # mean(0.01,0.02,0.03)
        assert result.iloc[3, 0] == pytest.approx(0.03, rel=1e-10)  # mean(0.02,0.03,0.04)
        assert result.iloc[4, 0] == pytest.approx(0.04, rel=1e-10)  # mean(0.03,0.04,0.05)

    def test_constant_input(self, factor):
        """Constant input => rolling mean equals that constant (after warmup)."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame({"A": [0.05] * 10}, index=dates)

        result = factor.compute(daily_ridge_return=data, T=5)

        assert result.iloc[4, 0] == pytest.approx(0.05, rel=1e-10)
        assert result.iloc[-1, 0] == pytest.approx(0.05, rel=1e-10)

    def test_two_stocks_independent(self, factor):
        """Two stocks computed independently."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame(
            {"A": [0.01, 0.02, 0.03, 0.04, 0.05], "B": [0.10, 0.20, 0.30, 0.40, 0.50]},
            index=dates,
        )

        result = factor.compute(daily_ridge_return=data, T=3)

        assert result.iloc[2, 0] == pytest.approx(0.02, rel=1e-10)
        assert result.iloc[2, 1] == pytest.approx(0.20, rel=1e-10)


class TestVolumeRidgeReturnEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame({"A": [0.01, np.nan, 0.03, 0.04, 0.05]}, index=dates)

        result = factor.compute(daily_ridge_return=data, T=3)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (5, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame({"A": [np.nan] * 5}, index=dates)

        result = factor.compute(daily_ridge_return=data, T=3)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame({"A": [0.0] * 5}, index=dates)

        result = factor.compute(daily_ridge_return=data, T=3)
        assert result.iloc[2, 0] == pytest.approx(0.0, abs=1e-15)
        assert result.iloc[-1, 0] == pytest.approx(0.0, abs=1e-15)

    def test_insufficient_window(self, factor):
        """Data shorter than T => all NaN."""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        data = pd.DataFrame({"A": [0.01, 0.02, 0.03]}, index=dates)

        result = factor.compute(daily_ridge_return=data, T=5)
        assert result.isna().all().all()


class TestVolumeRidgeReturnOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        data = pd.DataFrame(
            np.random.uniform(-0.05, 0.05, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_ridge_return=data, T=20)
        assert result.shape == data.shape
        assert list(result.columns) == list(data.columns)
        assert list(result.index) == list(data.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame({"A": [0.01, 0.02, 0.03, 0.04, 0.05]}, index=dates)

        result = factor.compute(daily_ridge_return=data, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        """First T-1 rows should be NaN (min_periods=T)."""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        data = pd.DataFrame(
            np.random.uniform(-0.05, 0.05, (25, 2)), index=dates, columns=["A", "B"]
        )
        T = 20

        result = factor.compute(daily_ridge_return=data, T=T)
        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
