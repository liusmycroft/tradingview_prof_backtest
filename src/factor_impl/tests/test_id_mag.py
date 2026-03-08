import numpy as np
import pandas as pd
import pytest

from factors.id_mag import IDMagFactor


@pytest.fixture
def factor():
    return IDMagFactor()


class TestIDMagMetadata:
    def test_name(self, factor):
        assert factor.name == "ID_MAG"

    def test_category(self, factor):
        assert factor.category == "量价因子改进"

    def test_repr(self, factor):
        assert "ID_MAG" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "ID_MAG"
        assert meta["category"] == "量价因子改进"


class TestIDMagHandCalculated:
    def test_simple_case(self, factor):
        """Hand-verify: all returns positive, pret positive, weights=1.
        signed_weighted = sign(ret)*w = 1*1 = 1 per day
        rolling_sum(220) over 220 days = 220
        result = -(1/220)*sign(pret)*220 = -1.0
        """
        N = 220
        dates = pd.date_range("2024-01-01", periods=N, freq="D")
        stocks = ["A"]
        daily_returns = pd.DataFrame(0.01, index=dates, columns=stocks)
        pret = pd.DataFrame(0.05, index=dates, columns=stocks)
        weights = pd.DataFrame(1.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_returns=daily_returns, pret=pret,
            daily_magnitude_weights=weights, N=N,
        )
        assert result.iloc[-1, 0] == pytest.approx(-1.0, rel=1e-10)

    def test_negative_returns_positive_pret(self, factor):
        """All returns negative, pret positive, weights=1.
        signed_weighted = sign(-0.01)*1 = -1 per day
        rolling_sum = -220
        result = -(1/220)*sign(0.05)*(-220) = 1.0
        """
        N = 220
        dates = pd.date_range("2024-01-01", periods=N, freq="D")
        stocks = ["A"]
        daily_returns = pd.DataFrame(-0.01, index=dates, columns=stocks)
        pret = pd.DataFrame(0.05, index=dates, columns=stocks)
        weights = pd.DataFrame(1.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_returns=daily_returns, pret=pret,
            daily_magnitude_weights=weights, N=N,
        )
        assert result.iloc[-1, 0] == pytest.approx(1.0, rel=1e-10)

    def test_min_periods_produces_nan(self, factor):
        """With N=220 and only 100 rows, all output should be NaN."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        stocks = ["A"]
        daily_returns = pd.DataFrame(0.01, index=dates, columns=stocks)
        pret = pd.DataFrame(0.05, index=dates, columns=stocks)
        weights = pd.DataFrame(1.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_returns=daily_returns, pret=pret,
            daily_magnitude_weights=weights, N=220,
        )
        assert result.isna().all().all()


class TestIDMagEdgeCases:
    def test_nan_in_input(self, factor):
        N = 220
        dates = pd.date_range("2024-01-01", periods=N + 10, freq="D")
        stocks = ["A"]
        daily_returns = pd.DataFrame(0.01, index=dates, columns=stocks)
        daily_returns.iloc[5, 0] = np.nan
        pret = pd.DataFrame(0.05, index=dates, columns=stocks)
        weights = pd.DataFrame(1.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_returns=daily_returns, pret=pret,
            daily_magnitude_weights=weights, N=N,
        )
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (N + 10, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=230, freq="D")
        stocks = ["A"]
        daily_returns = pd.DataFrame(np.nan, index=dates, columns=stocks)
        pret = pd.DataFrame(np.nan, index=dates, columns=stocks)
        weights = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(
            daily_returns=daily_returns, pret=pret,
            daily_magnitude_weights=weights, N=220,
        )
        assert result.isna().all().all()

    def test_zero_returns(self, factor):
        """Zero returns -> sign(0)=0 -> rolling_sum=0 -> result=0."""
        N = 220
        dates = pd.date_range("2024-01-01", periods=N, freq="D")
        stocks = ["A"]
        daily_returns = pd.DataFrame(0.0, index=dates, columns=stocks)
        pret = pd.DataFrame(0.05, index=dates, columns=stocks)
        weights = pd.DataFrame(1.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_returns=daily_returns, pret=pret,
            daily_magnitude_weights=weights, N=N,
        )
        assert result.iloc[-1, 0] == pytest.approx(0.0, abs=1e-10)


class TestIDMagOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=250, freq="D")
        stocks = ["A", "B", "C"]
        np.random.seed(42)
        daily_returns = pd.DataFrame(
            np.random.randn(250, 3) * 0.02, index=dates, columns=stocks
        )
        pret = pd.DataFrame(
            np.random.randn(250, 3) * 0.1, index=dates, columns=stocks
        )
        weights = pd.DataFrame(
            np.random.rand(250, 3), index=dates, columns=stocks
        )

        result = factor.compute(
            daily_returns=daily_returns, pret=pret,
            daily_magnitude_weights=weights, N=220,
        )
        assert result.shape == daily_returns.shape
        assert list(result.columns) == list(daily_returns.columns)
        assert list(result.index) == list(daily_returns.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=220, freq="D")
        stocks = ["A"]
        daily_returns = pd.DataFrame(0.01, index=dates, columns=stocks)
        pret = pd.DataFrame(0.05, index=dates, columns=stocks)
        weights = pd.DataFrame(1.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_returns=daily_returns, pret=pret,
            daily_magnitude_weights=weights, N=220,
        )
        assert isinstance(result, pd.DataFrame)
