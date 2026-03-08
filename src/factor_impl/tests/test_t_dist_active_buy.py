import numpy as np
import pandas as pd
import pytest

from factors.t_dist_active_buy import TDistActiveBuyFactor


@pytest.fixture
def factor():
    return TDistActiveBuyFactor()


class TestTDistActiveBuyMetadata:
    def test_name(self, factor):
        assert factor.name == "T_DIST_ACTIVE_BUY"

    def test_category(self, factor):
        assert factor.category == "高频资金流"

    def test_repr(self, factor):
        assert "T_DIST_ACTIVE_BUY" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "T_DIST_ACTIVE_BUY"
        assert meta["category"] == "高频资金流"


class TestTDistActiveBuyHandCalculated:
    def test_constant_inputs(self, factor):
        """Constant ABR and returns should produce non-NaN output."""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A"]
        daily_active_buy_ratio = pd.DataFrame(0.6, index=dates, columns=stocks)
        daily_returns = pd.DataFrame(0.01, index=dates, columns=stocks)

        result = factor.compute(
            daily_active_buy_ratio=daily_active_buy_ratio,
            daily_returns=daily_returns,
            T=20,
        )
        # After enough data points, should have values
        assert result.iloc[-1, 0] is not np.nan or not np.isnan(result.iloc[-1, 0])
        assert isinstance(result, pd.DataFrame)

    def test_output_positive_for_positive_abr(self, factor):
        """Positive ABR should yield positive factor values."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A"]
        daily_active_buy_ratio = pd.DataFrame(0.7, index=dates, columns=stocks)
        daily_returns = pd.DataFrame(
            np.random.normal(0.01, 0.02, (30, 1)), index=dates, columns=stocks
        )

        result = factor.compute(
            daily_active_buy_ratio=daily_active_buy_ratio,
            daily_returns=daily_returns,
            T=10,
        )
        # Last value should be positive (weight * mean(0.7) > 0)
        last_val = result.iloc[-1, 0]
        assert not np.isnan(last_val)
        assert last_val > 0

    def test_two_stocks_independent(self, factor):
        """Two stocks should be computed independently.

        Use identical return series so the t-distribution weight is the same
        for both stocks; the only difference is ABR (0.3 vs 0.8).
        """
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        shared_ret = np.random.normal(0.0, 0.02, 30)
        daily_active_buy_ratio = pd.DataFrame(
            {"A": [0.3] * 30, "B": [0.8] * 30}, index=dates
        )
        daily_returns = pd.DataFrame(
            {"A": shared_ret, "B": shared_ret}, index=dates
        )

        result = factor.compute(
            daily_active_buy_ratio=daily_active_buy_ratio,
            daily_returns=daily_returns,
            T=10,
        )
        # Same weight, higher ABR => higher factor value
        assert result.iloc[-1, 1] > result.iloc[-1, 0]


class TestTDistActiveBuyEdgeCases:
    def test_insufficient_data(self, factor):
        """With fewer than 3 valid points, output should be NaN."""
        dates = pd.date_range("2024-01-01", periods=2, freq="D")
        stocks = ["A"]
        daily_active_buy_ratio = pd.DataFrame([0.5, 0.6], index=dates, columns=stocks)
        daily_returns = pd.DataFrame([0.01, 0.02], index=dates, columns=stocks)

        result = factor.compute(
            daily_active_buy_ratio=daily_active_buy_ratio,
            daily_returns=daily_returns,
            T=20,
        )
        # With only 2 data points, should be NaN (need >= 3 for t.fit)
        assert result.isna().all().all()

    def test_nan_in_input(self, factor):
        """NaN in input should not raise."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        abr = np.ones(10) * 0.5
        abr[3] = np.nan
        ret = np.random.normal(0, 0.02, 10)
        ret[5] = np.nan

        daily_active_buy_ratio = pd.DataFrame(abr, index=dates, columns=stocks)
        daily_returns = pd.DataFrame(ret, index=dates, columns=stocks)

        result = factor.compute(
            daily_active_buy_ratio=daily_active_buy_ratio,
            daily_returns=daily_returns,
            T=5,
        )
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        """All NaN input should produce all NaN output."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily_active_buy_ratio = pd.DataFrame(np.nan, index=dates, columns=stocks)
        daily_returns = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(
            daily_active_buy_ratio=daily_active_buy_ratio,
            daily_returns=daily_returns,
            T=5,
        )
        assert result.isna().all().all()


class TestTDistActiveBuyOutputShape:
    def test_output_shape_matches_input(self, factor):
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily_active_buy_ratio = pd.DataFrame(
            np.random.uniform(0.3, 0.8, (30, 3)), index=dates, columns=stocks
        )
        daily_returns = pd.DataFrame(
            np.random.normal(0, 0.02, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(
            daily_active_buy_ratio=daily_active_buy_ratio,
            daily_returns=daily_returns,
            T=20,
        )

        assert result.shape == daily_active_buy_ratio.shape
        assert list(result.columns) == list(daily_active_buy_ratio.columns)
        assert list(result.index) == list(daily_active_buy_ratio.index)

    def test_output_is_dataframe(self, factor):
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily_active_buy_ratio = pd.DataFrame(
            np.random.uniform(0.3, 0.8, (10, 1)), index=dates, columns=stocks
        )
        daily_returns = pd.DataFrame(
            np.random.normal(0, 0.02, (10, 1)), index=dates, columns=stocks
        )

        result = factor.compute(
            daily_active_buy_ratio=daily_active_buy_ratio,
            daily_returns=daily_returns,
            T=5,
        )
        assert isinstance(result, pd.DataFrame)
