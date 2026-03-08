import numpy as np
import pandas as pd
import pytest

from factors.consistent_trading_volume import ConsistentTradingVolumeFactor


@pytest.fixture
def factor():
    return ConsistentTradingVolumeFactor()


class TestTCVMetadata:
    def test_name(self, factor):
        assert factor.name == "TCV"

    def test_category(self, factor):
        assert factor.category == "高频成交分布"

    def test_repr(self, factor):
        assert "TCV" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "TCV"
        assert meta["category"] == "高频成交分布"


class TestTCVHandCalculated:
    def test_constant_input(self, factor):
        """Constant input => rolling mean = constant."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(0.6, index=dates, columns=stocks)

        result = factor.compute(daily_consistent_volume_ratio=data, T=5)
        np.testing.assert_allclose(result["A"].values, 0.6, atol=1e-15)

    def test_rolling_mean_T3(self, factor):
        """T=3 rolling mean hand calculation.
        data = [0.2, 0.4, 0.6, 0.8, 1.0]
        rolling(3, min_periods=1):
          day0: mean([0.2]) = 0.2
          day1: mean([0.2, 0.4]) = 0.3
          day2: mean([0.2, 0.4, 0.6]) = 0.4
          day3: mean([0.4, 0.6, 0.8]) = 0.6
          day4: mean([0.6, 0.8, 1.0]) = 0.8
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        data = pd.DataFrame([0.2, 0.4, 0.6, 0.8, 1.0], index=dates, columns=stocks)

        result = factor.compute(daily_consistent_volume_ratio=data, T=3)
        expected = [0.2, 0.3, 0.4, 0.6, 0.8]
        np.testing.assert_allclose(result["A"].values, expected, atol=1e-15)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        data = pd.DataFrame({"A": [0.3] * 10, "B": [0.7] * 10}, index=dates)

        result = factor.compute(daily_consistent_volume_ratio=data, T=5)
        np.testing.assert_allclose(result["A"].values, 0.3, atol=1e-15)
        np.testing.assert_allclose(result["B"].values, 0.7, atol=1e-15)


class TestTCVEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        vals = np.ones(10) * 0.5
        vals[3] = np.nan
        data = pd.DataFrame(vals, index=dates, columns=stocks)

        result = factor.compute(daily_consistent_volume_ratio=data, T=5)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_consistent_volume_ratio=data, T=5)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_consistent_volume_ratio=data, T=5)
        np.testing.assert_array_almost_equal(result["A"].values, 0.0)


class TestTCVOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        data = pd.DataFrame(
            np.random.uniform(0, 1, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_consistent_volume_ratio=data, T=20)
        assert result.shape == data.shape
        assert list(result.columns) == list(data.columns)
        assert list(result.index) == list(data.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(0.5, index=dates, columns=stocks)

        result = factor.compute(daily_consistent_volume_ratio=data, T=3)
        assert isinstance(result, pd.DataFrame)
