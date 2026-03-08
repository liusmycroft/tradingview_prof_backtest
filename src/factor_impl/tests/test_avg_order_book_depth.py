import numpy as np
import pandas as pd
import pytest

from factors.avg_order_book_depth import AvgOrderBookDepthFactor


@pytest.fixture
def factor():
    return AvgOrderBookDepthFactor()


class TestAvgOrderBookDepthMetadata:
    def test_name(self, factor):
        assert factor.name == "AVG_ORDER_BOOK_DEPTH"

    def test_category(self, factor):
        assert factor.category == "高频流动性"

    def test_repr(self, factor):
        assert "AVG_ORDER_BOOK_DEPTH" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "AVG_ORDER_BOOK_DEPTH"
        assert meta["category"] == "高频流动性"


class TestAvgOrderBookDepthHandCalculated:
    def test_constant_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        data = pd.DataFrame(500.0, index=dates, columns=["A"])
        result = factor.compute(daily_avg_depth=data, T=20)
        np.testing.assert_array_almost_equal(result["A"].values, 500.0)

    def test_ema_manual_T3(self, factor):
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        data = pd.DataFrame([10.0, 20.0, 30.0, 40.0], index=dates, columns=["A"])
        result = factor.compute(daily_avg_depth=data, T=3)
        assert result.iloc[0, 0] == pytest.approx(10.0, rel=1e-6)
        assert result.iloc[1, 0] == pytest.approx(50 / 3, rel=1e-6)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame({"A": [100.0] * 10, "B": [500.0] * 10}, index=dates)
        result = factor.compute(daily_avg_depth=data, T=5)
        np.testing.assert_array_almost_equal(result["A"].values, 100.0)
        np.testing.assert_array_almost_equal(result["B"].values, 500.0)


class TestAvgOrderBookDepthEdgeCases:
    def test_single_value(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        data = pd.DataFrame([42.0], index=dates, columns=["A"])
        result = factor.compute(daily_avg_depth=data, T=20)
        assert result.iloc[0, 0] == pytest.approx(42.0, rel=1e-10)

    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        values = np.ones(10) * 100.0
        values[3] = np.nan
        data = pd.DataFrame(values, index=dates, columns=["A"])
        result = factor.compute(daily_avg_depth=data, T=5)
        assert isinstance(result, pd.DataFrame)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(np.nan, index=dates, columns=["A"])
        result = factor.compute(daily_avg_depth=data, T=5)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(0.0, index=dates, columns=["A"])
        result = factor.compute(daily_avg_depth=data, T=5)
        for val in result["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)


class TestAvgOrderBookDepthOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        data = pd.DataFrame(
            np.random.uniform(100, 1000, (30, 3)), index=dates, columns=stocks
        )
        result = factor.compute(daily_avg_depth=data, T=20)
        assert result.shape == data.shape

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame([100, 200, 300, 400, 500], index=dates, columns=["A"], dtype=float)
        result = factor.compute(daily_avg_depth=data, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_no_leading_nan_min_periods_1(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(
            np.random.uniform(100, 1000, (10, 2)), index=dates, columns=["A", "B"]
        )
        result = factor.compute(daily_avg_depth=data, T=20)
        assert result.iloc[0].notna().all()
