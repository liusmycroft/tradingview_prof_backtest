import numpy as np
import pandas as pd
import pytest

from factors.large_order_dpin import LargeOrderDpinFactor


@pytest.fixture
def factor():
    return LargeOrderDpinFactor()


class TestLargeOrderDpinMetadata:
    def test_name(self, factor):
        assert factor.name == "LARGE_ORDER_DPIN"

    def test_category(self, factor):
        assert factor.category == "高频资金流"

    def test_repr(self, factor):
        assert "LARGE_ORDER_DPIN" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "LARGE_ORDER_DPIN"
        assert meta["category"] == "高频资金流"


class TestLargeOrderDpinCompute:
    def test_constant_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        daily = pd.DataFrame(0.3, index=dates, columns=["A"])

        result = factor.compute(daily_large_dpin=daily, T=20)
        np.testing.assert_array_almost_equal(result["A"].values, 0.3)

    def test_rolling_mean(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        daily = pd.DataFrame([0.1, 0.2, 0.3, 0.4, 0.5], index=dates, columns=["A"])

        result = factor.compute(daily_large_dpin=daily, T=3)
        assert result.iloc[0, 0] == pytest.approx(0.1)
        assert result.iloc[1, 0] == pytest.approx(0.15)
        assert result.iloc[2, 0] == pytest.approx(0.2)
        assert result.iloc[3, 0] == pytest.approx(0.3)
        assert result.iloc[4, 0] == pytest.approx(0.4)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        daily = pd.DataFrame({"A": [0.2]*10, "B": [0.5]*10}, index=dates)

        result = factor.compute(daily_large_dpin=daily, T=5)
        np.testing.assert_array_almost_equal(result["A"].values, 0.2)
        np.testing.assert_array_almost_equal(result["B"].values, 0.5)

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily = pd.DataFrame(np.random.rand(30, 3), index=dates, columns=stocks)

        result = factor.compute(daily_large_dpin=daily, T=20)
        assert result.shape == daily.shape


class TestLargeOrderDpinEdgeCases:
    def test_single_value(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        daily = pd.DataFrame([0.25], index=dates, columns=["A"])

        result = factor.compute(daily_large_dpin=daily, T=20)
        assert result.iloc[0, 0] == pytest.approx(0.25)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        daily = pd.DataFrame(np.nan, index=dates, columns=["A"])

        result = factor.compute(daily_large_dpin=daily, T=5)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        daily = pd.DataFrame(0.0, index=dates, columns=["A"])

        result = factor.compute(daily_large_dpin=daily, T=5)
        for val in result["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)
