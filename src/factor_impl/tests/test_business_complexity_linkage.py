import numpy as np
import pandas as pd
import pytest

from factors.business_complexity_linkage import BusinessComplexityLinkageFactor


@pytest.fixture
def factor():
    return BusinessComplexityLinkageFactor()


class TestBusinessComplexityLinkageMetadata:
    def test_name(self, factor):
        assert factor.name == "BIZ_COMPLEXITY_LINKAGE"

    def test_category(self, factor):
        assert factor.category == "行为金融-投资者注意力"

    def test_repr(self, factor):
        assert "BIZ_COMPLEXITY_LINKAGE" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "BIZ_COMPLEXITY_LINKAGE"
        assert meta["category"] == "行为金融-投资者注意力"


class TestBusinessComplexityLinkageHandCalculated:
    def test_constant_inputs(self, factor):
        """complexity=2, sim_excess_ret=3 => linkage=6, EMA of constant=6."""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        complexity = pd.DataFrame(2.0, index=dates, columns=stocks)
        sim_excess_ret = pd.DataFrame(3.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_complexity=complexity,
            daily_sim_excess_ret=sim_excess_ret,
            T=20,
        )
        np.testing.assert_array_almost_equal(result["A"].values, 6.0)

    def test_zero_complexity(self, factor):
        """complexity=0 => linkage=0 regardless of sim_excess_ret."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        complexity = pd.DataFrame(0.0, index=dates, columns=stocks)
        sim_excess_ret = pd.DataFrame(5.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_complexity=complexity,
            daily_sim_excess_ret=sim_excess_ret,
            T=10,
        )
        np.testing.assert_array_almost_equal(result["A"].values, 0.0)

    def test_two_stocks_independent(self, factor):
        """Two stocks with constant but different values."""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A", "B"]
        complexity = pd.DataFrame({"A": [1.0] * 20, "B": [2.0] * 20}, index=dates)
        sim_excess_ret = pd.DataFrame({"A": [3.0] * 20, "B": [4.0] * 20}, index=dates)

        result = factor.compute(
            daily_complexity=complexity,
            daily_sim_excess_ret=sim_excess_ret,
            T=20,
        )
        np.testing.assert_array_almost_equal(result["A"].values, 3.0)
        np.testing.assert_array_almost_equal(result["B"].values, 8.0)


class TestBusinessComplexityLinkageEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        complexity_vals = np.ones(10) * 2.0
        complexity_vals[3] = np.nan
        complexity = pd.DataFrame(complexity_vals, index=dates, columns=stocks)
        sim_excess_ret = pd.DataFrame(3.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_complexity=complexity,
            daily_sim_excess_ret=sim_excess_ret,
            T=5,
        )
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        complexity = pd.DataFrame(np.nan, index=dates, columns=stocks)
        sim_excess_ret = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(
            daily_complexity=complexity,
            daily_sim_excess_ret=sim_excess_ret,
            T=5,
        )
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        complexity = pd.DataFrame(0.0, index=dates, columns=stocks)
        sim_excess_ret = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_complexity=complexity,
            daily_sim_excess_ret=sim_excess_ret,
            T=5,
        )
        np.testing.assert_array_almost_equal(result["A"].values, 0.0)


class TestBusinessComplexityLinkageOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        complexity = pd.DataFrame(
            np.random.uniform(0, 1, (30, 3)), index=dates, columns=stocks
        )
        sim_excess_ret = pd.DataFrame(
            np.random.uniform(-0.1, 0.1, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(
            daily_complexity=complexity,
            daily_sim_excess_ret=sim_excess_ret,
            T=20,
        )
        assert result.shape == complexity.shape
        assert list(result.columns) == list(complexity.columns)
        assert list(result.index) == list(complexity.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        complexity = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0], index=dates, columns=stocks)
        sim_excess_ret = pd.DataFrame([0.1, 0.2, 0.3, 0.4, 0.5], index=dates, columns=stocks)

        result = factor.compute(
            daily_complexity=complexity,
            daily_sim_excess_ret=sim_excess_ret,
            T=3,
        )
        assert isinstance(result, pd.DataFrame)
