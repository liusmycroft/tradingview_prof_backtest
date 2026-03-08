import numpy as np
import pandas as pd
import pytest

from factors.customer_centrality_change import CustomerCentralityChangeFactor


@pytest.fixture
def factor():
    return CustomerCentralityChangeFactor()


class TestCustomerCentralityChangeMetadata:
    def test_name(self, factor):
        assert factor.name == "CUSTOMER_CENTRALITY_CHANGE"

    def test_category(self, factor):
        assert factor.category == "图谱网络-网络结构"

    def test_repr(self, factor):
        assert "CUSTOMER_CENTRALITY_CHANGE" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "CUSTOMER_CENTRALITY_CHANGE"
        assert meta["category"] == "图谱网络-网络结构"


class TestCustomerCentralityChangeHandCalculated:
    """手算验证中心性变化。"""

    def test_simple_difference(self, factor):
        """current - previous 的简单差值。"""
        dates = pd.date_range("2024-01-01", periods=4, freq="QE")
        stocks = ["A", "B"]
        current = pd.DataFrame(
            {"A": [0.8, 0.7, 0.9, 0.6], "B": [0.3, 0.5, 0.4, 0.8]}, index=dates
        )
        previous = pd.DataFrame(
            {"A": [0.5, 0.6, 0.7, 0.8], "B": [0.4, 0.3, 0.5, 0.6]}, index=dates
        )

        result = factor.compute(
            current_centrality=current, previous_centrality=previous
        )

        expected_a = [0.3, 0.1, 0.2, -0.2]
        expected_b = [-0.1, 0.2, -0.1, 0.2]
        for i in range(4):
            assert result.iloc[i]["A"] == pytest.approx(expected_a[i], rel=1e-10)
            assert result.iloc[i]["B"] == pytest.approx(expected_b[i], rel=1e-10)

    def test_no_change(self, factor):
        """当前期与上期相同时, 因子值为 0。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="QE")
        stocks = ["A"]
        centrality = pd.DataFrame(0.5, index=dates, columns=stocks)

        result = factor.compute(
            current_centrality=centrality, previous_centrality=centrality
        )

        for val in result["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=2, freq="QE")
        stocks = ["A", "B"]
        current = pd.DataFrame({"A": [0.9, 0.8], "B": [0.2, 0.3]}, index=dates)
        previous = pd.DataFrame({"A": [0.5, 0.5], "B": [0.5, 0.5]}, index=dates)

        result = factor.compute(
            current_centrality=current, previous_centrality=previous
        )

        assert result.iloc[0]["A"] == pytest.approx(0.4, rel=1e-10)
        assert result.iloc[0]["B"] == pytest.approx(-0.3, rel=1e-10)


class TestCustomerCentralityChangeEdgeCases:
    def test_single_period(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="QE")
        stocks = ["A"]
        current = pd.DataFrame([0.8], index=dates, columns=stocks)
        previous = pd.DataFrame([0.3], index=dates, columns=stocks)

        result = factor.compute(
            current_centrality=current, previous_centrality=previous
        )
        assert result.iloc[0, 0] == pytest.approx(0.5, rel=1e-10)

    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=3, freq="QE")
        stocks = ["A"]
        current = pd.DataFrame([0.8, np.nan, 0.6], index=dates, columns=stocks)
        previous = pd.DataFrame([0.3, 0.4, 0.5], index=dates, columns=stocks)

        result = factor.compute(
            current_centrality=current, previous_centrality=previous
        )
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 1)
        assert np.isnan(result.iloc[1, 0])

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=3, freq="QE")
        stocks = ["A"]
        current = pd.DataFrame(np.nan, index=dates, columns=stocks)
        previous = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(
            current_centrality=current, previous_centrality=previous
        )
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=3, freq="QE")
        stocks = ["A"]
        current = pd.DataFrame(0.0, index=dates, columns=stocks)
        previous = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(
            current_centrality=current, previous_centrality=previous
        )
        for val in result["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)


class TestCustomerCentralityChangeOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=4, freq="QE")
        stocks = ["A", "B", "C"]
        current = pd.DataFrame(
            np.random.uniform(0, 1, (4, 3)), index=dates, columns=stocks
        )
        previous = pd.DataFrame(
            np.random.uniform(0, 1, (4, 3)), index=dates, columns=stocks
        )

        result = factor.compute(
            current_centrality=current, previous_centrality=previous
        )

        assert result.shape == current.shape
        assert list(result.columns) == list(current.columns)
        assert list(result.index) == list(current.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=2, freq="QE")
        stocks = ["A"]
        current = pd.DataFrame([0.8, 0.9], index=dates, columns=stocks)
        previous = pd.DataFrame([0.5, 0.6], index=dates, columns=stocks)

        result = factor.compute(
            current_centrality=current, previous_centrality=previous
        )
        assert isinstance(result, pd.DataFrame)
