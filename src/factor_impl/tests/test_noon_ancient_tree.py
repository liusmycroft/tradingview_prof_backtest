import numpy as np
import pandas as pd
import pytest

from factors.noon_ancient_tree import NoonAncientTreeFactor


@pytest.fixture
def factor():
    return NoonAncientTreeFactor()


class TestNoonAncientTreeMetadata:
    def test_name(self, factor):
        assert factor.name == "NOON_ANCIENT_TREE"

    def test_category(self, factor):
        assert factor.category == "高频量价"

    def test_repr(self, factor):
        assert "NOON_ANCIENT_TREE" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "NOON_ANCIENT_TREE"
        assert meta["category"] == "高频量价"


class TestNoonAncientTreeHandCalculated:
    def test_basic_rolling_mean(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=5)
        data = pd.DataFrame({"A": [0.5, 1.0, 1.5, 2.0, 2.5]}, index=dates)
        result = factor.compute(daily_noon_ancient_tree=data, T=3)
        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(1.0)
        assert result.iloc[3, 0] == pytest.approx(1.5)
        assert result.iloc[4, 0] == pytest.approx(2.0)

    def test_constant_input(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=5)
        data = pd.DataFrame({"A": [1.5] * 5}, index=dates)
        result = factor.compute(daily_noon_ancient_tree=data, T=3)
        assert result.iloc[2, 0] == pytest.approx(1.5)

    def test_multi_stock(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=5)
        data = pd.DataFrame(
            {"A": [1.0, 2.0, 3.0, 4.0, 5.0], "B": [10.0, 20.0, 30.0, 40.0, 50.0]},
            index=dates,
        )
        result = factor.compute(daily_noon_ancient_tree=data, T=3)
        assert result.iloc[2, 0] == pytest.approx(2.0)
        assert result.iloc[2, 1] == pytest.approx(20.0)


class TestNoonAncientTreeEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=5)
        data = pd.DataFrame({"A": [1.0, np.nan, 3.0, 4.0, 5.0]}, index=dates)
        result = factor.compute(daily_noon_ancient_tree=data, T=3)
        assert isinstance(result, pd.DataFrame)
        assert np.isnan(result.iloc[2, 0])

    def test_all_nan(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=10)
        data = pd.DataFrame(np.nan, index=dates, columns=["A"])
        result = factor.compute(daily_noon_ancient_tree=data, T=5)
        assert result.isna().all().all()

    def test_negative_values(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=5)
        data = pd.DataFrame({"A": [-1.0, -2.0, -3.0, -4.0, -5.0]}, index=dates)
        result = factor.compute(daily_noon_ancient_tree=data, T=3)
        assert result.iloc[2, 0] == pytest.approx(-2.0)


class TestNoonAncientTreeOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=30)
        stocks = ["A", "B", "C"]
        data = pd.DataFrame(
            np.random.uniform(-2.0, 2.0, (30, 3)), index=dates, columns=stocks
        )
        result = factor.compute(daily_noon_ancient_tree=data, T=20)
        assert result.shape == data.shape
        assert list(result.columns) == list(data.columns)

    def test_output_is_dataframe(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=5)
        data = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=dates)
        result = factor.compute(daily_noon_ancient_tree=data, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=25)
        data = pd.DataFrame(
            np.random.uniform(-2.0, 2.0, (25, 2)), index=dates, columns=["A", "B"]
        )
        T = 20
        result = factor.compute(daily_noon_ancient_tree=data, T=T)
        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
