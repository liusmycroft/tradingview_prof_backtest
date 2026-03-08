import numpy as np
import pandas as pd
import pytest

from factors.smm import SMMFactor


@pytest.fixture
def factor():
    return SMMFactor()


class TestSMMMetadata:
    def test_name(self, factor):
        assert factor.name == "SMM"

    def test_category(self, factor):
        assert factor.category == "图谱网络-动量溢出"

    def test_repr(self, factor):
        assert "SMM" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "SMM"
        assert meta["category"] == "图谱网络-动量溢出"


class TestSMMHandCalculated:
    def test_constant_input(self, factor):
        """Constant input: rolling sum = constant * min(i+1, T)."""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        data = pd.DataFrame(0.01, index=dates, columns=["A"])

        result = factor.compute(daily_supplier_ret=data, T=10)

        # After window fills: sum = 0.01 * 10 = 0.1
        assert result.iloc[9, 0] == pytest.approx(0.1, rel=1e-10)
        assert result.iloc[19, 0] == pytest.approx(0.1, rel=1e-10)

    def test_rolling_sum_T3(self, factor):
        """T=3 rolling sum hand-calculated (min_periods=1)."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame(
            [0.01, 0.02, 0.03, 0.04, 0.05], index=dates, columns=["A"]
        )

        result = factor.compute(daily_supplier_ret=data, T=3)

        assert result.iloc[0, 0] == pytest.approx(0.01, rel=1e-10)
        assert result.iloc[1, 0] == pytest.approx(0.03, rel=1e-10)
        assert result.iloc[2, 0] == pytest.approx(0.06, rel=1e-10)
        # sum(0.02, 0.03, 0.04) = 0.09
        assert result.iloc[3, 0] == pytest.approx(0.09, rel=1e-10)
        # sum(0.03, 0.04, 0.05) = 0.12
        assert result.iloc[4, 0] == pytest.approx(0.12, rel=1e-10)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(
            {"A": [0.01] * 10, "B": [0.05] * 10}, index=dates
        )

        result = factor.compute(daily_supplier_ret=data, T=5)
        np.testing.assert_array_almost_equal(result["A"].iloc[4:].values, 0.05)
        np.testing.assert_array_almost_equal(result["B"].iloc[4:].values, 0.25)


class TestSMMEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        values = np.ones(10) * 0.01
        values[3] = np.nan
        data = pd.DataFrame(values, index=dates, columns=["A"])

        result = factor.compute(daily_supplier_ret=data, T=5)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(np.nan, index=dates, columns=["A"])

        result = factor.compute(daily_supplier_ret=data, T=5)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(0.0, index=dates, columns=["A"])

        result = factor.compute(daily_supplier_ret=data, T=5)
        for val in result["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)


class TestSMMOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        data = pd.DataFrame(
            np.random.uniform(-0.05, 0.05, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_supplier_ret=data, T=20)
        assert result.shape == data.shape
        assert list(result.columns) == list(data.columns)
        assert list(result.index) == list(data.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame(
            [0.01, 0.02, 0.03, 0.04, 0.05], index=dates, columns=["A"]
        )

        result = factor.compute(daily_supplier_ret=data, T=3)
        assert isinstance(result, pd.DataFrame)
