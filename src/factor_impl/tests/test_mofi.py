import numpy as np
import pandas as pd
import pytest

from factors.mofi import MOFIFactor


@pytest.fixture
def factor():
    return MOFIFactor()


def _make_ofi(dates, stocks, value):
    """Helper to create constant OFI DataFrames."""
    return pd.DataFrame(value, index=dates, columns=stocks)


class TestMOFIMetadata:
    def test_name(self, factor):
        assert factor.name == "MOFI"

    def test_category(self, factor):
        assert factor.category == "高频流动性"

    def test_repr(self, factor):
        assert "MOFI" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "MOFI"
        assert meta["category"] == "高频流动性"


class TestMOFIHandCalculated:
    def test_constant_input(self, factor):
        """所有档位OFI相同常数时，MOFI应等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        ofi = {f"daily_ofi_{i}": _make_ofi(dates, stocks, 100.0) for i in range(1, 6)}

        result = factor.compute(**ofi, T=5)
        # weighted avg of same value = same value
        np.testing.assert_array_almost_equal(result["A"].values, 100.0)

    def test_weighted_average_manual(self, factor):
        """手动验证加权平均：w=[0.2,0.4,0.6,0.8,1.0], sum=3.0。"""
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A"]
        # OFI values: 10, 20, 30, 40, 50
        ofi = {}
        for i in range(1, 6):
            ofi[f"daily_ofi_{i}"] = pd.DataFrame([i * 10.0], index=dates, columns=stocks)

        result = factor.compute(**ofi, T=1)

        # weighted = (0.2*10 + 0.4*20 + 0.6*30 + 0.8*40 + 1.0*50) / 3.0
        # = (2 + 8 + 18 + 32 + 50) / 3.0 = 110 / 3.0 = 36.6667
        expected = (0.2 * 10 + 0.4 * 20 + 0.6 * 30 + 0.8 * 40 + 1.0 * 50) / 3.0
        assert result.iloc[0, 0] == pytest.approx(expected, rel=1e-6)

    def test_zero_ofi(self, factor):
        """全零OFI时，MOFI应为0。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        ofi = {f"daily_ofi_{i}": _make_ofi(dates, stocks, 0.0) for i in range(1, 6)}

        result = factor.compute(**ofi, T=5)
        for val in result["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        ofi = {}
        for i in range(1, 6):
            ofi[f"daily_ofi_{i}"] = pd.DataFrame(
                {"A": [10.0] * 10, "B": [50.0] * 10}, index=dates
            )

        result = factor.compute(**ofi, T=5)
        np.testing.assert_array_almost_equal(result["A"].values, 10.0)
        np.testing.assert_array_almost_equal(result["B"].values, 50.0)


class TestMOFIEdgeCases:
    def test_single_value(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A"]
        ofi = {f"daily_ofi_{i}": pd.DataFrame([42.0], index=dates, columns=stocks) for i in range(1, 6)}

        result = factor.compute(**ofi, T=20)
        assert result.iloc[0, 0] == pytest.approx(42.0, rel=1e-10)

    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        values = np.ones(10) * 10.0
        values[3] = np.nan
        ofi = {}
        for i in range(1, 6):
            ofi[f"daily_ofi_{i}"] = pd.DataFrame(values.copy(), index=dates, columns=stocks)

        result = factor.compute(**ofi, T=5)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        ofi = {f"daily_ofi_{i}": pd.DataFrame(np.nan, index=dates, columns=stocks) for i in range(1, 6)}

        result = factor.compute(**ofi, T=5)
        assert result.isna().all().all()


class TestMOFIOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        ofi = {f"daily_ofi_{i}": pd.DataFrame(
            np.random.randn(30, 3) * 100, index=dates, columns=stocks
        ) for i in range(1, 6)}

        result = factor.compute(**ofi, T=20)
        assert result.shape == (30, 3)
        assert list(result.columns) == stocks
        assert list(result.index) == list(dates)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        ofi = {f"daily_ofi_{i}": pd.DataFrame(
            [1.0, 2.0, 3.0, 4.0, 5.0], index=dates, columns=stocks
        ) for i in range(1, 6)}

        result = factor.compute(**ofi, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_no_leading_nan_min_periods_1(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        ofi = {f"daily_ofi_{i}": pd.DataFrame(
            np.random.uniform(1, 100, (10, 2)), index=dates, columns=stocks
        ) for i in range(1, 6)}

        result = factor.compute(**ofi, T=20)
        assert result.iloc[0].notna().all()
        assert result.notna().all().all()
