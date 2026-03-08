import numpy as np
import pandas as pd
import pytest

from factors.cpqsi import CPQSIFactor


@pytest.fixture
def factor():
    return CPQSIFactor()


class TestCPQSIMetadata:
    def test_name(self, factor):
        assert factor.name == "CPQSI"

    def test_category(self, factor):
        assert factor.category == "高频流动性"

    def test_repr(self, factor):
        assert "CPQSI" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "CPQSI"
        assert meta["category"] == "高频流动性"


class TestCPQSICompute:
    def test_known_values(self, factor):
        """手算验证: CPQS = (ask-bid)/mid, CPQSI = CPQS / dollar_volume"""
        dates = pd.bdate_range("2025-01-01", periods=2)
        closing_ask = pd.DataFrame({"A": [10.10, 20.20]}, index=dates)
        closing_bid = pd.DataFrame({"A": [9.90, 19.80]}, index=dates)
        dollar_volume = pd.DataFrame({"A": [1e8, 2e8]}, index=dates)

        result = factor.compute(
            closing_ask=closing_ask, closing_bid=closing_bid, dollar_volume=dollar_volume,
        )

        # row 0: mid=10.0, cpqs=0.2/10.0=0.02, cpqsi=0.02/1e8=2e-10
        assert result.iloc[0, 0] == pytest.approx(2e-10)
        # row 1: mid=20.0, cpqs=0.4/20.0=0.02, cpqsi=0.02/2e8=1e-10
        assert result.iloc[1, 0] == pytest.approx(1e-10)

    def test_zero_spread(self, factor):
        """ask == bid 时, CPQSI = 0。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        price = pd.DataFrame({"A": [10.0, 20.0, 30.0]}, index=dates)
        dollar_volume = pd.DataFrame({"A": [1e8, 1e8, 1e8]}, index=dates)

        result = factor.compute(
            closing_ask=price, closing_bid=price, dollar_volume=dollar_volume,
        )

        for val in result["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)

    def test_multi_stock(self, factor):
        """多只股票独立计算。"""
        dates = pd.bdate_range("2025-01-01", periods=2)
        closing_ask = pd.DataFrame({"A": [10.10, 20.20], "B": [50.50, 100.50]}, index=dates)
        closing_bid = pd.DataFrame({"A": [9.90, 19.80], "B": [49.50, 99.50]}, index=dates)
        dollar_volume = pd.DataFrame({"A": [1e8, 2e8], "B": [5e8, 1e9]}, index=dates)

        result = factor.compute(
            closing_ask=closing_ask, closing_bid=closing_bid, dollar_volume=dollar_volume,
        )

        assert result.shape == (2, 2)
        # A, row 0: mid=10.0, cpqs=0.02, cpqsi=2e-10
        assert result.loc[dates[0], "A"] == pytest.approx(2e-10)
        # B, row 0: mid=50.0, cpqs=1.0/50.0=0.02, cpqsi=0.02/5e8=4e-11
        assert result.loc[dates[0], "B"] == pytest.approx(4e-11)

    def test_larger_spread_larger_factor(self, factor):
        """价差越大，因子值越大。"""
        dates = pd.bdate_range("2025-01-01", periods=2)
        closing_ask = pd.DataFrame({"A": [10.05, 10.20]}, index=dates)
        closing_bid = pd.DataFrame({"A": [9.95, 9.80]}, index=dates)
        dollar_volume = pd.DataFrame({"A": [1e8, 1e8]}, index=dates)

        result = factor.compute(
            closing_ask=closing_ask, closing_bid=closing_bid, dollar_volume=dollar_volume,
        )

        assert result.iloc[1, 0] > result.iloc[0, 0]


class TestCPQSIEdgeCases:
    def test_nan_propagation(self, factor):
        """输入含 NaN 时，输出对应位置也应为 NaN。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        closing_ask = pd.DataFrame({"A": [10.1, np.nan, 10.1]}, index=dates)
        closing_bid = pd.DataFrame({"A": [9.9, 9.9, np.nan]}, index=dates)
        dollar_volume = pd.DataFrame({"A": [1e8, 1e8, 1e8]}, index=dates)

        result = factor.compute(
            closing_ask=closing_ask, closing_bid=closing_bid, dollar_volume=dollar_volume,
        )

        assert not np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert np.isnan(result.iloc[2, 0])


class TestCPQSIOutputShape:
    def test_output_is_dataframe(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=5)
        closing_ask = pd.DataFrame({"A": np.random.uniform(10, 20, 5)}, index=dates)
        closing_bid = closing_ask - 0.1
        dollar_volume = pd.DataFrame({"A": [1e8] * 5}, index=dates)

        result = factor.compute(
            closing_ask=closing_ask, closing_bid=closing_bid, dollar_volume=dollar_volume,
        )
        assert isinstance(result, pd.DataFrame)

    def test_output_shape_matches_input(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=10)
        stocks = ["A", "B", "C"]
        closing_ask = pd.DataFrame(
            np.random.uniform(10, 20, (10, 3)), index=dates, columns=stocks
        )
        closing_bid = closing_ask - 0.1
        dollar_volume = pd.DataFrame(1e8, index=dates, columns=stocks)

        result = factor.compute(
            closing_ask=closing_ask, closing_bid=closing_bid, dollar_volume=dollar_volume,
        )
        assert result.shape == (10, 3)
        assert list(result.columns) == stocks
