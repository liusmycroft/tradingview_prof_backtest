import numpy as np
import pandas as pd
import pytest

from factors.retail_panic import RetailPanicFactor


@pytest.fixture
def factor():
    return RetailPanicFactor()


class TestRetailPanicMetadata:
    def test_name(self, factor):
        assert factor.name == "RETAIL_PANIC"

    def test_category(self, factor):
        assert factor.category == "高频动量反转"

    def test_repr(self, factor):
        assert "RETAIL_PANIC" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "RETAIL_PANIC"
        assert meta["category"] == "高频动量反转"


class TestRetailPanicHandCalculated:
    def test_zero_market_return(self, factor):
        """市场收益为 0 时, deviation = |r_i|,
        base = |r_i| + 0.1, panic_degree = |r_i| / (|r_i| + 0.1)。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily_return = pd.DataFrame(0.02, index=dates, columns=stocks)
        daily_market_return = pd.Series(0.0, index=dates)
        daily_retail_ratio = pd.DataFrame(1.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_return=daily_return,
            daily_market_return=daily_market_return,
            daily_retail_ratio=daily_retail_ratio,
            T=20,
        )
        # panic_degree = 0.02 / (0.02 + 0.1) = 0.02/0.12 = 1/6
        # weighted_ret = (1/6) * 0.02 * 1.0 = 0.02/6
        # panic_return = 0.02/6 (constant => mean = 0.02/6)
        # panic_vol = 0 (constant => std = 0)
        # factor = 0.5 * 0.02/6 + 0.5 * 0 = 0.01/6
        expected = 0.5 * (0.02 / 6.0)
        assert result.iloc[-1, 0] == pytest.approx(expected, rel=1e-6)

    def test_equal_stock_and_market(self, factor):
        """个股收益 == 市场收益时, deviation = 0, panic_degree = 0,
        weighted_ret = 0, factor = 0。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily_return = pd.DataFrame(0.01, index=dates, columns=stocks)
        daily_market_return = pd.Series(0.01, index=dates)
        daily_retail_ratio = pd.DataFrame(0.3, index=dates, columns=stocks)

        result = factor.compute(
            daily_return=daily_return,
            daily_market_return=daily_market_return,
            daily_retail_ratio=daily_retail_ratio,
            T=20,
        )
        assert result.iloc[-1, 0] == pytest.approx(0.0, abs=1e-10)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        ret = pd.DataFrame({"A": [0.02] * 25, "B": [0.01] * 25}, index=dates)
        mkt = pd.Series(0.01, index=dates)
        ratio = pd.DataFrame({"A": [0.3] * 25, "B": [0.3] * 25}, index=dates)

        result = factor.compute(
            daily_return=ret,
            daily_market_return=mkt,
            daily_retail_ratio=ratio,
            T=20,
        )
        # B: stock == market => factor = 0
        assert result.iloc[-1, 1] == pytest.approx(0.0, abs=1e-10)
        # A: stock != market => factor != 0
        assert not np.isclose(result.iloc[-1, 0], 0.0)


class TestRetailPanicEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        ret = pd.DataFrame(np.random.randn(10) * 0.02, index=dates, columns=stocks)
        ret.iloc[3, 0] = np.nan
        mkt = pd.Series(np.random.randn(10) * 0.01, index=dates)
        ratio = pd.DataFrame(0.3, index=dates, columns=stocks)

        result = factor.compute(
            daily_return=ret,
            daily_market_return=mkt,
            daily_retail_ratio=ratio,
            T=5,
        )
        assert isinstance(result, pd.DataFrame)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        ret = pd.DataFrame(np.nan, index=dates, columns=stocks)
        mkt = pd.Series(np.nan, index=dates)
        ratio = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(
            daily_return=ret,
            daily_market_return=mkt,
            daily_retail_ratio=ratio,
            T=5,
        )
        assert result.isna().all().all()

    def test_zero_returns(self, factor):
        """所有收益为 0 时, factor = 0。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        ret = pd.DataFrame(0.0, index=dates, columns=stocks)
        mkt = pd.Series(0.0, index=dates)
        ratio = pd.DataFrame(0.3, index=dates, columns=stocks)

        result = factor.compute(
            daily_return=ret,
            daily_market_return=mkt,
            daily_retail_ratio=ratio,
            T=20,
        )
        assert result.iloc[-1, 0] == pytest.approx(0.0, abs=1e-10)


class TestRetailPanicOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        np.random.seed(42)
        ret = pd.DataFrame(np.random.randn(30, 3) * 0.02, index=dates, columns=stocks)
        mkt = pd.Series(np.random.randn(30) * 0.01, index=dates)
        ratio = pd.DataFrame(
            np.random.uniform(0.1, 0.5, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(
            daily_return=ret,
            daily_market_return=mkt,
            daily_retail_ratio=ratio,
            T=20,
        )
        assert result.shape == ret.shape
        assert list(result.columns) == list(ret.columns)
        assert list(result.index) == list(ret.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        ret = pd.DataFrame(0.01, index=dates, columns=stocks)
        mkt = pd.Series(0.005, index=dates)
        ratio = pd.DataFrame(0.3, index=dates, columns=stocks)

        result = factor.compute(
            daily_return=ret,
            daily_market_return=mkt,
            daily_retail_ratio=ratio,
            T=20,
        )
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        T = 20
        np.random.seed(99)
        ret = pd.DataFrame(np.random.randn(30, 2) * 0.02, index=dates, columns=stocks)
        mkt = pd.Series(np.random.randn(30) * 0.01, index=dates)
        ratio = pd.DataFrame(0.3, index=dates, columns=stocks)

        result = factor.compute(
            daily_return=ret,
            daily_market_return=mkt,
            daily_retail_ratio=ratio,
            T=T,
        )
        assert result.iloc[: T - 1].isna().all().all()
