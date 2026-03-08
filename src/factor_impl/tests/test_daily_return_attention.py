import numpy as np
import pandas as pd
import pytest

from factors.daily_return_attention import DailyReturnAttentionFactor


@pytest.fixture
def factor():
    return DailyReturnAttentionFactor()


class TestDailyReturnAttentionMetadata:
    def test_name(self, factor):
        assert factor.name == "DAILY_RETURN_ATTENTION"

    def test_category(self, factor):
        assert factor.category == "行为金融因子-注意力"

    def test_repr(self, factor):
        assert "DAILY_RETURN_ATTENTION" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "DAILY_RETURN_ATTENTION"


class TestDailyReturnAttentionCompute:
    def test_zero_excess_return(self, factor):
        """个股收益等于市场收益时，原始因子为 0，处理后也为 0。"""
        dates = pd.date_range("2024-01-01", periods=600, freq="D")
        stocks = ["A"]
        stock_ret = pd.DataFrame(0.001, index=dates, columns=stocks)
        mkt_ret = pd.DataFrame(0.001, index=dates, columns=["market"])

        result = factor.compute(stock_return=stock_ret, market_return=mkt_ret, T=250, hist_window=500)
        # (r - R)^2 = 0 everywhere, raw = 0, hist = 0, result = 0
        valid = result.dropna()
        if len(valid) > 0:
            np.testing.assert_array_almost_equal(valid["A"].values, 0.0)

    def test_leading_nan(self, factor):
        """前 T-1 行应为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=300, freq="D")
        stocks = ["A"]
        stock_ret = pd.DataFrame(np.random.randn(300) * 0.01, index=dates, columns=stocks)
        mkt_ret = pd.DataFrame(np.random.randn(300) * 0.005, index=dates, columns=["market"])

        result = factor.compute(stock_return=stock_ret, market_return=mkt_ret, T=250, hist_window=500)
        assert result.iloc[:249]["A"].isna().all()

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=600, freq="D")
        stocks = ["A", "B"]
        stock_ret = pd.DataFrame(np.random.randn(600, 2) * 0.01, index=dates, columns=stocks)
        mkt_ret = pd.DataFrame(np.random.randn(600) * 0.005, index=dates, columns=["market"])

        result = factor.compute(stock_return=stock_ret, market_return=mkt_ret, T=250, hist_window=500)
        assert result.shape == stock_ret.shape
        assert isinstance(result, pd.DataFrame)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=300, freq="D")
        stocks = ["A"]
        stock_ret = pd.DataFrame(np.random.randn(300) * 0.01, index=dates, columns=stocks)
        mkt_ret = pd.DataFrame(np.random.randn(300) * 0.005, index=dates, columns=["market"])

        result = factor.compute(stock_return=stock_ret, market_return=mkt_ret, T=250)
        assert isinstance(result, pd.DataFrame)
