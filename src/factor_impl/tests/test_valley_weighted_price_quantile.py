import numpy as np
import pandas as pd
import pytest

from factors.valley_weighted_price_quantile import ValleyWeightedPriceQuantileFactor


@pytest.fixture
def factor():
    return ValleyWeightedPriceQuantileFactor()


class TestValleyWeightedPriceQuantileMetadata:
    def test_name(self, factor):
        assert factor.name == "VALLEY_WEIGHTED_PRICE_QUANTILE"

    def test_category(self, factor):
        assert factor.category == "高频因子-成交分布类"

    def test_repr(self, factor):
        assert "VALLEY_WEIGHTED_PRICE_QUANTILE" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "VALLEY_WEIGHTED_PRICE_QUANTILE"


class TestValleyWeightedPriceQuantileCompute:
    def test_midpoint_vwap(self, factor):
        """VWAP 在区间中点时，分位点应为 0.5。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        valley_vwap = pd.DataFrame(10.0, index=dates, columns=stocks)
        high_price = pd.DataFrame(12.0, index=dates, columns=stocks)
        low_price = pd.DataFrame(8.0, index=dates, columns=stocks)
        prev_close = pd.DataFrame(10.0, index=dates, columns=stocks)

        result = factor.compute(
            valley_vwap=valley_vwap, high_price=high_price,
            low_price=low_price, prev_close=prev_close, T=20,
        )
        np.testing.assert_array_almost_equal(result["A"].values, 0.5)

    def test_vwap_at_low(self, factor):
        """VWAP 等于区间最低价时，分位点应为 0。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        valley_vwap = pd.DataFrame(8.0, index=dates, columns=stocks)
        high_price = pd.DataFrame(12.0, index=dates, columns=stocks)
        low_price = pd.DataFrame(8.0, index=dates, columns=stocks)
        prev_close = pd.DataFrame(10.0, index=dates, columns=stocks)

        result = factor.compute(
            valley_vwap=valley_vwap, high_price=high_price,
            low_price=low_price, prev_close=prev_close, T=20,
        )
        np.testing.assert_array_almost_equal(result["A"].values, 0.0)

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        valley_vwap = pd.DataFrame(np.random.uniform(9, 11, (30, 2)), index=dates, columns=stocks)
        high_price = pd.DataFrame(12.0, index=dates, columns=stocks)
        low_price = pd.DataFrame(8.0, index=dates, columns=stocks)
        prev_close = pd.DataFrame(10.0, index=dates, columns=stocks)

        result = factor.compute(
            valley_vwap=valley_vwap, high_price=high_price,
            low_price=low_price, prev_close=prev_close, T=20,
        )
        assert result.shape == valley_vwap.shape
        assert isinstance(result, pd.DataFrame)
