import numpy as np
import pandas as pd
import pytest

from factors.time_weighted_rpp import TimeWeightedRelativePricePositionFactor


@pytest.fixture
def factor():
    return TimeWeightedRelativePricePositionFactor()


class TestTimeWeightedRPPMetadata:
    def test_name(self, factor):
        assert factor.name == "TIME_WEIGHTED_RPP"

    def test_category(self, factor):
        assert factor.category == "高频因子-收益分布类"

    def test_repr(self, factor):
        assert "TIME_WEIGHTED_RPP" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "TIME_WEIGHTED_RPP"


class TestTimeWeightedRPPCompute:
    def test_midpoint(self, factor):
        """TWAP 在区间中点时，ARPP = 0.5。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        twap = pd.DataFrame(10.0, index=dates, columns=stocks)
        high = pd.DataFrame(12.0, index=dates, columns=stocks)
        low = pd.DataFrame(8.0, index=dates, columns=stocks)

        result = factor.compute(twap=twap, high_price=high, low_price=low)
        np.testing.assert_array_almost_equal(result["A"].values, 0.5)

    def test_at_high(self, factor):
        """TWAP 等于最高价时，ARPP = 1.0。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        twap = pd.DataFrame(12.0, index=dates, columns=stocks)
        high = pd.DataFrame(12.0, index=dates, columns=stocks)
        low = pd.DataFrame(8.0, index=dates, columns=stocks)

        result = factor.compute(twap=twap, high_price=high, low_price=low)
        np.testing.assert_array_almost_equal(result["A"].values, 1.0)

    def test_at_low(self, factor):
        """TWAP 等于最低价时，ARPP = 0.0。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        twap = pd.DataFrame(8.0, index=dates, columns=stocks)
        high = pd.DataFrame(12.0, index=dates, columns=stocks)
        low = pd.DataFrame(8.0, index=dates, columns=stocks)

        result = factor.compute(twap=twap, high_price=high, low_price=low)
        np.testing.assert_array_almost_equal(result["A"].values, 0.0)

    def test_zero_range_nan(self, factor):
        """最高价等于最低价时应返回 NaN。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        twap = pd.DataFrame(10.0, index=dates, columns=stocks)
        high = pd.DataFrame(10.0, index=dates, columns=stocks)
        low = pd.DataFrame(10.0, index=dates, columns=stocks)

        result = factor.compute(twap=twap, high_price=high, low_price=low)
        assert result.isna().all().all()

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A", "B"]
        twap = pd.DataFrame(np.random.uniform(9, 11, (20, 2)), index=dates, columns=stocks)
        high = pd.DataFrame(12.0, index=dates, columns=stocks)
        low = pd.DataFrame(8.0, index=dates, columns=stocks)

        result = factor.compute(twap=twap, high_price=high, low_price=low)
        assert result.shape == twap.shape
        assert isinstance(result, pd.DataFrame)
