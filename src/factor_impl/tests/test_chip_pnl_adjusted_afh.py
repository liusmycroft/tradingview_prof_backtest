import numpy as np
import pandas as pd
import pytest

from factors.chip_pnl_adjusted_afh import ChipPnlAdjustedAFHFactor


@pytest.fixture
def factor():
    return ChipPnlAdjustedAFHFactor()


class TestChipPnlAdjustedAFHMetadata:
    def test_name(self, factor):
        assert factor.name == "CHIP_PNL_ADJUSTED_AFH"

    def test_category(self, factor):
        assert factor.category == "行为金融因子-筹码分布"

    def test_repr(self, factor):
        assert "CHIP_PNL_ADJUSTED_AFH" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "CHIP_PNL_ADJUSTED_AFH"


class TestChipPnlAdjustedAFHCompute:
    def test_basic(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        trade_price_vol = pd.DataFrame(500.0, index=dates, columns=stocks)
        total_vol = pd.DataFrame(1000.0, index=dates, columns=stocks)
        close = pd.DataFrame(10.0, index=dates, columns=stocks)

        result = factor.compute(
            close_price=close,
            trade_price_vol_afh=trade_price_vol,
            total_volume_20d=total_vol,
        )
        np.testing.assert_array_almost_equal(result.values, 0.5)

    def test_zero_volume_nan(self, factor):
        """总成交量为 0 时应返回 NaN。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        trade_price_vol = pd.DataFrame(500.0, index=dates, columns=stocks)
        total_vol = pd.DataFrame(0.0, index=dates, columns=stocks)
        close = pd.DataFrame(10.0, index=dates, columns=stocks)

        result = factor.compute(
            close_price=close,
            trade_price_vol_afh=trade_price_vol,
            total_volume_20d=total_vol,
        )
        assert result.isna().all().all()

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        trade_price_vol = pd.DataFrame(np.random.rand(10, 2) * 1000, index=dates, columns=stocks)
        total_vol = pd.DataFrame(np.random.rand(10, 2) * 2000 + 100, index=dates, columns=stocks)
        close = pd.DataFrame(np.random.rand(10, 2) * 20 + 5, index=dates, columns=stocks)

        result = factor.compute(
            close_price=close,
            trade_price_vol_afh=trade_price_vol,
            total_volume_20d=total_vol,
        )
        assert result.shape == trade_price_vol.shape
        assert isinstance(result, pd.DataFrame)
