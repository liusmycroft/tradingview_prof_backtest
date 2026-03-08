"""峰岭成交比因子测试"""

import numpy as np
import pandas as pd
import pytest
from factors.peak_ridge_ratio import PeakRidgeRatioFactor


@pytest.fixture
def factor():
    return PeakRidgeRatioFactor()


@pytest.fixture
def sample_data():
    dates = pd.date_range("2024-01-01", periods=25)
    stocks = ["000001", "000002"]
    peak_volume = pd.DataFrame(
        np.full((25, 2), 3000.0), index=dates, columns=stocks
    )
    valley_volume = pd.DataFrame(
        np.full((25, 2), 1000.0), index=dates, columns=stocks
    )
    return peak_volume, valley_volume


class TestMetadata:
    def test_name(self, factor):
        assert factor.name == "PEAK_RIDGE_RATIO"

    def test_category(self, factor):
        assert factor.category == "高频成交分布"

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "PEAK_RIDGE_RATIO"

    def test_repr(self, factor):
        assert "PeakRidgeRatioFactor" in repr(factor)


class TestCompute:
    def test_known_values(self, factor, sample_data):
        peak_volume, valley_volume = sample_data
        result = factor.compute(peak_volume=peak_volume, valley_volume=valley_volume)
        # 3000/1000 = 3.0
        assert pytest.approx(result.iloc[-1, 0], rel=1e-6) == 3.0

    def test_output_shape(self, factor, sample_data):
        peak_volume, valley_volume = sample_data
        result = factor.compute(peak_volume=peak_volume, valley_volume=valley_volume)
        assert result.shape == peak_volume.shape

    def test_varying_volumes(self, factor):
        dates = pd.date_range("2024-01-01", periods=5)
        stocks = ["A"]
        peak = pd.DataFrame([[100], [200], [300], [400], [500]], index=dates, columns=stocks, dtype=float)
        valley = pd.DataFrame([[50], [100], [150], [200], [250]], index=dates, columns=stocks, dtype=float)
        result = factor.compute(peak_volume=peak, valley_volume=valley, T=5)
        # sum(peak) / sum(valley) = 1500 / 750 = 2.0
        assert pytest.approx(result.iloc[-1, 0], rel=1e-6) == 2.0

    def test_custom_window(self, factor):
        dates = pd.date_range("2024-01-01", periods=5)
        stocks = ["A"]
        peak = pd.DataFrame([[100], [200], [300], [400], [500]], index=dates, columns=stocks, dtype=float)
        valley = pd.DataFrame(np.full((5, 1), 100.0), index=dates, columns=stocks)
        result = factor.compute(peak_volume=peak, valley_volume=valley, T=3)
        # Last window: sum(300,400,500)/sum(100,100,100) = 1200/300 = 4.0
        assert pytest.approx(result.iloc[-1, 0], rel=1e-6) == 4.0

    def test_equal_volumes(self, factor):
        dates = pd.date_range("2024-01-01", periods=5)
        stocks = ["A"]
        vol = pd.DataFrame(np.full((5, 1), 500.0), index=dates, columns=stocks)
        result = factor.compute(peak_volume=vol, valley_volume=vol)
        assert (result == 1.0).all().all()

    def test_zero_valley_inf(self, factor):
        dates = pd.date_range("2024-01-01", periods=3)
        stocks = ["A"]
        peak = pd.DataFrame(np.full((3, 1), 100.0), index=dates, columns=stocks)
        valley = pd.DataFrame(np.zeros((3, 1)), index=dates, columns=stocks)
        result = factor.compute(peak_volume=peak, valley_volume=valley)
        assert np.isinf(result.iloc[-1, 0])
