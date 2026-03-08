"""谷岭加权价格比因子测试"""

import numpy as np
import pandas as pd
import pytest
from factors.valley_ridge_ratio import ValleyRidgeRatioFactor


@pytest.fixture
def factor():
    return ValleyRidgeRatioFactor()


@pytest.fixture
def sample_data():
    dates = pd.date_range("2024-01-01", periods=25)
    stocks = ["000001", "000002"]
    valley_vwap = pd.DataFrame(
        np.full((25, 2), 9.0), index=dates, columns=stocks
    )
    ridge_vwap = pd.DataFrame(
        np.full((25, 2), 12.0), index=dates, columns=stocks
    )
    return valley_vwap, ridge_vwap


class TestMetadata:
    def test_name(self, factor):
        assert factor.name == "VALLEY_RIDGE_RATIO"

    def test_category(self, factor):
        assert factor.category == "高频量价"

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "VALLEY_RIDGE_RATIO"

    def test_repr(self, factor):
        assert "ValleyRidgeRatioFactor" in repr(factor)


class TestCompute:
    def test_known_values(self, factor, sample_data):
        valley_vwap, ridge_vwap = sample_data
        result = factor.compute(valley_vwap=valley_vwap, ridge_vwap=ridge_vwap)
        # 9/12 = 0.75, rolling mean of constant = 0.75
        assert pytest.approx(result.iloc[-1, 0], rel=1e-6) == 0.75

    def test_output_shape(self, factor, sample_data):
        valley_vwap, ridge_vwap = sample_data
        result = factor.compute(valley_vwap=valley_vwap, ridge_vwap=ridge_vwap)
        assert result.shape == valley_vwap.shape

    def test_custom_window(self, factor):
        dates = pd.date_range("2024-01-01", periods=10)
        stocks = ["A"]
        valley = pd.DataFrame(np.arange(1, 11, dtype=float).reshape(-1, 1), index=dates, columns=stocks)
        ridge = pd.DataFrame(np.full((10, 1), 10.0), index=dates, columns=stocks)
        result = factor.compute(valley_vwap=valley, ridge_vwap=ridge, T=5)
        # Last value: mean of [6/10, 7/10, 8/10, 9/10, 10/10] = mean([0.6,0.7,0.8,0.9,1.0]) = 0.8
        assert pytest.approx(result.iloc[-1, 0], rel=1e-6) == 0.8

    def test_equal_vwap(self, factor):
        dates = pd.date_range("2024-01-01", periods=5)
        stocks = ["A"]
        vwap = pd.DataFrame(np.full((5, 1), 10.0), index=dates, columns=stocks)
        result = factor.compute(valley_vwap=vwap, ridge_vwap=vwap)
        assert (result == 1.0).all().all()

    def test_nan_in_ratio(self, factor):
        """NaN in valley_vwap produces NaN ratio; rolling mean skips NaN"""
        dates = pd.date_range("2024-01-01", periods=5)
        stocks = ["A"]
        valley = pd.DataFrame([[10.0], [np.nan], [10.0], [10.0], [10.0]], index=dates, columns=stocks)
        ridge = pd.DataFrame(np.full((5, 1), 10.0), index=dates, columns=stocks)
        # daily ratio: [1.0, NaN, 1.0, 1.0, 1.0]
        result = factor.compute(valley_vwap=valley, ridge_vwap=ridge, T=3)
        # rolling mean skips NaN, so all valid windows still yield 1.0
        assert pytest.approx(result.iloc[-1, 0], rel=1e-6) == 1.0
        # The raw ratio at index 1 is NaN
        raw_ratio = valley / ridge
        assert np.isnan(raw_ratio.iloc[1, 0])
