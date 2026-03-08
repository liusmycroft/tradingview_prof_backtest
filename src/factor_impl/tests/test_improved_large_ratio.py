"""改进大单交易占比因子测试"""

import numpy as np
import pandas as pd
import pytest
from factors.improved_large_ratio import ImprovedLargeRatioFactor


@pytest.fixture
def factor():
    return ImprovedLargeRatioFactor()


@pytest.fixture
def sample_data():
    dates = pd.date_range("2024-01-01", periods=25)
    stocks = ["000001", "000002"]
    lb_nls = pd.DataFrame(np.full((25, 2), 100.0), index=dates, columns=stocks)
    nlb_ls = pd.DataFrame(np.full((25, 2), 200.0), index=dates, columns=stocks)
    lb_ls = pd.DataFrame(np.full((25, 2), 500.0), index=dates, columns=stocks)
    total_volume = pd.DataFrame(np.full((25, 2), 1000.0), index=dates, columns=stocks)
    return lb_nls, nlb_ls, lb_ls, total_volume


class TestMetadata:
    def test_name(self, factor):
        assert factor.name == "IMPROVED_LARGE_RATIO"

    def test_category(self, factor):
        assert factor.category == "高频资金流"

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "IMPROVED_LARGE_RATIO"

    def test_repr(self, factor):
        assert "ImprovedLargeRatioFactor" in repr(factor)


class TestCompute:
    def test_known_values(self, factor, sample_data):
        lb_nls, nlb_ls, lb_ls, total_volume = sample_data
        result = factor.compute(
            lb_nls=lb_nls, nlb_ls=nlb_ls, lb_ls=lb_ls, total_volume=total_volume
        )
        # (-100 - 200 + 500) / 1000 = 200/1000 = 0.2
        assert pytest.approx(result.iloc[-1, 0], rel=1e-6) == 0.2

    def test_output_shape(self, factor, sample_data):
        lb_nls, nlb_ls, lb_ls, total_volume = sample_data
        result = factor.compute(
            lb_nls=lb_nls, nlb_ls=nlb_ls, lb_ls=lb_ls, total_volume=total_volume
        )
        assert result.shape == lb_nls.shape

    def test_negative_result(self, factor):
        """当大买大卖不足以抵消交叉交易时，结果为负"""
        dates = pd.date_range("2024-01-01", periods=5)
        stocks = ["A"]
        lb_nls = pd.DataFrame(np.full((5, 1), 300.0), index=dates, columns=stocks)
        nlb_ls = pd.DataFrame(np.full((5, 1), 300.0), index=dates, columns=stocks)
        lb_ls = pd.DataFrame(np.full((5, 1), 100.0), index=dates, columns=stocks)
        total = pd.DataFrame(np.full((5, 1), 1000.0), index=dates, columns=stocks)
        result = factor.compute(lb_nls=lb_nls, nlb_ls=nlb_ls, lb_ls=lb_ls, total_volume=total)
        # (-300 - 300 + 100) / 1000 = -0.5
        assert pytest.approx(result.iloc[-1, 0], rel=1e-6) == -0.5

    def test_custom_window(self, factor):
        dates = pd.date_range("2024-01-01", periods=5)
        stocks = ["A"]
        lb_nls = pd.DataFrame([[100], [200], [300], [400], [500]], index=dates, columns=stocks, dtype=float)
        nlb_ls = pd.DataFrame(np.zeros((5, 1)), index=dates, columns=stocks)
        lb_ls = pd.DataFrame(np.zeros((5, 1)), index=dates, columns=stocks)
        total = pd.DataFrame(np.full((5, 1), 1000.0), index=dates, columns=stocks)
        result = factor.compute(lb_nls=lb_nls, nlb_ls=nlb_ls, lb_ls=lb_ls, total_volume=total, T=3)
        # Last window: mean(-300/1000, -400/1000, -500/1000) = mean(-0.3, -0.4, -0.5) = -0.4
        assert pytest.approx(result.iloc[-1, 0], rel=1e-6) == -0.4

    def test_zero_volume_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=3)
        stocks = ["A"]
        zeros = pd.DataFrame(np.zeros((3, 1)), index=dates, columns=stocks)
        result = factor.compute(lb_nls=zeros, nlb_ls=zeros, lb_ls=zeros, total_volume=zeros)
        assert result.isna().all().all() or (result == 0).all().all() or np.isinf(result.iloc[0, 0])
