import numpy as np
import pandas as pd
import pytest

from factors.residual_flow_strength import ResidualFlowStrengthFactor


@pytest.fixture
def factor():
    return ResidualFlowStrengthFactor()


class TestResidualFlowStrengthMetadata:
    def test_name(self, factor):
        assert factor.name == "RESIDUAL_FLOW_STRENGTH"

    def test_category(self, factor):
        assert factor.category == "高频因子-资金流类"

    def test_repr(self, factor):
        assert "RESIDUAL_FLOW_STRENGTH" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "RESIDUAL_FLOW_STRENGTH"


class TestResidualFlowStrengthCompute:
    def test_basic_cross_section(self, factor):
        """横截面回归后残差均值应接近 0。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A", "B", "C", "D", "E"]
        np.random.seed(42)
        buy = pd.DataFrame(np.random.uniform(100, 200, (25, 5)), index=dates, columns=stocks)
        sell = pd.DataFrame(np.random.uniform(80, 180, (25, 5)), index=dates, columns=stocks)
        ret20 = pd.DataFrame(np.random.randn(25, 5) * 0.05, index=dates, columns=stocks)

        result = factor.compute(buy_amount=buy, sell_amount=sell, ret20=ret20, T=20)
        # 残差的横截面均值应接近 0
        last_row = result.iloc[-1].dropna()
        if len(last_row) > 0:
            assert abs(last_row.mean()) < 0.5

    def test_leading_nan(self, factor):
        """前 T-1 行应为 NaN（滚动窗口不足）。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A", "B", "C"]
        buy = pd.DataFrame(np.random.uniform(100, 200, (25, 3)), index=dates, columns=stocks)
        sell = pd.DataFrame(np.random.uniform(80, 180, (25, 3)), index=dates, columns=stocks)
        ret20 = pd.DataFrame(np.random.randn(25, 3) * 0.05, index=dates, columns=stocks)

        result = factor.compute(buy_amount=buy, sell_amount=sell, ret20=ret20, T=20)
        assert result.iloc[:19].isna().all().all()

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A", "B", "C"]
        buy = pd.DataFrame(np.random.uniform(100, 200, (25, 3)), index=dates, columns=stocks)
        sell = pd.DataFrame(np.random.uniform(80, 180, (25, 3)), index=dates, columns=stocks)
        ret20 = pd.DataFrame(np.random.randn(25, 3) * 0.05, index=dates, columns=stocks)

        result = factor.compute(buy_amount=buy, sell_amount=sell, ret20=ret20, T=20)
        assert result.shape == buy.shape
        assert isinstance(result, pd.DataFrame)
