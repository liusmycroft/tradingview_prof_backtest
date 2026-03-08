import numpy as np
import pandas as pd
import pytest

from factors.price_impact_bias import PriceImpactBiasFactor


@pytest.fixture
def factor():
    return PriceImpactBiasFactor()


class TestPriceImpactBiasMetadata:
    def test_name(self, factor):
        assert factor.name == "PRICE_IMPACT_BIAS"

    def test_category(self, factor):
        assert factor.category == "高频因子-资金流类"

    def test_repr(self, factor):
        assert "PRICE_IMPACT_BIAS" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "PRICE_IMPACT_BIAS"


class TestPriceImpactBiasCompute:
    def test_symmetric_impact(self, factor):
        """上下冲击对称时，bias 应接近 0。"""
        np.random.seed(42)
        idx = pd.RangeIndex(100)
        stocks = ["A"]
        mf = pd.DataFrame(np.random.uniform(-1, 1, (100, 1)), index=idx, columns=stocks)
        # ret = 0.5 * |mf| (对称)
        ret = pd.DataFrame(0.5 * mf.values, index=idx, columns=stocks)

        result = factor.compute(bar_return=ret, bar_money_flow_ratio=mf)
        # gamma_up ~= gamma_down ~= 0.5, bias ~= 0
        assert abs(result.iloc[0, 0]) < 0.5

    def test_output_is_dataframe(self, factor):
        idx = pd.RangeIndex(50)
        stocks = ["A", "B"]
        mf = pd.DataFrame(np.random.uniform(-1, 1, (50, 2)), index=idx, columns=stocks)
        ret = pd.DataFrame(np.random.randn(50, 2) * 0.01, index=idx, columns=stocks)

        result = factor.compute(bar_return=ret, bar_money_flow_ratio=mf)
        assert isinstance(result, pd.DataFrame)

    def test_all_positive_mf(self, factor):
        """所有 MF > 0 时，gamma_down 无数据，回归可能退化。"""
        idx = pd.RangeIndex(50)
        stocks = ["A"]
        mf = pd.DataFrame(np.random.uniform(0.1, 1, (50, 1)), index=idx, columns=stocks)
        ret = pd.DataFrame(np.random.randn(50, 1) * 0.01, index=idx, columns=stocks)

        result = factor.compute(bar_return=ret, bar_money_flow_ratio=mf)
        assert isinstance(result, pd.DataFrame)
