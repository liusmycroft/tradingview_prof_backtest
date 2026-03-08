import numpy as np
import pandas as pd
import pytest

from factors.uniform_active_ratio import UniformActiveRatioFactor


@pytest.fixture
def factor():
    return UniformActiveRatioFactor()


class TestUniformActiveRatioMetadata:
    def test_name(self, factor):
        assert factor.name == "UNIFORM_ACTIVE_RATIO"

    def test_category(self, factor):
        assert factor.category == "高频因子-资金流类"

    def test_repr(self, factor):
        assert "UNIFORM_ACTIVE_RATIO" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "UNIFORM_ACTIVE_RATIO"


class TestUniformActiveRatioCompute:
    def test_zero_return(self, factor):
        """收益率为 0 时，主动买入占比 = (0+0.1)/0.2 = 0.5。"""
        idx = pd.RangeIndex(10)
        stocks = ["A"]
        amount = pd.DataFrame(100.0, index=idx, columns=stocks)
        ret = pd.DataFrame(0.0, index=idx, columns=stocks)

        result = factor.compute(minute_amount=amount, minute_return=ret)
        assert result.iloc[0, 0] == pytest.approx(0.5, rel=1e-6)

    def test_max_return(self, factor):
        """收益率为 0.1 时，主动买入占比 = 1.0。"""
        idx = pd.RangeIndex(10)
        stocks = ["A"]
        amount = pd.DataFrame(100.0, index=idx, columns=stocks)
        ret = pd.DataFrame(0.1, index=idx, columns=stocks)

        result = factor.compute(minute_amount=amount, minute_return=ret)
        assert result.iloc[0, 0] == pytest.approx(1.0, rel=1e-6)

    def test_min_return(self, factor):
        """收益率为 -0.1 时，主动买入占比 = 0.0。"""
        idx = pd.RangeIndex(10)
        stocks = ["A"]
        amount = pd.DataFrame(100.0, index=idx, columns=stocks)
        ret = pd.DataFrame(-0.1, index=idx, columns=stocks)

        result = factor.compute(minute_amount=amount, minute_return=ret)
        assert result.iloc[0, 0] == pytest.approx(0.0, abs=1e-10)

    def test_output_is_dataframe(self, factor):
        idx = pd.RangeIndex(10)
        stocks = ["A", "B"]
        amount = pd.DataFrame(np.random.rand(10, 2) * 100, index=idx, columns=stocks)
        ret = pd.DataFrame(np.random.uniform(-0.1, 0.1, (10, 2)), index=idx, columns=stocks)

        result = factor.compute(minute_amount=amount, minute_return=ret)
        assert isinstance(result, pd.DataFrame)
