import numpy as np
import pandas as pd
import pytest

from factors.improved_analyst_coverage_momentum import ImprovedAnalystCoverageMomentumFactor


@pytest.fixture
def factor():
    return ImprovedAnalystCoverageMomentumFactor()


class TestImprovedAnalystCoverageMomentumMetadata:
    def test_name(self, factor):
        assert factor.name == "IMPROVED_ANALYST_COVERAGE_MOMENTUM"

    def test_category(self, factor):
        assert factor.category == "图谱网络-动量溢出"

    def test_repr(self, factor):
        assert "IMPROVED_ANALYST_COVERAGE_MOMENTUM" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "IMPROVED_ANALYST_COVERAGE_MOMENTUM"


class TestImprovedAnalystCoverageMomentumCompute:
    def test_basic(self, factor):
        """基本计算验证。"""
        stocks = ["A", "B", "C"]
        direct = pd.DataFrame(
            [[0, 2, 1], [2, 0, 3], [1, 3, 0]],
            index=stocks, columns=stocks, dtype=float,
        )
        ret = pd.Series([0.05, 0.10, -0.02], index=stocks)
        num_direct = pd.Series([2, 2, 2], index=stocks)

        result = factor.compute(direct_coverage=direct, ret_20d=ret, num_direct=num_direct)
        assert result.shape == (3, 1)
        assert result.notna().all().all()

    def test_isolated_stock(self, factor):
        """孤立股票（无关联）应为 NaN。"""
        stocks = ["A", "B", "C"]
        direct = pd.DataFrame(
            [[0, 0, 0], [0, 0, 3], [0, 3, 0]],
            index=stocks, columns=stocks, dtype=float,
        )
        ret = pd.Series([0.05, 0.10, -0.02], index=stocks)
        num_direct = pd.Series([0, 1, 1], index=stocks)

        result = factor.compute(direct_coverage=direct, ret_20d=ret, num_direct=num_direct)
        assert np.isnan(result.loc["A", "factor"])

    def test_output_is_dataframe(self, factor):
        stocks = ["A", "B"]
        direct = pd.DataFrame([[0, 1], [1, 0]], index=stocks, columns=stocks, dtype=float)
        ret = pd.Series([0.05, 0.10], index=stocks)
        num_direct = pd.Series([1, 1], index=stocks)

        result = factor.compute(direct_coverage=direct, ret_20d=ret, num_direct=num_direct)
        assert isinstance(result, pd.DataFrame)
