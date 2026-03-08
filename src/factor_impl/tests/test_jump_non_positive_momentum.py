import numpy as np
import pandas as pd
import pytest

from factors.jump_non_positive_momentum import JumpNonPositiveMomentumFactor


@pytest.fixture
def factor():
    return JumpNonPositiveMomentumFactor()


class TestJumpNonPositiveMomentumMetadata:
    def test_name(self, factor):
        assert factor.name == "JUMP_NON_POSITIVE_MOMENTUM"

    def test_category(self, factor):
        assert factor.category == "图谱网络-动量溢出"

    def test_repr(self, factor):
        assert "JUMP_NON_POSITIVE_MOMENTUM" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "JUMP_NON_POSITIVE_MOMENTUM"


class TestJumpNonPositiveMomentumCompute:
    def test_basic(self, factor):
        stocks = ["A", "B", "C", "D"]
        corr = pd.DataFrame(
            [[0, 0.8, 0.6, 0.3],
             [0.8, 0, 0.5, 0.2],
             [0.6, 0.5, 0, 0.7],
             [0.3, 0.2, 0.7, 0]],
            index=stocks, columns=stocks, dtype=float,
        )
        neg_ret = pd.Series([-0.02, -0.01, -0.03, -0.015], index=stocks)
        no_ret = pd.Series([0.01, 0.02, -0.01, 0.005], index=stocks)
        ret20 = pd.Series([0.05, 0.03, -0.02, 0.01], index=stocks)

        result = factor.compute(
            jump_corr=corr, neg_jump_ret=neg_ret,
            no_jump_ret=no_ret, ret_20d=ret20, threshold=0.3,
        )
        assert result.shape == (4, 1)
        assert isinstance(result, pd.DataFrame)

    def test_output_shape(self, factor):
        N = 5
        stocks = [f"S{i}" for i in range(N)]
        corr = pd.DataFrame(np.random.rand(N, N), index=stocks, columns=stocks)
        np.fill_diagonal(corr.values, 0)
        neg_ret = pd.Series(np.random.randn(N) * 0.01, index=stocks)
        no_ret = pd.Series(np.random.randn(N) * 0.01, index=stocks)
        ret20 = pd.Series(np.random.randn(N) * 0.05, index=stocks)

        result = factor.compute(
            jump_corr=corr, neg_jump_ret=neg_ret,
            no_jump_ret=no_ret, ret_20d=ret20,
        )
        assert result.shape == (N, 1)
