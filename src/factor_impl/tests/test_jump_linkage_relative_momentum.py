import numpy as np
import pandas as pd
import pytest

from factors.jump_linkage_relative_momentum import JumpLinkageRelativeMomentumFactor


@pytest.fixture
def factor():
    return JumpLinkageRelativeMomentumFactor()


class TestJumpLinkageRelativeMomentumMetadata:
    def test_name(self, factor):
        assert factor.name == "JUMP_LINKAGE_RELATIVE_MOMENTUM"

    def test_category(self, factor):
        assert factor.category == "图谱网络"

    def test_repr(self, factor):
        assert "JUMP_LINKAGE_RELATIVE_MOMENTUM" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "JUMP_LINKAGE_RELATIVE_MOMENTUM"
        assert meta["category"] == "图谱网络"


class TestJumpLinkageRelativeMomentumCompute:
    def test_residual_zero_when_perfect_fit(self, factor):
        """完美线性关系时残差为0。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        x = pd.DataFrame(np.linspace(1, 20, 20), index=dates, columns=stocks)
        y = 3.0 * x + 1.0

        result = factor.compute(peer_ret=y, own_ret=x)
        np.testing.assert_array_almost_equal(result["A"].values, 0.0, decimal=10)

    def test_residual_mean_near_zero(self, factor):
        """OLS残差均值应接近0。"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        stocks = ["A"]
        x = pd.DataFrame(np.random.randn(100), index=dates, columns=stocks)
        y = 2.0 * x + 1.0 + pd.DataFrame(
            np.random.randn(100) * 0.5, index=dates, columns=stocks
        )

        result = factor.compute(peer_ret=y, own_ret=x)
        assert abs(result["A"].mean()) < 1e-10

    def test_two_stocks_independent(self, factor):
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        x = pd.DataFrame(np.random.randn(30, 2), index=dates, columns=stocks)
        y = pd.DataFrame(np.random.randn(30, 2), index=dates, columns=stocks)

        result = factor.compute(peer_ret=y, own_ret=x)
        assert result.shape == (30, 2)
        assert result.notna().all().all()

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A", "B", "C"]
        x = pd.DataFrame(np.random.randn(20, 3), index=dates, columns=stocks)
        y = pd.DataFrame(np.random.randn(20, 3), index=dates, columns=stocks)

        result = factor.compute(peer_ret=y, own_ret=x)
        assert result.shape == (20, 3)


class TestJumpLinkageRelativeMomentumEdgeCases:
    def test_constant_x(self, factor):
        """x为常数时，beta无法估计，残差应为y-mean(y)。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        x = pd.DataFrame(5.0, index=dates, columns=stocks)
        y = pd.DataFrame(np.arange(10, dtype=float), index=dates, columns=stocks)

        result = factor.compute(peer_ret=y, own_ret=x)
        expected = y["A"] - y["A"].mean()
        np.testing.assert_array_almost_equal(result["A"].values, expected.values)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        x = pd.DataFrame(np.nan, index=dates, columns=stocks)
        y = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(peer_ret=y, own_ret=x)
        assert result.isna().all().all()
