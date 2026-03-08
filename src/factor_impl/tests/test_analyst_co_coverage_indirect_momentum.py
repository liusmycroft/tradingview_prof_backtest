import numpy as np
import pandas as pd
import pytest

from factors.analyst_co_coverage_indirect_momentum import AnalystCoCoverageIndirectMomentumFactor


@pytest.fixture
def factor():
    return AnalystCoCoverageIndirectMomentumFactor()


class TestAnalystCoCoverageIndirectMomentumMetadata:
    def test_name(self, factor):
        assert factor.name == "ANALYST_CO_COVERAGE_INDIRECT_MOMENTUM"

    def test_category(self, factor):
        assert factor.category == "图谱网络"

    def test_repr(self, factor):
        assert "ANALYST_CO_COVERAGE_INDIRECT_MOMENTUM" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "ANALYST_CO_COVERAGE_INDIRECT_MOMENTUM"
        assert meta["category"] == "图谱网络"


class TestAnalystCoCoverageIndirectMomentumCompute:
    def test_residual_zero_when_perfect_fit(self, factor):
        """当 indirect_strength 完全由 peer_returns 线性解释时，残差应为0。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        x = pd.DataFrame(np.linspace(1, 20, 20), index=dates, columns=stocks)
        y = 2.0 * x + 3.0  # 完美线性关系

        result = factor.compute(indirect_strength=y, peer_returns=x)
        np.testing.assert_array_almost_equal(result["A"].values, 0.0, decimal=10)

    def test_residual_nonzero_with_noise(self, factor):
        """有噪声时残差不为0。"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        stocks = ["A"]
        x = pd.DataFrame(np.random.randn(50), index=dates, columns=stocks)
        y = 2.0 * x + 3.0 + pd.DataFrame(
            np.random.randn(50) * 0.5, index=dates, columns=stocks
        )

        result = factor.compute(indirect_strength=y, peer_returns=x)
        assert not np.allclose(result["A"].values, 0.0)

    def test_residual_mean_near_zero(self, factor):
        """OLS残差均值应接近0。"""
        np.random.seed(123)
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        stocks = ["A"]
        x = pd.DataFrame(np.random.randn(100), index=dates, columns=stocks)
        y = 1.5 * x + 0.5 + pd.DataFrame(
            np.random.randn(100) * 0.3, index=dates, columns=stocks
        )

        result = factor.compute(indirect_strength=y, peer_returns=x)
        assert abs(result["A"].mean()) < 1e-10

    def test_two_stocks_independent(self, factor):
        """两只股票应独立回归。"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        x = pd.DataFrame(np.random.randn(30, 2), index=dates, columns=stocks)
        y = pd.DataFrame(np.random.randn(30, 2), index=dates, columns=stocks)

        result = factor.compute(indirect_strength=y, peer_returns=x)
        assert result.shape == (30, 2)
        assert result.notna().all().all()

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A", "B", "C"]
        x = pd.DataFrame(np.random.randn(20, 3), index=dates, columns=stocks)
        y = pd.DataFrame(np.random.randn(20, 3), index=dates, columns=stocks)

        result = factor.compute(indirect_strength=y, peer_returns=x)
        assert result.shape == (20, 3)


class TestAnalystCoCoverageIndirectMomentumEdgeCases:
    def test_all_nan_column(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        x = pd.DataFrame(np.nan, index=dates, columns=stocks)
        y = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(indirect_strength=y, peer_returns=x)
        assert result.isna().all().all()
