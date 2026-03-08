import numpy as np
import pandas as pd
import pytest

from factors.realized_jump_volatility import RealizedJumpVolatilityFactor


@pytest.fixture
def factor():
    return RealizedJumpVolatilityFactor()


class TestRealizedJumpVolatilityMetadata:
    def test_name(self, factor):
        assert factor.name == "REALIZED_JUMP_VOLATILITY"

    def test_category(self, factor):
        assert factor.category == "高频波动跳跃"

    def test_repr(self, factor):
        assert "REALIZED_JUMP_VOLATILITY" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "REALIZED_JUMP_VOLATILITY"
        assert meta["category"] == "高频波动跳跃"


class TestRealizedJumpVolatilityCompute:
    def test_rv_greater_than_iv(self, factor):
        """RV > IV 时，RVJ = RV - IV > 0。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        rv = pd.DataFrame([0.01, 0.02, 0.03, 0.04, 0.05], index=dates, columns=stocks)
        iv = pd.DataFrame([0.005, 0.01, 0.015, 0.02, 0.025], index=dates, columns=stocks)

        result = factor.compute(rv=rv, iv_hat=iv)
        expected = rv - iv
        np.testing.assert_array_almost_equal(result.values, expected.values)

    def test_rv_less_than_iv_clipped(self, factor):
        """RV < IV 时，RVJ 应被截断为0。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]
        rv = pd.DataFrame([0.005, 0.01, 0.015], index=dates, columns=stocks)
        iv = pd.DataFrame([0.01, 0.02, 0.03], index=dates, columns=stocks)

        result = factor.compute(rv=rv, iv_hat=iv)
        np.testing.assert_array_almost_equal(result.values, 0.0)

    def test_mixed_positive_negative(self, factor):
        """混合情况。"""
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        stocks = ["A"]
        rv = pd.DataFrame([0.02, 0.005, 0.03, 0.01], index=dates, columns=stocks)
        iv = pd.DataFrame([0.01, 0.01, 0.01, 0.01], index=dates, columns=stocks)

        result = factor.compute(rv=rv, iv_hat=iv)
        expected = [0.01, 0.0, 0.02, 0.0]
        np.testing.assert_array_almost_equal(result["A"].values, expected)

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        rv = pd.DataFrame(np.random.uniform(0, 0.1, (30, 3)), index=dates, columns=stocks)
        iv = pd.DataFrame(np.random.uniform(0, 0.1, (30, 3)), index=dates, columns=stocks)

        result = factor.compute(rv=rv, iv_hat=iv)
        assert result.shape == (30, 3)

    def test_result_non_negative(self, factor):
        """结果应始终非负。"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        stocks = ["A", "B"]
        rv = pd.DataFrame(np.random.uniform(0, 0.1, (50, 2)), index=dates, columns=stocks)
        iv = pd.DataFrame(np.random.uniform(0, 0.1, (50, 2)), index=dates, columns=stocks)

        result = factor.compute(rv=rv, iv_hat=iv)
        assert (result >= 0).all().all()


class TestRealizedJumpVolatilityIVHat:
    def test_compute_iv_hat_basic(self):
        """验证 IV_hat 计算不报错且返回正值。"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        intraday = pd.DataFrame(
            np.random.randn(5, 48) * 0.001,
            index=dates, columns=range(48),
        )

        iv_hat = RealizedJumpVolatilityFactor.compute_iv_hat(intraday, k=3)
        assert len(iv_hat) == 5
        assert (iv_hat > 0).all()

    def test_compute_iv_hat_too_few_columns(self):
        """列数不足k时返回NaN。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        intraday = pd.DataFrame(
            np.random.randn(3, 2) * 0.001,
            index=dates, columns=range(2),
        )

        iv_hat = RealizedJumpVolatilityFactor.compute_iv_hat(intraday, k=3)
        assert iv_hat.isna().all()


class TestRealizedJumpVolatilityEdgeCases:
    def test_zero_rv_and_iv(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        rv = pd.DataFrame(0.0, index=dates, columns=stocks)
        iv = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(rv=rv, iv_hat=iv)
        np.testing.assert_array_almost_equal(result.values, 0.0)
