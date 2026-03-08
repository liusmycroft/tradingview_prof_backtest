import numpy as np
import pandas as pd
import pytest

from factors.tcc import TCCFactor


@pytest.fixture
def factor():
    return TCCFactor()


class TestTCCMetadata:
    def test_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "TCC"
        assert meta["category"] == "网络结构"
        assert "中心度" in meta["description"]

    def test_repr(self, factor):
        r = repr(factor)
        assert "TCCFactor" in r
        assert "TCC" in r


class TestTCCCompute:
    def test_known_values(self, factor):
        """用已知数据验证 TCC 计算。"""
        np.random.seed(42)
        dates = pd.bdate_range("2025-01-01", periods=20)
        returns = pd.DataFrame(
            np.random.randn(20, 3) * 0.02,
            index=dates,
            columns=["A", "B", "C"],
        )

        result = factor.compute(returns=returns, T=20)

        # 手动计算最后一行
        r_hat_m = returns.mean(axis=1)
        sigma_m = returns.std(axis=1, ddof=1)
        z = returns.sub(r_hat_m, axis=0).div(sigma_m, axis=0)
        z_sq = z ** 2
        z_bar_sq = z_sq.mean(axis=0)  # T=20, 全部 20 行的均值
        expected_tcc = 1.0 / z_bar_sq

        for col in ["A", "B", "C"]:
            np.testing.assert_almost_equal(
                result.loc[dates[-1], col], expected_tcc[col], decimal=10
            )

    def test_identical_returns_high_tcc(self, factor):
        """所有股票收益率相同时，z-score 不确定（std=0），TCC 应为 NaN/Inf。"""
        dates = pd.bdate_range("2025-01-01", periods=20)
        returns = pd.DataFrame(
            np.tile([0.01], (20, 3)),
            index=dates,
            columns=["A", "B", "C"],
        )

        result = factor.compute(returns=returns, T=20)

        # 截面标准差为 0，z-score 为 NaN -> TCC 为 NaN
        assert result.iloc[-1].isna().all() or np.isinf(result.iloc[-1]).all()

    def test_min_periods(self, factor):
        """窗口不足 T 天时，结果应为 NaN。"""
        dates = pd.bdate_range("2025-01-01", periods=10)
        returns = pd.DataFrame(
            np.random.randn(10, 3) * 0.02,
            index=dates,
            columns=["A", "B", "C"],
        )

        result = factor.compute(returns=returns, T=20)

        assert result.isna().all().all()

    def test_output_shape(self, factor):
        """输出形状应与输入一致。"""
        dates = pd.bdate_range("2025-01-01", periods=25)
        returns = pd.DataFrame(
            np.random.randn(25, 5) * 0.02,
            index=dates,
            columns=["A", "B", "C", "D", "E"],
        )

        result = factor.compute(returns=returns, T=20)

        assert result.shape == (25, 5)

    def test_positive_values(self, factor):
        """TCC = 1/z_bar^2 应为正值（当 z_bar > 0 时）。"""
        np.random.seed(123)
        dates = pd.bdate_range("2025-01-01", periods=25)
        returns = pd.DataFrame(
            np.random.randn(25, 5) * 0.02,
            index=dates,
            columns=["A", "B", "C", "D", "E"],
        )

        result = factor.compute(returns=returns, T=20)

        valid = result.dropna()
        assert (valid > 0).all().all()
