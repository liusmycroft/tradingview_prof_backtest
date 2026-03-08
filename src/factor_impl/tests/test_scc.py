import numpy as np
import pandas as pd
import pytest

from factors.scc import SCCFactor


@pytest.fixture
def factor():
    return SCCFactor()


class TestSCCMetadata:
    def test_name(self, factor):
        assert factor.name == "SCC"

    def test_category(self, factor):
        assert factor.category == "网络结构"

    def test_repr(self, factor):
        assert "SCC" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "SCC"
        assert meta["category"] == "网络结构"


class TestSCCHandCalculated:
    def test_two_perfectly_correlated_stocks(self, factor):
        """两只完全正相关的股票，corr≈1, d_bar≈0, SCC 极大。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A", "B"]
        data = np.array([[0.01, 0.02],
                         [0.02, 0.04],
                         [-0.01, -0.02],
                         [0.03, 0.06],
                         [0.01, 0.02]])
        returns = pd.DataFrame(data, index=dates, columns=stocks)

        result = factor.compute(returns=returns, T=5)
        # corr ≈ 1.0, d_bar^2 ≈ 0, SCC = 1/d_bar^2 -> 极大值、inf 或 NaN
        val = result.iloc[4, 0]
        assert np.isnan(val) or np.isinf(val) or val > 1e6

    def test_two_uncorrelated_stocks(self, factor):
        """手动构造两只股票，验证 SCC 计算。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A", "B"]
        # 简单数据
        data = np.array([[0.01, 0.03],
                         [0.02, 0.01],
                         [0.03, 0.02]])
        returns = pd.DataFrame(data, index=dates, columns=stocks)

        result = factor.compute(returns=returns, T=3)

        # 手算相关系数
        a = data[:, 0]
        b = data[:, 1]
        a_norm = (a - a.mean()) / a.std(ddof=0)
        b_norm = (b - b.mean()) / b.std(ddof=0)
        corr_ab = np.dot(a_norm, b_norm) / 3.0

        # 对 A: p_bar = corr_ab, SCC = 1 / (2*(1-corr_ab))
        d_bar_sq = 2.0 * (1.0 - corr_ab)
        expected_scc = 1.0 / d_bar_sq

        assert result.iloc[2, 0] == pytest.approx(expected_scc, rel=1e-6)
        # 对称性: B 的 SCC 应相同
        assert result.iloc[2, 1] == pytest.approx(expected_scc, rel=1e-6)

    def test_three_stocks(self, factor):
        """三只股票，验证平均相关系数计算。"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A", "B", "C"]
        data = np.random.normal(0, 0.02, (5, 3))
        returns = pd.DataFrame(data, index=dates, columns=stocks)

        result = factor.compute(returns=returns, T=5)

        # 手算 A 的 SCC
        a, b, c = data[:, 0], data[:, 1], data[:, 2]
        def corr(x, y):
            xn = (x - x.mean()) / x.std(ddof=0)
            yn = (y - y.mean()) / y.std(ddof=0)
            return np.dot(xn, yn) / len(x)

        p_bar_a = (corr(a, b) + corr(a, c)) / 2.0
        expected_scc_a = 1.0 / (2.0 * (1.0 - p_bar_a))

        assert result.iloc[4, 0] == pytest.approx(expected_scc_a, rel=1e-6)


class TestSCCEdgeCases:
    def test_nan_in_returns(self, factor):
        """含 NaN 的股票应被跳过。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A", "B"]
        data = np.array([[0.01, np.nan],
                         [0.02, 0.01],
                         [0.03, 0.02]])
        returns = pd.DataFrame(data, index=dates, columns=stocks)

        result = factor.compute(returns=returns, T=3)
        # B 有 NaN，A 没有有效的 peer -> NaN
        assert np.isnan(result.iloc[2, 0])
        assert np.isnan(result.iloc[2, 1])

    def test_single_stock(self, factor):
        """只有一只股票时，无法计算相关系数，结果为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        returns = pd.DataFrame(
            np.random.normal(0, 0.02, (5, 1)), index=dates, columns=stocks
        )
        result = factor.compute(returns=returns, T=5)
        assert np.isnan(result.iloc[4, 0])

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        returns = pd.DataFrame(
            np.random.normal(0, 0.02, (30, 3)), index=dates, columns=stocks
        )
        result = factor.compute(returns=returns, T=20)
        assert result.shape == returns.shape
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B", "C"]
        returns = pd.DataFrame(
            np.random.normal(0, 0.02, (10, 3)), index=dates, columns=stocks
        )
        T = 5
        result = factor.compute(returns=returns, T=T)
        assert result.iloc[: T - 1].isna().all().all()
