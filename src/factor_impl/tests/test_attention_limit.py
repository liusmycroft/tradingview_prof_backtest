import numpy as np
import pandas as pd
import pytest

from factors.attention_limit import AttentionLimitFactor


@pytest.fixture
def factor():
    return AttentionLimitFactor()


class TestAttentionLimitMetadata:
    def test_name(self, factor):
        assert factor.name == "ATTENTION_LIMIT"

    def test_category(self, factor):
        assert factor.category == "行为金融"

    def test_repr(self, factor):
        assert "ATTENTION_LIMIT" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "ATTENTION_LIMIT"
        assert meta["category"] == "行为金融"


class TestAttentionLimitHandCalculated:
    """用手算数据验证滚动 OLS 回归 |beta| 的正确性。"""

    def test_perfect_positive_linear(self, factor):
        """y = 2*x 的完美线性关系, |beta| 应为 2。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        x_vals = np.linspace(0.01, 0.05, 20)
        y_vals = 2.0 * x_vals

        stock_returns = pd.DataFrame({"A": y_vals}, index=dates)
        limit_prop = pd.Series(x_vals, index=dates)

        result = factor.compute(stock_returns=stock_returns, limit_prop=limit_prop, T=20)
        assert result.iloc[-1, 0] == pytest.approx(2.0, rel=1e-6)

    def test_perfect_negative_linear(self, factor):
        """y = -3*x, 负 beta 应取绝对值, |beta| = 3。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        x_vals = np.linspace(0.01, 0.05, 20)
        y_vals = -3.0 * x_vals

        stock_returns = pd.DataFrame({"A": y_vals}, index=dates)
        limit_prop = pd.Series(x_vals, index=dates)

        result = factor.compute(stock_returns=stock_returns, limit_prop=limit_prop, T=20)
        assert result.iloc[-1, 0] == pytest.approx(3.0, rel=1e-6)

    def test_constant_returns_zero_beta(self, factor):
        """收益率为常数时, cov(x, y) = 0, beta = 0。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stock_returns = pd.DataFrame({"A": [0.01] * 20}, index=dates)
        limit_prop = pd.Series(np.linspace(0.01, 0.05, 20), index=dates)

        result = factor.compute(stock_returns=stock_returns, limit_prop=limit_prop, T=20)
        assert result.iloc[-1, 0] == pytest.approx(0.0, abs=1e-10)

    def test_T3_manual_ols(self, factor):
        """T=3, 手动计算 OLS beta。

        x = [0.01, 0.02, 0.03], y = [0.05, 0.08, 0.11]
        x_mean = 0.02, y_mean = 0.08
        x_demean = [-0.01, 0, 0.01]
        cov_xy = (-0.01)*(-0.03) + 0*0 + 0.01*0.03 = 0.0006
        var_x  = 0.0001 + 0 + 0.0001 = 0.0002
        beta = 0.0006 / 0.0002 = 3.0
        |beta| = 3.0
        """
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stock_returns = pd.DataFrame({"A": [0.05, 0.08, 0.11]}, index=dates)
        limit_prop = pd.Series([0.01, 0.02, 0.03], index=dates)

        result = factor.compute(stock_returns=stock_returns, limit_prop=limit_prop, T=3)

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(3.0, rel=1e-10)

    def test_two_stocks(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        x_vals = np.linspace(0.01, 0.05, 20)
        stock_returns = pd.DataFrame(
            {"S1": 2.0 * x_vals, "S2": -1.0 * x_vals}, index=dates
        )
        limit_prop = pd.Series(x_vals, index=dates)

        result = factor.compute(stock_returns=stock_returns, limit_prop=limit_prop, T=20)
        assert result.iloc[-1, 0] == pytest.approx(2.0, rel=1e-6)
        assert result.iloc[-1, 1] == pytest.approx(1.0, rel=1e-6)


class TestAttentionLimitEdgeCases:
    def test_nan_in_window_skips(self, factor):
        """窗口内含 NaN 时, mask.sum() < T, 该窗口结果为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stock_returns = pd.DataFrame({"A": [0.01, np.nan, 0.03, 0.04, 0.05]}, index=dates)
        limit_prop = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05], index=dates)

        result = factor.compute(stock_returns=stock_returns, limit_prop=limit_prop, T=3)
        # 窗口 [0,1,2] 含 NaN -> NaN; 窗口 [1,2,3] 含 NaN -> NaN
        assert np.isnan(result.iloc[2, 0])
        assert np.isnan(result.iloc[3, 0])
        # 窗口 [2,3,4] 无 NaN -> 有值
        assert not np.isnan(result.iloc[4, 0])

    def test_constant_x_zero_var(self, factor):
        """x 为常数时, var(x) = 0, beta 应为 0。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stock_returns = pd.DataFrame({"A": [0.01, 0.02, 0.03, 0.04, 0.05]}, index=dates)
        limit_prop = pd.Series([0.03] * 5, index=dates)

        result = factor.compute(stock_returns=stock_returns, limit_prop=limit_prop, T=5)
        assert result.iloc[-1, 0] == pytest.approx(0.0, abs=1e-10)

    def test_insufficient_window_returns_nan(self, factor):
        """数据不足 T 天时应返回 NaN。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stock_returns = pd.DataFrame({"A": np.random.randn(10) * 0.02}, index=dates)
        limit_prop = pd.Series(np.random.uniform(0.01, 0.05, 10), index=dates)

        result = factor.compute(stock_returns=stock_returns, limit_prop=limit_prop, T=20)
        assert result.isna().all().all()


class TestAttentionLimitOutputShape:
    def test_output_shape_matches_input(self, factor):
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        stock_returns = pd.DataFrame(
            np.random.randn(30, 2) * 0.02, index=dates, columns=stocks
        )
        limit_prop = pd.Series(np.random.uniform(0.01, 0.05, 30), index=dates)

        result = factor.compute(stock_returns=stock_returns, limit_prop=limit_prop, T=20)

        assert result.shape == stock_returns.shape
        assert list(result.columns) == list(stock_returns.columns)
        assert list(result.index) == list(stock_returns.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        stock_returns = pd.DataFrame(np.random.randn(25) * 0.02, index=dates, columns=stocks)
        limit_prop = pd.Series(np.random.uniform(0.01, 0.05, 25), index=dates)

        result = factor.compute(stock_returns=stock_returns, limit_prop=limit_prop, T=20)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        """前 T-1 行应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        T = 10
        x_vals = np.linspace(0.01, 0.05, 30)
        stock_returns = pd.DataFrame(
            {"A": 2.0 * x_vals, "B": -1.0 * x_vals}, index=dates
        )
        limit_prop = pd.Series(x_vals, index=dates)

        result = factor.compute(stock_returns=stock_returns, limit_prop=limit_prop, T=T)

        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
