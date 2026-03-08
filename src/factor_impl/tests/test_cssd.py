import numpy as np
import pandas as pd
import pytest

from factors.cssd import CSSDFactor


@pytest.fixture
def factor():
    return CSSDFactor()


class TestCSSDMetadata:
    def test_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "CSSD"
        assert meta["category"] == "行为金融"
        assert "CSSD" in meta["description"] or "截面" in meta["description"]

    def test_repr(self, factor):
        r = repr(factor)
        assert "CSSDFactor" in r
        assert "CSSD" in r


class TestCSSDCompute:
    def test_cssd_known_values(self, factor):
        """用已知数据验证 CSSD 计算。"""
        dates = pd.bdate_range("2025-01-01", periods=1)
        stock_returns = pd.DataFrame(
            {"A": [0.01], "B": [0.03], "C": [-0.02]}, index=dates
        )
        market_return = pd.Series([0.01], index=dates)

        result = factor.compute(
            stock_returns=stock_returns,
            market_return=market_return,
            threshold_quantile=0.05,
        )

        # CSSD = sqrt(((0.01-0.01)^2 + (0.03-0.01)^2 + (-0.02-0.01)^2) / 2)
        expected_cssd = np.sqrt((0.0 + 0.0004 + 0.0009) / 2)
        np.testing.assert_almost_equal(result["cssd"].iloc[0], expected_cssd)

    def test_output_keys(self, factor):
        """输出应包含所有必要的键。"""
        np.random.seed(42)
        dates = pd.bdate_range("2025-01-01", periods=100)
        stock_returns = pd.DataFrame(
            np.random.randn(100, 10) * 0.02,
            index=dates,
            columns=[f"S{i}" for i in range(10)],
        )
        market_return = stock_returns.mean(axis=1)

        result = factor.compute(
            stock_returns=stock_returns, market_return=market_return
        )

        assert "cssd" in result
        assert "beta1" in result
        assert "beta2" in result
        assert "alpha" in result
        assert "contribution" in result

    def test_cssd_series_length(self, factor):
        """CSSD 序列长度应与输入天数一致。"""
        np.random.seed(42)
        dates = pd.bdate_range("2025-01-01", periods=50)
        stock_returns = pd.DataFrame(
            np.random.randn(50, 5) * 0.02,
            index=dates,
            columns=[f"S{i}" for i in range(5)],
        )
        market_return = stock_returns.mean(axis=1)

        result = factor.compute(
            stock_returns=stock_returns, market_return=market_return
        )

        assert len(result["cssd"]) == 50

    def test_contribution_shape(self, factor):
        """个股贡献度的长度应等于股票数量。"""
        np.random.seed(42)
        dates = pd.bdate_range("2025-01-01", periods=50)
        stock_returns = pd.DataFrame(
            np.random.randn(50, 5) * 0.02,
            index=dates,
            columns=[f"S{i}" for i in range(5)],
        )
        market_return = stock_returns.mean(axis=1)

        result = factor.compute(
            stock_returns=stock_returns, market_return=market_return
        )

        assert len(result["contribution"]) == 5

    def test_cssd_non_negative(self, factor):
        """CSSD 应始终非负。"""
        np.random.seed(42)
        dates = pd.bdate_range("2025-01-01", periods=100)
        stock_returns = pd.DataFrame(
            np.random.randn(100, 10) * 0.02,
            index=dates,
            columns=[f"S{i}" for i in range(10)],
        )
        market_return = stock_returns.mean(axis=1)

        result = factor.compute(
            stock_returns=stock_returns, market_return=market_return
        )

        assert (result["cssd"] >= 0).all()

    def test_identical_returns_zero_cssd(self, factor):
        """所有股票收益率相同时，CSSD 应为 0。"""
        dates = pd.bdate_range("2025-01-01", periods=10)
        stock_returns = pd.DataFrame(
            np.tile([0.01], (10, 3)),
            index=dates,
            columns=["A", "B", "C"],
        )
        market_return = pd.Series([0.01] * 10, index=dates)

        result = factor.compute(
            stock_returns=stock_returns, market_return=market_return
        )

        np.testing.assert_array_almost_equal(result["cssd"].values, 0.0)

    def test_beta_types(self, factor):
        """beta1 和 beta2 应为浮点数。"""
        np.random.seed(42)
        dates = pd.bdate_range("2025-01-01", periods=100)
        stock_returns = pd.DataFrame(
            np.random.randn(100, 10) * 0.02,
            index=dates,
            columns=[f"S{i}" for i in range(10)],
        )
        market_return = stock_returns.mean(axis=1)

        result = factor.compute(
            stock_returns=stock_returns, market_return=market_return
        )

        assert isinstance(result["beta1"], float)
        assert isinstance(result["beta2"], float)
