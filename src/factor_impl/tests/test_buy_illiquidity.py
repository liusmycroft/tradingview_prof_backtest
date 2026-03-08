import numpy as np
import pandas as pd
import pytest

from factors.buy_illiquidity import BuyIlliquidityFactor


@pytest.fixture
def factor():
    return BuyIlliquidityFactor()


class TestBuyIlliquidityMetadata:
    def test_name(self, factor):
        assert factor.name == "BUY_ILLIQUIDITY"

    def test_category(self, factor):
        assert factor.category == "高频流动性"

    def test_repr(self, factor):
        assert "BUY_ILLIQUIDITY" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "BUY_ILLIQUIDITY"
        assert meta["category"] == "高频流动性"
        assert "买单" in meta["description"] or "流动性" in meta["description"]


class TestBuyIlliquidityCompute:
    """测试 compute 方法。"""

    def test_basic_regression(self, factor):
        """用已知线性关系验证回归系数。

        构造 r = 0.01 + 0.5*S + 2.0*B，beta2 应接近 2.0。
        """
        dates = pd.bdate_range("2025-01-01", periods=1)
        np.random.seed(42)
        n_stocks = 50
        stocks = [f"S{i:03d}" for i in range(n_stocks)]

        s = np.random.uniform(1, 10, n_stocks)
        b = np.random.uniform(1, 10, n_stocks)
        r = 0.01 + 0.5 * s + 2.0 * b

        returns = pd.DataFrame([r], index=dates, columns=stocks)
        sell_amount = pd.DataFrame([s], index=dates, columns=stocks)
        buy_amount = pd.DataFrame([b], index=dates, columns=stocks)

        result = factor.compute(
            returns=returns, sell_amount=sell_amount, buy_amount=buy_amount
        )

        # beta2 应接近 2.0
        assert result.iloc[0, 0] == pytest.approx(2.0, abs=1e-10)

    def test_beta2_with_noise(self, factor):
        """带噪声的回归，beta2 应接近真实值。"""
        dates = pd.bdate_range("2025-01-01", periods=1)
        np.random.seed(123)
        n_stocks = 200
        stocks = [f"S{i:03d}" for i in range(n_stocks)]

        s = np.random.uniform(1e6, 1e7, n_stocks)
        b = np.random.uniform(1e6, 1e7, n_stocks)
        noise = np.random.randn(n_stocks) * 0.001
        beta2_true = 3e-9
        r = 0.01 + 1e-9 * s + beta2_true * b + noise

        returns = pd.DataFrame([r], index=dates, columns=stocks)
        sell_amount = pd.DataFrame([s], index=dates, columns=stocks)
        buy_amount = pd.DataFrame([b], index=dates, columns=stocks)

        result = factor.compute(
            returns=returns, sell_amount=sell_amount, buy_amount=buy_amount
        )

        # 由于噪声，允许较大的容差
        assert abs(result.iloc[0, 0] - beta2_true) < beta2_true * 5

    def test_multi_day(self, factor):
        """多日截面回归。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        np.random.seed(42)
        n_stocks = 30
        stocks = [f"S{i:03d}" for i in range(n_stocks)]

        results_list = []
        for _ in range(3):
            s = np.random.uniform(1, 10, n_stocks)
            b = np.random.uniform(1, 10, n_stocks)
            r = 0.01 + 0.5 * s + 1.5 * b
            results_list.append((r, s, b))

        returns = pd.DataFrame(
            [x[0] for x in results_list], index=dates, columns=stocks
        )
        sell_amount = pd.DataFrame(
            [x[1] for x in results_list], index=dates, columns=stocks
        )
        buy_amount = pd.DataFrame(
            [x[2] for x in results_list], index=dates, columns=stocks
        )

        result = factor.compute(
            returns=returns, sell_amount=sell_amount, buy_amount=buy_amount
        )

        assert result.shape == (3, n_stocks)
        # 每天的 beta2 都应接近 1.5
        for t in range(3):
            assert result.iloc[t, 0] == pytest.approx(1.5, abs=1e-10)

    def test_same_beta2_across_stocks(self, factor):
        """同一天截面回归，所有股票的 beta2 应相同。"""
        dates = pd.bdate_range("2025-01-01", periods=1)
        np.random.seed(42)
        n_stocks = 20
        stocks = [f"S{i:03d}" for i in range(n_stocks)]

        s = np.random.uniform(1, 10, n_stocks)
        b = np.random.uniform(1, 10, n_stocks)
        r = 0.5 * s + 2.0 * b

        returns = pd.DataFrame([r], index=dates, columns=stocks)
        sell_amount = pd.DataFrame([s], index=dates, columns=stocks)
        buy_amount = pd.DataFrame([b], index=dates, columns=stocks)

        result = factor.compute(
            returns=returns, sell_amount=sell_amount, buy_amount=buy_amount
        )

        # 同一天所有股票的 beta2 应相同
        vals = result.iloc[0].values
        valid = vals[~np.isnan(vals)]
        assert len(set(np.round(valid, 10))) == 1


class TestBuyIlliquidityEdgeCases:
    def test_too_few_stocks(self, factor):
        """股票数少于 3 时，无法回归，结果应为 NaN。"""
        dates = pd.bdate_range("2025-01-01", periods=1)
        stocks = ["A", "B"]

        returns = pd.DataFrame([[0.01, 0.02]], index=dates, columns=stocks)
        sell_amount = pd.DataFrame([[1e6, 2e6]], index=dates, columns=stocks)
        buy_amount = pd.DataFrame([[1e6, 2e6]], index=dates, columns=stocks)

        result = factor.compute(
            returns=returns, sell_amount=sell_amount, buy_amount=buy_amount
        )

        assert result.isna().all().all()

    def test_nan_in_input(self, factor):
        """输入含 NaN 时不应抛异常。"""
        dates = pd.bdate_range("2025-01-01", periods=1)
        stocks = ["A", "B", "C", "D", "E"]

        returns = pd.DataFrame(
            [[0.01, np.nan, 0.03, 0.04, 0.05]], index=dates, columns=stocks
        )
        sell_amount = pd.DataFrame(
            [[1e6, 2e6, 3e6, 4e6, 5e6]], index=dates, columns=stocks
        )
        buy_amount = pd.DataFrame(
            [[1e6, 2e6, 3e6, 4e6, 5e6]], index=dates, columns=stocks
        )

        result = factor.compute(
            returns=returns, sell_amount=sell_amount, buy_amount=buy_amount
        )

        assert isinstance(result, pd.DataFrame)


class TestBuyIlliquidityOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=10)
        stocks = ["A", "B", "C", "D", "E"]
        np.random.seed(42)

        returns = pd.DataFrame(
            np.random.randn(10, 5) * 0.02, index=dates, columns=stocks
        )
        sell_amount = pd.DataFrame(
            np.random.uniform(1e6, 1e7, (10, 5)), index=dates, columns=stocks
        )
        buy_amount = pd.DataFrame(
            np.random.uniform(1e6, 1e7, (10, 5)), index=dates, columns=stocks
        )

        result = factor.compute(
            returns=returns, sell_amount=sell_amount, buy_amount=buy_amount
        )

        assert result.shape == returns.shape
        assert list(result.columns) == list(returns.columns)

    def test_output_is_dataframe(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=3)
        stocks = ["A", "B", "C", "D"]
        np.random.seed(42)

        returns = pd.DataFrame(
            np.random.randn(3, 4) * 0.02, index=dates, columns=stocks
        )
        sell_amount = pd.DataFrame(
            np.random.uniform(1e6, 1e7, (3, 4)), index=dates, columns=stocks
        )
        buy_amount = pd.DataFrame(
            np.random.uniform(1e6, 1e7, (3, 4)), index=dates, columns=stocks
        )

        result = factor.compute(
            returns=returns, sell_amount=sell_amount, buy_amount=buy_amount
        )
        assert isinstance(result, pd.DataFrame)
