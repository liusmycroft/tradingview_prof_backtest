import numpy as np
import pandas as pd
import pytest

from factors.ideal_reversal import IdealReversalFactor


@pytest.fixture
def factor():
    return IdealReversalFactor()


class TestIdealReversalMetadata:
    def test_name(self, factor):
        assert factor.name == "IDEAL_REVERSAL"

    def test_category(self, factor):
        assert factor.category == "高频因子-动量反转类"

    def test_repr(self, factor):
        assert "IDEAL_REVERSAL" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "IDEAL_REVERSAL"
        assert meta["category"] == "高频因子-动量反转类"


class TestIdealReversalCompute:
    def test_basic_computation(self, factor):
        """高分位日收益高于低分位日收益时，因子为正。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        # 分位值高的日子收益也高
        quantiles = np.arange(1, 21, dtype=float)
        returns = np.arange(1, 21, dtype=float) * 0.01
        daily_return = pd.DataFrame(returns, index=dates, columns=stocks)
        daily_amount_quantile = pd.DataFrame(quantiles, index=dates, columns=stocks)

        result = factor.compute(
            daily_return=daily_return,
            daily_amount_quantile=daily_amount_quantile,
            N=20,
        )
        # 最后一行应有值且为正（高分位日收益 > 低分位日收益）
        assert not np.isnan(result.iloc[-1, 0])
        assert result.iloc[-1, 0] > 0

    def test_symmetric_returns_zero(self, factor):
        """分位值与收益无关时，高低组差异应较小。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        returns = np.array([0.01] * 20)
        quantiles = np.arange(1, 21, dtype=float)
        daily_return = pd.DataFrame(returns, index=dates, columns=stocks)
        daily_amount_quantile = pd.DataFrame(quantiles, index=dates, columns=stocks)

        result = factor.compute(
            daily_return=daily_return,
            daily_amount_quantile=daily_amount_quantile,
            N=20,
        )
        # 所有收益相同，M_high - M_low = 0
        assert result.iloc[-1, 0] == pytest.approx(0.0, abs=1e-10)

    def test_leading_nan(self, factor):
        """前 N-1 行应为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily_return = pd.DataFrame(np.random.randn(25) * 0.01, index=dates, columns=stocks)
        daily_amount_quantile = pd.DataFrame(np.random.rand(25), index=dates, columns=stocks)

        result = factor.compute(
            daily_return=daily_return,
            daily_amount_quantile=daily_amount_quantile,
            N=20,
        )
        assert result.iloc[:19]["A"].isna().all()
        assert result.iloc[19:]["A"].notna().all()

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A", "B"]
        ret_a = np.arange(1, 21, dtype=float) * 0.01
        ret_b = np.arange(20, 0, -1, dtype=float) * 0.01
        daily_return = pd.DataFrame({"A": ret_a, "B": ret_b}, index=dates)
        quantiles = np.arange(1, 21, dtype=float)
        daily_amount_quantile = pd.DataFrame({"A": quantiles, "B": quantiles}, index=dates)

        result = factor.compute(
            daily_return=daily_return,
            daily_amount_quantile=daily_amount_quantile,
            N=20,
        )
        # A: 高分位日收益高 -> 正; B: 高分位日收益低 -> 负
        assert result.iloc[-1]["A"] > 0
        assert result.iloc[-1]["B"] < 0


class TestIdealReversalEdgeCases:
    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        daily_return = pd.DataFrame(np.random.randn(30, 2) * 0.01, index=dates, columns=stocks)
        daily_amount_quantile = pd.DataFrame(np.random.rand(30, 2), index=dates, columns=stocks)

        result = factor.compute(
            daily_return=daily_return,
            daily_amount_quantile=daily_amount_quantile,
            N=20,
        )
        assert result.shape == daily_return.shape
        assert list(result.columns) == stocks

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        daily_return = pd.DataFrame(np.random.randn(20) * 0.01, index=dates, columns=stocks)
        daily_amount_quantile = pd.DataFrame(np.random.rand(20), index=dates, columns=stocks)

        result = factor.compute(
            daily_return=daily_return,
            daily_amount_quantile=daily_amount_quantile,
            N=20,
        )
        assert isinstance(result, pd.DataFrame)
