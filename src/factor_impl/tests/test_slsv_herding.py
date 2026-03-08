import numpy as np
import pandas as pd
import pytest

from factors.slsv_herding import SLSVHerdingFactor


@pytest.fixture
def factor():
    return SLSVHerdingFactor()


class TestSLSVHerdingMetadata:
    def test_name(self, factor):
        assert factor.name == "SLSV_HERDING"

    def test_category(self, factor):
        assert factor.category == "行为金融-羊群效应"

    def test_repr(self, factor):
        assert "SLSV_HERDING" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "SLSV_HERDING"
        assert meta["category"] == "行为金融-羊群效应"


class TestSLSVHerdingCompute:
    def test_symmetric_buy_sell_gives_zero_lsv(self, factor):
        """所有股票 p_it 相同时, |p_it - p_t| = 0, LSV = -AF <= 0。"""
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A", "B", "C"]
        # 所有股票买卖比例相同
        buy = pd.DataFrame([[100, 100, 100]], index=dates, columns=stocks)
        sell = pd.DataFrame([[100, 100, 100]], index=dates, columns=stocks)

        result = factor.compute(daily_buy_count=buy, daily_sell_count=sell)
        # p_it = 0.5 for all, p_t = 0.5, diff = 0, sign = 0 => SLSV = 0
        for s in stocks:
            assert result.iloc[0][s] == pytest.approx(0.0, abs=1e-6) or pd.isna(result.iloc[0][s]) or result.iloc[0][s] <= 0

    def test_strong_buy_herding(self, factor):
        """一只股票被大量买入, 其他被卖出 => 该股票 SLSV > 0。"""
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A", "B"]
        buy = pd.DataFrame([[200, 50]], index=dates, columns=stocks)
        sell = pd.DataFrame([[50, 200]], index=dates, columns=stocks)

        result = factor.compute(daily_buy_count=buy, daily_sell_count=sell)
        # A: p_it=0.8, B: p_it=0.2, p_t=0.5
        # A: diff > 0, sign = +1 => SLSV_A > 0 (if LSV > 0)
        # B: diff < 0, sign = -1 => SLSV_B < 0 (if LSV > 0)
        assert result.iloc[0, 0] > 0  # A has buy herding
        assert result.iloc[0, 1] < 0  # B has sell herding

    def test_sign_opposite_for_buy_vs_sell(self, factor):
        """买入从众和卖出从众应有相反符号。"""
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A", "B"]
        buy = pd.DataFrame([[300, 100]], index=dates, columns=stocks)
        sell = pd.DataFrame([[100, 300]], index=dates, columns=stocks)

        result = factor.compute(daily_buy_count=buy, daily_sell_count=sell)
        # A: p_it = 0.75, B: p_it = 0.25, p_t = 0.5
        # 对称情况下 |LSV| 相同, 但符号相反
        assert result.iloc[0, 0] == pytest.approx(-result.iloc[0, 1], rel=1e-6)

    def test_zero_total_gives_nan(self, factor):
        """买卖数量都为 0 时, 结果应为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A"]
        buy = pd.DataFrame([[0]], index=dates, columns=stocks)
        sell = pd.DataFrame([[0]], index=dates, columns=stocks)

        result = factor.compute(daily_buy_count=buy, daily_sell_count=sell)
        assert pd.isna(result.iloc[0, 0])

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=4, freq="Q")
        stocks = ["A", "B", "C"]
        np.random.seed(42)
        buy = pd.DataFrame(np.random.randint(50, 300, (4, 3)), index=dates, columns=stocks)
        sell = pd.DataFrame(np.random.randint(50, 300, (4, 3)), index=dates, columns=stocks)

        result = factor.compute(daily_buy_count=buy, daily_sell_count=sell)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (4, 3)
        assert list(result.columns) == stocks

    def test_single_stock_p_it_equals_p_t(self, factor):
        """单只股票时 p_it = p_t, diff = 0。"""
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A"]
        buy = pd.DataFrame([[150]], index=dates, columns=stocks)
        sell = pd.DataFrame([[50]], index=dates, columns=stocks)

        result = factor.compute(daily_buy_count=buy, daily_sell_count=sell)
        # p_it = p_t => diff = 0, sign = 0 => SLSV = 0
        assert result.iloc[0, 0] == pytest.approx(0.0, abs=1e-6) or pd.isna(result.iloc[0, 0])
