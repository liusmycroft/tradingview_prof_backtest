import numpy as np
import pandas as pd
import pytest
from scipy.special import comb

from factors.lsv_herding import LSVHerdingFactor


@pytest.fixture
def factor():
    return LSVHerdingFactor()


class TestLSVHerdingMetadata:
    def test_name(self, factor):
        assert factor.name == "LSV_HERDING"

    def test_category(self, factor):
        assert factor.category == "行为金融-羊群效应"

    def test_repr(self, factor):
        assert "LSV_HERDING" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "LSV_HERDING"
        assert meta["category"] == "行为金融-羊群效应"


class TestLSVHerdingHandCalculated:
    """手算验证 LSV 因子。"""

    def test_symmetric_buy_sell(self, factor):
        """所有股票买卖各半时，p_it=0.5, p_t=0.5, |p_it-p_t|=0，
        AF也应接近0（二项分布对称），LSV接近0。"""
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A", "B", "C"]
        buy = pd.DataFrame([[100, 100, 100]], index=dates, columns=stocks)
        sell = pd.DataFrame([[100, 100, 100]], index=dates, columns=stocks)

        result = factor.compute(daily_buy_count=buy, daily_sell_count=sell)
        # |0.5 - 0.5| = 0, AF > 0 (binomial expectation), so LSV < 0
        for s in stocks:
            assert result.loc[dates[0], s] < 0

    def test_extreme_herding(self, factor):
        """一只股票全部买入，其他全部卖出，应有较大正LSV。"""
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A", "B"]
        buy = pd.DataFrame([[200, 0]], index=dates, columns=stocks)
        sell = pd.DataFrame([[0, 200]], index=dates, columns=stocks)

        result = factor.compute(daily_buy_count=buy, daily_sell_count=sell)
        # A: p_it=1.0, B: p_it=0.0, p_t=0.5
        # |1.0 - 0.5| = 0.5, AF < 0.5 => LSV > 0
        assert result.iloc[0, 0] > 0  # stock A
        assert result.iloc[0, 1] > 0  # stock B (|0 - 0.5| = 0.5 too)

    def test_manual_small_example(self, factor):
        """手算小例子：2只股票，B=[3,1], S=[1,3], N=[4,4]。"""
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A", "B"]
        buy = pd.DataFrame([[3, 1]], index=dates, columns=stocks)
        sell = pd.DataFrame([[1, 3]], index=dates, columns=stocks)

        result = factor.compute(daily_buy_count=buy, daily_sell_count=sell)

        # p_A = 3/4 = 0.75, p_B = 1/4 = 0.25, p_t = 0.5
        # |p_A - p_t| = 0.25, |p_B - p_t| = 0.25
        # AF for N=4, p_t=0.5:
        # k=0: C(4,0)*0.5^4 * |0-0.5| = 1*0.0625*0.5 = 0.03125
        # k=1: C(4,1)*0.5^4 * |0.25-0.5| = 4*0.0625*0.25 = 0.0625
        # k=2: C(4,2)*0.5^4 * |0.5-0.5| = 6*0.0625*0 = 0
        # k=3: C(4,3)*0.5^4 * |0.75-0.5| = 4*0.0625*0.25 = 0.0625
        # k=4: C(4,4)*0.5^4 * |1.0-0.5| = 1*0.0625*0.5 = 0.03125
        # AF = 0.03125 + 0.0625 + 0 + 0.0625 + 0.03125 = 0.1875
        expected_af = 0.1875
        expected_lsv = 0.25 - expected_af  # = 0.0625

        assert result.iloc[0, 0] == pytest.approx(expected_lsv, abs=1e-10)
        assert result.iloc[0, 1] == pytest.approx(expected_lsv, abs=1e-10)

    def test_two_dates_independent(self, factor):
        """两个日期应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=2, freq="D")
        stocks = ["A", "B"]
        buy = pd.DataFrame([[3, 1], [2, 2]], index=dates, columns=stocks)
        sell = pd.DataFrame([[1, 3], [2, 2]], index=dates, columns=stocks)

        result = factor.compute(daily_buy_count=buy, daily_sell_count=sell)
        assert result.shape == (2, 2)
        # Day 2: p_it = 0.5 for both, p_t = 0.5, |diff| = 0
        # LSV should be negative (0 - AF < 0)
        assert result.iloc[1, 0] < 0
        assert result.iloc[1, 1] < 0


class TestLSVHerdingEdgeCases:
    def test_zero_total(self, factor):
        """B=0, S=0 时应返回 NaN。"""
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A"]
        buy = pd.DataFrame([[0]], index=dates, columns=stocks)
        sell = pd.DataFrame([[0]], index=dates, columns=stocks)

        result = factor.compute(daily_buy_count=buy, daily_sell_count=sell)
        assert result.isna().all().all()

    def test_single_stock(self, factor):
        """单只股票时 p_t = p_it，|diff|=0。"""
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A"]
        buy = pd.DataFrame([[10]], index=dates, columns=stocks)
        sell = pd.DataFrame([[5]], index=dates, columns=stocks)

        result = factor.compute(daily_buy_count=buy, daily_sell_count=sell)
        # |p_it - p_t| = 0, AF >= 0, so LSV <= 0
        assert result.iloc[0, 0] <= 0


class TestLSVHerdingOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A", "B", "C"]
        buy = pd.DataFrame(
            np.random.randint(10, 100, (5, 3)), index=dates, columns=stocks
        )
        sell = pd.DataFrame(
            np.random.randint(10, 100, (5, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_buy_count=buy, daily_sell_count=sell)
        assert result.shape == buy.shape
        assert list(result.columns) == list(buy.columns)
        assert list(result.index) == list(buy.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]
        buy = pd.DataFrame([10, 20, 30], index=dates, columns=stocks)
        sell = pd.DataFrame([5, 10, 15], index=dates, columns=stocks)

        result = factor.compute(daily_buy_count=buy, daily_sell_count=sell)
        assert isinstance(result, pd.DataFrame)
