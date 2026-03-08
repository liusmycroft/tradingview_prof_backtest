import numpy as np
import pandas as pd
import pytest

from factors.opening_buy_intention import OpeningBuyIntentionFactor


@pytest.fixture
def factor():
    return OpeningBuyIntentionFactor()


class TestOpeningBuyIntentionMetadata:
    def test_name(self, factor):
        assert factor.name == "OPENING_BUY_INTENTION"

    def test_category(self, factor):
        assert factor.category == "高频资金流"

    def test_repr(self, factor):
        assert "OPENING_BUY_INTENTION" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "OPENING_BUY_INTENTION"
        assert meta["category"] == "高频资金流"


class TestOpeningBuyIntentionHandCalculated:
    """用手算数据验证开盘后买入意愿占比因子。"""

    def test_constant_ratio(self, factor):
        """常数: buy_intention=50, amount=500 => ratio=0.1, 均值=0.1。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        intention = pd.DataFrame(50.0, index=dates, columns=stocks)
        amount = pd.DataFrame(500.0, index=dates, columns=stocks)

        result = factor.compute(buy_intention=intention, amount=amount, T=3)

        # min_periods=T=3, 前2行NaN
        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(0.1, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(0.1, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(0.1, rel=1e-10)

    def test_varying_ratio_T3(self, factor):
        """T=3, 变化的买入意愿。

        daily_ratio = [0.1, 0.2, 0.3]
        rolling(3, min_periods=3):
          day0: NaN
          day1: NaN
          day2: mean([0.1, 0.2, 0.3]) = 0.2
        """
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        intention = pd.DataFrame([100.0, 200.0, 300.0], index=dates, columns=stocks)
        amount = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(buy_intention=intention, amount=amount, T=3)

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(0.2, rel=1e-10)

    def test_two_stocks(self, factor):
        """两只股票并行计算。

        Stock A: intention/amount = [0.1, 0.2] -> T=2: [NaN, 0.15]
        Stock B: intention/amount = [-0.2, -0.4] -> T=2: [NaN, -0.3]
        """
        dates = pd.date_range("2024-01-01", periods=2, freq="D")
        stocks = ["A", "B"]

        intention = pd.DataFrame(
            [[100.0, -200.0], [200.0, -400.0]], index=dates, columns=stocks
        )
        amount = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(buy_intention=intention, amount=amount, T=2)

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[0, 1])
        assert result.loc[dates[1], "A"] == pytest.approx(0.15, rel=1e-10)
        assert result.loc[dates[1], "B"] == pytest.approx(-0.3, rel=1e-10)

    def test_rolling_window_slides(self, factor):
        """验证滚动窗口正确滑动 (T=2)。

        daily_ratio = [0.1, 0.2, 0.3, 0.4]
        rolling(2, min_periods=2):
          day0: NaN
          day1: mean([0.1, 0.2]) = 0.15
          day2: mean([0.2, 0.3]) = 0.25
          day3: mean([0.3, 0.4]) = 0.35
        """
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        stocks = ["A"]

        intention = pd.DataFrame(
            [100.0, 200.0, 300.0, 400.0], index=dates, columns=stocks
        )
        amount = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(buy_intention=intention, amount=amount, T=2)

        assert np.isnan(result.iloc[0, 0])
        assert result.iloc[1, 0] == pytest.approx(0.15, rel=1e-10)
        assert result.iloc[2, 0] == pytest.approx(0.25, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(0.35, rel=1e-10)


class TestOpeningBuyIntentionEdgeCases:
    def test_nan_in_intention(self, factor):
        """buy_intention 中含 NaN 时, 结果不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        intention = pd.DataFrame([100.0, np.nan, 300.0], index=dates, columns=stocks)
        amount = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(buy_intention=intention, amount=amount, T=3)
        assert isinstance(result.iloc[2, 0], float)

    def test_zero_amount(self, factor):
        """成交金额为 0 时, 除法产生 inf/nan。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        intention = pd.DataFrame([100.0, 200.0, 300.0], index=dates, columns=stocks)
        amount = pd.DataFrame([1000.0, 0.0, 1000.0], index=dates, columns=stocks)

        result = factor.compute(buy_intention=intention, amount=amount, T=3)
        # 不应抛异常
        assert isinstance(result.iloc[2, 0], float)

    def test_negative_intention(self, factor):
        """负买入意愿（卖出主导）时, 因子值应为负。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        intention = pd.DataFrame([-100.0, -200.0, -300.0], index=dates, columns=stocks)
        amount = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(buy_intention=intention, amount=amount, T=3)
        assert result.iloc[2, 0] < 0

    def test_zero_intention(self, factor):
        """买入意愿为 0 时, 因子值应为 0。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        intention = pd.DataFrame(0.0, index=dates, columns=stocks)
        amount = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(buy_intention=intention, amount=amount, T=3)
        assert result.iloc[2, 0] == pytest.approx(0.0, abs=1e-15)


class TestOpeningBuyIntentionOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        stocks = ["A", "B", "C"]

        intention = pd.DataFrame(
            np.random.randn(50, 3) * 100, index=dates, columns=stocks
        )
        amount = pd.DataFrame(
            np.random.uniform(100, 1000, (50, 3)), index=dates, columns=stocks
        )

        result = factor.compute(buy_intention=intention, amount=amount, T=20)

        assert result.shape == intention.shape
        assert list(result.columns) == list(intention.columns)
        assert list(result.index) == list(intention.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        intention = pd.DataFrame(50.0, index=dates, columns=stocks)
        amount = pd.DataFrame(500.0, index=dates, columns=stocks)

        result = factor.compute(buy_intention=intention, amount=amount, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        """前 T-1 行应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        T = 5

        intention = pd.DataFrame(
            np.random.randn(10, 2) * 100, index=dates, columns=stocks
        )
        amount = pd.DataFrame(
            np.random.uniform(100, 1000, (10, 2)), index=dates, columns=stocks
        )

        result = factor.compute(buy_intention=intention, amount=amount, T=T)

        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
