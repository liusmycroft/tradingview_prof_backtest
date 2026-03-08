import numpy as np
import pandas as pd
import pytest

from factors.loss_selling import LossSellingFactor


@pytest.fixture
def factor():
    return LossSellingFactor()


class TestLossSellingMetadata:
    def test_name(self, factor):
        assert factor.name == "LOSS_SELLING"

    def test_category(self, factor):
        assert factor.category == "行为金融"

    def test_repr(self, factor):
        assert "LOSS_SELLING" in repr(factor)
        assert "LossSellingFactor" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "LOSS_SELLING"
        assert meta["category"] == "行为金融"
        assert "亏损" in meta["description"]


class TestLossSellingHandCalculated:
    """用手算数据验证权重计算和亏损卖出因子公式的正确性。"""

    def test_T3_single_stock_all_loss(self, factor):
        """T=3, 单只股票, close < vwap 全部亏损。

        3 天数据 (day 0, 1, 2):
          turnover = [0.2, 0.3, 0.5]
          close    = [9,   11,  13]
          vwap     = [10,  12,  14]

        daily_loss = min(0, (close - vwap) / vwap):
          day0: min(0, (9-10)/10)  = -0.1
          day1: min(0, (11-12)/12) = -1/12
          day2: min(0, (13-14)/14) = -1/14

        在 t=2, T=3, 窗口 = [day0, day1, day2], 翻转后 n=0 对应 day2:
          tv_rev = [0.5, 0.3, 0.2]
          loss_rev = [-1/14, -1/12, -0.1]

          权重计算:
            w_0 = 0.5
            w_1 = 0.3  (shifted cum_keep[0] = 1)
            w_2 = 0.2 * (1 - 0.3) = 0.14

          k = 0.5 + 0.3 + 0.14 = 0.94
          Loss = (0.5*(-1/14) + 0.3*(-1/12) + 0.14*(-0.1)) / 0.94
        """
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        close = pd.DataFrame([9.0, 11.0, 13.0], index=dates, columns=stocks)
        vwap = pd.DataFrame([10.0, 12.0, 14.0], index=dates, columns=stocks)
        turnover = pd.DataFrame([0.2, 0.3, 0.5], index=dates, columns=stocks)

        result = factor.compute(close=close, vwap=vwap, turnover=turnover, T=3)

        # 前 T-1=2 天应为 NaN
        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])

        w = np.array([0.5, 0.3, 0.14])
        loss_rev = np.array([-1.0 / 14, -1.0 / 12, -0.1])
        k = w.sum()
        expected = (w * loss_rev).sum() / k
        assert result.iloc[2, 0] == pytest.approx(expected, rel=1e-10)

    def test_T2_two_stocks(self, factor):
        """T=2, 两只股票。

        Stock A: close=[9, 11], vwap=[10, 12], turnover=[0.1, 0.4]
          daily_loss: [-0.1, -1/12]
          t=1: tv_rev=[0.4, 0.1], loss_rev=[-1/12, -0.1]
            w_0=0.4, w_1=0.1, k=0.5
            Loss = (0.4*(-1/12) + 0.1*(-0.1)) / 0.5

        Stock B: close=[30, 35], vwap=[25, 30], turnover=[0.5, 0.2]
          daily_loss: [min(0,(30-25)/25), min(0,(35-30)/30)] = [0, 0]
          t=1: Loss = 0
        """
        dates = pd.date_range("2024-01-01", periods=2, freq="D")
        stocks = ["A", "B"]

        close = pd.DataFrame([[9.0, 30.0], [11.0, 35.0]], index=dates, columns=stocks)
        vwap = pd.DataFrame([[10.0, 25.0], [12.0, 30.0]], index=dates, columns=stocks)
        turnover = pd.DataFrame([[0.1, 0.5], [0.4, 0.2]], index=dates, columns=stocks)

        result = factor.compute(close=close, vwap=vwap, turnover=turnover, T=2)

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[0, 1])

        # Stock A
        expected_a = (0.4 * (-1.0 / 12) + 0.1 * (-0.1)) / 0.5
        assert result.loc[dates[1], "A"] == pytest.approx(expected_a, rel=1e-10)

        # Stock B: all gains, loss clipped to 0
        assert result.loc[dates[1], "B"] == pytest.approx(0.0, abs=1e-15)

    def test_no_loss_gives_zero(self, factor):
        """close >= vwap 时, daily_loss 全为 0, 因子值应为 0。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        close = pd.DataFrame([12.0, 14.0, 16.0], index=dates, columns=stocks)
        vwap = pd.DataFrame([10.0, 12.0, 14.0], index=dates, columns=stocks)
        turnover = pd.DataFrame([0.2, 0.3, 0.5], index=dates, columns=stocks)

        result = factor.compute(close=close, vwap=vwap, turnover=turnover, T=3)

        assert result.iloc[2, 0] == pytest.approx(0.0, abs=1e-15)

    def test_T1_single_day(self, factor):
        """T=1 时只有 n=0, w_0 = V_t, Loss = loss_t。

        close=9, vwap=10, turnover=0.3
        daily_loss = min(0, (9-10)/10) = -0.1
        w_0 = 0.3, k = 0.3
        Loss = 0.3 * (-0.1) / 0.3 = -0.1
        """
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A"]

        close = pd.DataFrame([9.0], index=dates, columns=stocks)
        vwap = pd.DataFrame([10.0], index=dates, columns=stocks)
        turnover = pd.DataFrame([0.3], index=dates, columns=stocks)

        result = factor.compute(close=close, vwap=vwap, turnover=turnover, T=1)

        assert result.iloc[0, 0] == pytest.approx(-0.1, rel=1e-10)


class TestLossSellingEdgeCases:
    def test_nan_in_turnover(self, factor):
        """turnover 中含 NaN 时, 结果不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        close = pd.DataFrame([9.0, 11.0, 13.0], index=dates, columns=stocks)
        vwap = pd.DataFrame([10.0, 12.0, 14.0], index=dates, columns=stocks)
        turnover = pd.DataFrame([np.nan, 0.3, 0.5], index=dates, columns=stocks)

        result = factor.compute(close=close, vwap=vwap, turnover=turnover, T=3)

        assert isinstance(result.iloc[2, 0], float)

    def test_zero_turnover(self, factor):
        """换手率全为 0 时, 所有权重为 0, k=0, 因子值应为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        close = pd.DataFrame([9.0, 11.0, 13.0], index=dates, columns=stocks)
        vwap = pd.DataFrame([10.0, 12.0, 14.0], index=dates, columns=stocks)
        turnover = pd.DataFrame([0.0, 0.0, 0.0], index=dates, columns=stocks)

        result = factor.compute(close=close, vwap=vwap, turnover=turnover, T=3)
        assert np.isnan(result.iloc[2, 0])

    def test_zero_vwap(self, factor):
        """vwap 为 0 时, daily_loss 计算会产生 inf/nan, 不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        close = pd.DataFrame([9.0, 11.0, 13.0], index=dates, columns=stocks)
        vwap = pd.DataFrame([0.0, 0.0, 0.0], index=dates, columns=stocks)
        turnover = pd.DataFrame([0.2, 0.3, 0.5], index=dates, columns=stocks)

        result = factor.compute(close=close, vwap=vwap, turnover=turnover, T=3)
        assert result.shape == (3, 1)

    def test_factor_values_are_non_positive(self, factor):
        """因子值应始终 <= 0 (亏损部分被 clip 到 max=0)。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        np.random.seed(42)
        close = pd.DataFrame(
            np.random.uniform(8, 15, (5, 1)), index=dates, columns=stocks
        )
        vwap = pd.DataFrame(
            np.random.uniform(8, 15, (5, 1)), index=dates, columns=stocks
        )
        turnover = pd.DataFrame(
            np.random.uniform(0.01, 0.3, (5, 1)), index=dates, columns=stocks
        )

        result = factor.compute(close=close, vwap=vwap, turnover=turnover, T=3)

        valid = result.dropna()
        assert (valid.values <= 1e-15).all()


class TestLossSellingOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=80, freq="D")
        stocks = ["A", "B", "C"]

        close = pd.DataFrame(
            np.random.uniform(10, 50, (80, 3)), index=dates, columns=stocks
        )
        vwap = close * np.random.uniform(0.99, 1.01, (80, 3))
        turnover = pd.DataFrame(
            np.random.uniform(0.01, 0.1, (80, 3)), index=dates, columns=stocks
        )

        result = factor.compute(close=close, vwap=vwap, turnover=turnover, T=60)

        assert result.shape == close.shape
        assert list(result.columns) == list(close.columns)
        assert list(result.index) == list(close.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        close = pd.DataFrame([9.0, 10.0, 11.0, 12.0, 13.0], index=dates, columns=stocks)
        vwap = pd.DataFrame([10.0, 11.0, 12.0, 13.0, 14.0], index=dates, columns=stocks)
        turnover = pd.DataFrame([0.05] * 5, index=dates, columns=stocks)

        result = factor.compute(close=close, vwap=vwap, turnover=turnover, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        """前 T-1 行应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        T = 5

        close = pd.DataFrame(
            np.random.uniform(10, 50, (10, 2)), index=dates, columns=stocks
        )
        vwap = close * np.random.uniform(0.99, 1.01, (10, 2))
        turnover = pd.DataFrame(
            np.random.uniform(0.01, 0.1, (10, 2)), index=dates, columns=stocks
        )

        result = factor.compute(close=close, vwap=vwap, turnover=turnover, T=T)

        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
