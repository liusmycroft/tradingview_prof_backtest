import numpy as np
import pandas as pd
import pytest

from factors.cgo import CGOFactor


@pytest.fixture
def factor():
    return CGOFactor()


class TestCGOMetadata:
    def test_name(self, factor):
        assert factor.name == "CGO"

    def test_category(self, factor):
        assert factor.category == "行为金融"

    def test_repr(self, factor):
        assert "CGO" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "CGO"
        assert meta["category"] == "行为金融"


class TestCGOHandCalculated:
    """用手算数据验证权重计算和 CGO 公式的正确性。"""

    def test_T3_single_stock(self, factor):
        """T=3, 单只股票, 手动计算验证。

        设 3 天数据 (day 0, 1, 2):
          turnover = [0.2, 0.3, 0.5]
          vwap     = [10,  12,  14]
          close    = [10,  12,  15]

        在 t=2 (最后一天), T=3, 窗口 = [day0, day1, day2]:
          n=0: day2, V_{t}   = 0.5,  w_0 = 0.5
          n=1: day1, V_{t-1} = 0.3,  w_1 = 0.3  (prod_{s=1}^{0} = 1)
          n=2: day0, V_{t-2} = 0.2,  w_2 = 0.2 * (1 - V_{t-1}) = 0.2 * 0.7 = 0.14

        k = 0.5 + 0.3 + 0.14 = 0.94
        RP = (0.5*14 + 0.3*12 + 0.14*10) / 0.94
           = (7 + 3.6 + 1.4) / 0.94
           = 12.0 / 0.94
           = 12.765957...
        CGO = (15 - 12.765957...) / 15 = 2.234042... / 15 = 0.148936...
        """
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        close = pd.DataFrame([10.0, 12.0, 15.0], index=dates, columns=stocks)
        vwap = pd.DataFrame([10.0, 12.0, 14.0], index=dates, columns=stocks)
        turnover = pd.DataFrame([0.2, 0.3, 0.5], index=dates, columns=stocks)

        result = factor.compute(close=close, vwap=vwap, turnover=turnover, T=3)

        # 前 T-1=2 天应为 NaN
        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])

        expected_rp = (0.5 * 14 + 0.3 * 12 + 0.14 * 10) / 0.94
        expected_cgo = (15.0 - expected_rp) / 15.0
        assert result.iloc[2, 0] == pytest.approx(expected_cgo, rel=1e-10)

    def test_T2_two_stocks(self, factor):
        """T=2, 两只股票, 验证多列并行计算。

        Stock A:                    Stock B:
          turnover = [0.1, 0.4]       turnover = [0.5, 0.2]
          vwap     = [20,  22]        vwap     = [30,  28]
          close    = [20,  25]        close    = [30,  26]

        t=1, T=2:
        Stock A:
          n=0: w_0 = 0.4,  n=1: w_1 = 0.1
          k = 0.5
          RP = (0.4*22 + 0.1*20) / 0.5 = (8.8 + 2.0) / 0.5 = 21.6
          CGO = (25 - 21.6) / 25 = 0.136

        Stock B:
          n=0: w_0 = 0.2,  n=1: w_1 = 0.5
          k = 0.7
          RP = (0.2*28 + 0.5*30) / 0.7 = (5.6 + 15.0) / 0.7 = 29.4285714...
          CGO = (26 - 29.4285714...) / 26 = -0.131868...
        """
        dates = pd.date_range("2024-01-01", periods=2, freq="D")
        stocks = ["A", "B"]

        close = pd.DataFrame([[20.0, 30.0], [25.0, 26.0]], index=dates, columns=stocks)
        vwap = pd.DataFrame([[20.0, 30.0], [22.0, 28.0]], index=dates, columns=stocks)
        turnover = pd.DataFrame([[0.1, 0.5], [0.4, 0.2]], index=dates, columns=stocks)

        result = factor.compute(close=close, vwap=vwap, turnover=turnover, T=2)

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[0, 1])

        # Stock A
        expected_rp_a = (0.4 * 22 + 0.1 * 20) / 0.5
        expected_cgo_a = (25.0 - expected_rp_a) / 25.0
        assert result.loc[dates[1], "A"] == pytest.approx(expected_cgo_a, rel=1e-10)

        # Stock B
        expected_rp_b = (0.2 * 28 + 0.5 * 30) / 0.7
        expected_cgo_b = (26.0 - expected_rp_b) / 26.0
        assert result.loc[dates[1], "B"] == pytest.approx(expected_cgo_b, rel=1e-10)

    def test_uniform_turnover(self, factor):
        """换手率恒定时, 权重呈几何衰减。

        turnover 恒为 0.1, T=4, 4 天数据:
          w_0 = 0.1
          w_1 = 0.1
          w_2 = 0.1 * (1 - 0.1) = 0.09
          w_3 = 0.1 * (1 - 0.1)^2 = 0.081
        """
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        stocks = ["X"]

        close = pd.DataFrame([100.0, 102.0, 104.0, 106.0], index=dates, columns=stocks)
        vwap = pd.DataFrame([100.0, 101.0, 103.0, 105.0], index=dates, columns=stocks)
        turnover = pd.DataFrame([0.1, 0.1, 0.1, 0.1], index=dates, columns=stocks)

        result = factor.compute(close=close, vwap=vwap, turnover=turnover, T=4)

        w = np.array([0.1, 0.1, 0.09, 0.081])
        vwap_rev = np.array([105.0, 103.0, 101.0, 100.0])
        k = w.sum()
        rp = (w * vwap_rev).sum() / k
        expected_cgo = (106.0 - rp) / 106.0

        assert result.iloc[3, 0] == pytest.approx(expected_cgo, rel=1e-10)


class TestCGOEdgeCases:
    def test_single_day_T1(self, factor):
        """T=1 时只有 n=0, w_0 = V_t, RP = VWAP_t, CGO = (Close - VWAP) / Close."""
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A"]

        close = pd.DataFrame([50.0], index=dates, columns=stocks)
        vwap = pd.DataFrame([48.0], index=dates, columns=stocks)
        turnover = pd.DataFrame([0.05], index=dates, columns=stocks)

        result = factor.compute(close=close, vwap=vwap, turnover=turnover, T=1)

        expected = (50.0 - 48.0) / 50.0
        assert result.iloc[0, 0] == pytest.approx(expected, rel=1e-10)

    def test_nan_in_turnover(self, factor):
        """turnover 中含 NaN 时, 结果应为 NaN (nansum 会跳过, 但权重异常)。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        close = pd.DataFrame([10.0, 12.0, 15.0], index=dates, columns=stocks)
        vwap = pd.DataFrame([10.0, 12.0, 14.0], index=dates, columns=stocks)
        turnover = pd.DataFrame([np.nan, 0.3, 0.5], index=dates, columns=stocks)

        result = factor.compute(close=close, vwap=vwap, turnover=turnover, T=3)

        # 结果不应抛异常, 且应为 float
        assert isinstance(result.iloc[2, 0], float)

    def test_zero_turnover(self, factor):
        """换手率全为 0 时, 所有权重为 0, k=0, CGO 应为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        close = pd.DataFrame([10.0, 12.0, 15.0], index=dates, columns=stocks)
        vwap = pd.DataFrame([10.0, 12.0, 14.0], index=dates, columns=stocks)
        turnover = pd.DataFrame([0.0, 0.0, 0.0], index=dates, columns=stocks)

        result = factor.compute(close=close, vwap=vwap, turnover=turnover, T=3)
        assert np.isnan(result.iloc[2, 0])

    def test_close_equals_rp_gives_zero(self, factor):
        """当 close == RP 时, CGO 应为 0。

        T=1: RP = VWAP, 所以 close == vwap -> CGO = 0。
        """
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A"]

        close = pd.DataFrame([100.0], index=dates, columns=stocks)
        vwap = pd.DataFrame([100.0], index=dates, columns=stocks)
        turnover = pd.DataFrame([0.05], index=dates, columns=stocks)

        result = factor.compute(close=close, vwap=vwap, turnover=turnover, T=1)
        assert result.iloc[0, 0] == pytest.approx(0.0, abs=1e-15)


class TestCGOOutputShape:
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

        close = pd.DataFrame([10.0, 11.0, 12.0, 13.0, 14.0], index=dates, columns=stocks)
        vwap = close.copy()
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
        vwap = close.copy()
        turnover = pd.DataFrame(
            np.random.uniform(0.01, 0.1, (10, 2)), index=dates, columns=stocks
        )

        result = factor.compute(close=close, vwap=vwap, turnover=turnover, T=T)

        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
