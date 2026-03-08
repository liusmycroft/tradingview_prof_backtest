import numpy as np
import pandas as pd
import pytest

from factors.closing_return import ClosingReturnFactor


@pytest.fixture
def factor():
    return ClosingReturnFactor()


class TestClosingReturnMetadata:
    def test_name(self, factor):
        assert factor.name == "CLOSING_RETURN"

    def test_category(self, factor):
        assert factor.category == "高频动量反转"

    def test_repr(self, factor):
        assert "CLOSING_RETURN" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "CLOSING_RETURN"
        assert meta["category"] == "高频动量反转"


class TestClosingReturnHandCalculated:
    """用手算数据验证 rolling(T).sum() 的正确性。"""

    def test_simple_T3(self, factor):
        """T=3 滚动求和手算验证。

        close_1430 = [100, 100, 100, 100, 100]
        close_1500 = [101, 99, 102, 98, 103]
        daily_ret  = [0.01, -0.01, 0.02, -0.02, 0.03]

        T=3:
          row 0,1: NaN (min_periods=3)
          row 2: 0.01 + (-0.01) + 0.02 = 0.02
          row 3: (-0.01) + 0.02 + (-0.02) = -0.01
          row 4: 0.02 + (-0.02) + 0.03 = 0.03
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        close_1430 = pd.DataFrame([100.0] * 5, index=dates, columns=stocks)
        close_1500 = pd.DataFrame(
            [101.0, 99.0, 102.0, 98.0, 103.0], index=dates, columns=stocks
        )

        result = factor.compute(close_1430=close_1430, close_1500=close_1500, T=3)

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(0.02, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(-0.01, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(0.03, rel=1e-10)

    def test_constant_return(self, factor):
        """每日尾盘收益率恒定时, T 日求和 = T * daily_ret。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        close_1430 = pd.DataFrame([100.0] * 5, index=dates, columns=stocks)
        close_1500 = pd.DataFrame([101.0] * 5, index=dates, columns=stocks)

        result = factor.compute(close_1430=close_1430, close_1500=close_1500, T=3)

        expected = 3 * 0.01
        assert result.iloc[2, 0] == pytest.approx(expected, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(expected, rel=1e-10)

    def test_two_stocks(self, factor):
        """两只股票并行计算。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A", "B"]

        close_1430 = pd.DataFrame(
            [[100.0, 200.0], [100.0, 200.0], [100.0, 200.0]],
            index=dates,
            columns=stocks,
        )
        close_1500 = pd.DataFrame(
            [[102.0, 198.0], [101.0, 204.0], [103.0, 196.0]],
            index=dates,
            columns=stocks,
        )

        result = factor.compute(close_1430=close_1430, close_1500=close_1500, T=3)

        # A: (102/100-1) + (101/100-1) + (103/100-1) = 0.02+0.01+0.03 = 0.06
        expected_a = (102 / 100 - 1) + (101 / 100 - 1) + (103 / 100 - 1)
        assert result.iloc[2, 0] == pytest.approx(expected_a, rel=1e-10)

        # B: (198/200-1) + (204/200-1) + (196/200-1) = -0.01+0.02+(-0.02) = -0.01
        expected_b = (198 / 200 - 1) + (204 / 200 - 1) + (196 / 200 - 1)
        assert result.iloc[2, 1] == pytest.approx(expected_b, rel=1e-10)


class TestClosingReturnEdgeCases:
    def test_nan_propagation(self, factor):
        """close_1430 含 NaN 时, 对应 daily_ret 为 NaN, 滚动窗口内含 NaN -> NaN。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        close_1430 = pd.DataFrame(
            [100.0, np.nan, 100.0, 100.0, 100.0], index=dates, columns=stocks
        )
        close_1500 = pd.DataFrame(
            [101.0, 99.0, 102.0, 98.0, 103.0], index=dates, columns=stocks
        )

        result = factor.compute(close_1430=close_1430, close_1500=close_1500, T=3)
        # 窗口 [0,1,2] 和 [1,2,3] 含 NaN -> NaN
        assert np.isnan(result.iloc[2, 0])
        assert np.isnan(result.iloc[3, 0])
        # 窗口 [2,3,4] 无 NaN -> 有值
        assert not np.isnan(result.iloc[4, 0])

    def test_zero_division(self, factor):
        """close_1430 为 0 时产生 inf, 不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        close_1430 = pd.DataFrame([0.0, 100.0, 100.0], index=dates, columns=stocks)
        close_1500 = pd.DataFrame([101.0, 101.0, 101.0], index=dates, columns=stocks)

        result = factor.compute(close_1430=close_1430, close_1500=close_1500, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_all_nan(self, factor):
        """全 NaN 输入时, 结果应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        close_1430 = pd.DataFrame(np.nan, index=dates, columns=stocks)
        close_1500 = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(close_1430=close_1430, close_1500=close_1500, T=3)
        assert result.isna().all().all()


class TestClosingReturnOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        close_1430 = pd.DataFrame(
            np.random.uniform(10, 50, (30, 2)), index=dates, columns=stocks
        )
        close_1500 = close_1430 * (1 + np.random.normal(0, 0.005, (30, 2)))

        result = factor.compute(close_1430=close_1430, close_1500=close_1500, T=20)

        assert result.shape == close_1430.shape
        assert list(result.columns) == list(close_1430.columns)
        assert list(result.index) == list(close_1430.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        close_1430 = pd.DataFrame([100.0] * 5, index=dates, columns=stocks)
        close_1500 = pd.DataFrame([101.0] * 5, index=dates, columns=stocks)

        result = factor.compute(close_1430=close_1430, close_1500=close_1500, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        """前 T-1 行应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        T = 5
        close_1430 = pd.DataFrame([100.0] * 10, index=dates, columns=stocks)
        close_1500 = pd.DataFrame([101.0] * 10, index=dates, columns=stocks)

        result = factor.compute(close_1430=close_1430, close_1500=close_1500, T=T)

        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
