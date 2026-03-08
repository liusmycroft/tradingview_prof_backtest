import numpy as np
import pandas as pd
import pytest

from factors.overnight_gap import OvernightGapFactor


@pytest.fixture
def factor():
    return OvernightGapFactor()


class TestOvernightGapMetadata:
    def test_name(self, factor):
        assert factor.name == "OVERNIGHT_GAP"

    def test_category(self, factor):
        assert factor.category == "高频动量反转"

    def test_repr(self, factor):
        assert "OVERNIGHT_GAP" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "OVERNIGHT_GAP"
        assert meta["category"] == "高频动量反转"


class TestOvernightGapHandCalculated:
    """用手算数据验证隔夜跳空因子。"""

    def test_known_values_T3(self, factor):
        """T=3, 已知数据验证。

        open/prev_close = [10.5/10.0, 11.0/10.5, 10.0/11.0]
        |ln(10.5/10.0)| = ln(1.05)
        |ln(11.0/10.5)| = ln(11/10.5)
        |ln(10.0/11.0)| = ln(11/10)

        min_periods=1 rolling sum:
          day0: gap0
          day1: gap0 + gap1
          day2: gap0 + gap1 + gap2
        """
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        open_price = pd.DataFrame([10.5, 11.0, 10.0], index=dates, columns=stocks)
        prev_close = pd.DataFrame([10.0, 10.5, 11.0], index=dates, columns=stocks)

        result = factor.compute(open_price=open_price, prev_close=prev_close, T=3)

        gap0 = abs(np.log(10.5 / 10.0))
        gap1 = abs(np.log(11.0 / 10.5))
        gap2 = abs(np.log(10.0 / 11.0))

        assert result.iloc[0, 0] == pytest.approx(gap0, rel=1e-10)
        assert result.iloc[1, 0] == pytest.approx(gap0 + gap1, rel=1e-10)
        assert result.iloc[2, 0] == pytest.approx(gap0 + gap1 + gap2, rel=1e-10)

    def test_no_gap(self, factor):
        """开盘价等于前收盘价时, 隔夜跳空为零。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        prices = pd.DataFrame(10.0, index=dates, columns=stocks)

        result = factor.compute(open_price=prices, prev_close=prices, T=3)

        np.testing.assert_allclose(result["A"].values, [0.0] * 5, atol=1e-15)

    def test_constant_gap_rolling_sum(self, factor):
        """每天相同跳空, 验证滚动求和。

        open=11, prev_close=10 => gap = ln(1.1)
        T=3, min_periods=1:
          day0: gap
          day1: 2*gap
          day2: 3*gap
          day3: 3*gap (窗口滑动)
          day4: 3*gap
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        open_price = pd.DataFrame(11.0, index=dates, columns=stocks)
        prev_close = pd.DataFrame(10.0, index=dates, columns=stocks)

        result = factor.compute(open_price=open_price, prev_close=prev_close, T=3)

        gap = abs(np.log(11.0 / 10.0))
        expected = [gap, 2 * gap, 3 * gap, 3 * gap, 3 * gap]
        np.testing.assert_allclose(result["A"].values, expected, rtol=1e-10)

    def test_two_stocks(self, factor):
        """两只股票并行计算。"""
        dates = pd.date_range("2024-01-01", periods=2, freq="D")
        stocks = ["A", "B"]

        open_price = pd.DataFrame(
            [[10.5, 20.0], [11.0, 22.0]], index=dates, columns=stocks
        )
        prev_close = pd.DataFrame(
            [[10.0, 21.0], [10.5, 20.0]], index=dates, columns=stocks
        )

        result = factor.compute(open_price=open_price, prev_close=prev_close, T=2)

        gap_a0 = abs(np.log(10.5 / 10.0))
        gap_a1 = abs(np.log(11.0 / 10.5))
        gap_b0 = abs(np.log(20.0 / 21.0))
        gap_b1 = abs(np.log(22.0 / 20.0))

        assert result.loc[dates[1], "A"] == pytest.approx(gap_a0 + gap_a1, rel=1e-10)
        assert result.loc[dates[1], "B"] == pytest.approx(gap_b0 + gap_b1, rel=1e-10)


class TestOvernightGapEdgeCases:
    def test_zero_prev_close(self, factor):
        """前收盘价为零时, replace(0, nan) 使比率为 NaN, rolling(min_periods=1) 跳过。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        open_price = pd.DataFrame([10.0, 11.0, 12.0], index=dates, columns=stocks)
        prev_close = pd.DataFrame([10.0, 0.0, 11.0], index=dates, columns=stocks)

        result = factor.compute(open_price=open_price, prev_close=prev_close, T=3)

        # day0: |ln(10/10)| = 0
        # day1: NaN (zero prev_close)
        # day2: |ln(12/11)|
        # rolling sum with min_periods=1 skips NaN:
        # day1: sum([0, NaN]) = 0.0
        assert result.iloc[1, 0] == pytest.approx(0.0, abs=1e-12)
        # day2: sum([0, NaN, |ln(12/11)|]) = |ln(12/11)|
        expected_day2 = abs(np.log(12.0 / 11.0))
        assert result.iloc[2, 0] == pytest.approx(expected_day2, rel=1e-10)

    def test_nan_in_open_price(self, factor):
        """open_price 中含 NaN 时, 不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        open_price = pd.DataFrame([10.0, np.nan, 12.0], index=dates, columns=stocks)
        prev_close = pd.DataFrame([10.0, 10.0, 10.0], index=dates, columns=stocks)

        result = factor.compute(open_price=open_price, prev_close=prev_close, T=3)
        assert isinstance(result.iloc[2, 0], float)

    def test_result_non_negative(self, factor):
        """隔夜跳空因子值应非负（绝对值之和）。"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A"]

        prev_close = pd.DataFrame(
            np.random.uniform(10, 20, (30, 1)), index=dates, columns=stocks
        )
        open_price = prev_close * (1 + np.random.randn(30, 1) * 0.05)

        result = factor.compute(open_price=open_price, prev_close=prev_close, T=20)
        assert (result.dropna().values >= 0).all()

    def test_gap_down_same_as_gap_up(self, factor):
        """跳空向下和跳空向上的绝对值相同时, 因子值相同。"""
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["UP", "DOWN"]

        open_price = pd.DataFrame([[11.0, 10.0]], index=dates, columns=stocks)
        prev_close = pd.DataFrame([[10.0, 11.0]], index=dates, columns=stocks)

        result = factor.compute(open_price=open_price, prev_close=prev_close, T=1)

        assert result.loc[dates[0], "UP"] == pytest.approx(
            result.loc[dates[0], "DOWN"], rel=1e-10
        )


class TestOvernightGapOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        stocks = ["A", "B", "C"]

        open_price = pd.DataFrame(
            np.random.uniform(10, 20, (50, 3)), index=dates, columns=stocks
        )
        prev_close = pd.DataFrame(
            np.random.uniform(10, 20, (50, 3)), index=dates, columns=stocks
        )

        result = factor.compute(open_price=open_price, prev_close=prev_close, T=20)

        assert result.shape == open_price.shape
        assert list(result.columns) == list(open_price.columns)
        assert list(result.index) == list(open_price.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        open_price = pd.DataFrame(11.0, index=dates, columns=stocks)
        prev_close = pd.DataFrame(10.0, index=dates, columns=stocks)

        result = factor.compute(open_price=open_price, prev_close=prev_close, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_min_periods_1_all_rows_have_values(self, factor):
        """min_periods=1, 所以从第一行起就有值（假设无 NaN 输入）。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]

        open_price = pd.DataFrame(
            np.random.uniform(10, 20, (10, 2)), index=dates, columns=stocks
        )
        prev_close = pd.DataFrame(
            np.random.uniform(10, 20, (10, 2)), index=dates, columns=stocks
        )

        result = factor.compute(open_price=open_price, prev_close=prev_close, T=5)

        assert result.notna().all().all()
