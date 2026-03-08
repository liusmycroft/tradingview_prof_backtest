import numpy as np
import pandas as pd
import pytest

from factors.ask_depth import AskDepthFactor


@pytest.fixture
def factor():
    return AskDepthFactor()


class TestAskDepthMetadata:
    def test_name(self, factor):
        assert factor.name == "ASK_DEPTH"

    def test_category(self, factor):
        assert factor.category == "高频流动性"

    def test_repr(self, factor):
        assert "ASK_DEPTH" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "ASK_DEPTH"
        assert meta["category"] == "高频流动性"


class TestAskDepthHandCalculated:
    """用手算数据验证 EWM(span=T, min_periods=T) 计算的正确性。"""

    def test_constant_input(self, factor):
        """常数输入时, EMA 应等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily_ask_depth = pd.DataFrame(5.0, index=dates, columns=stocks)

        result = factor.compute(daily_ask_depth=daily_ask_depth, T=20)

        valid = result.dropna()
        np.testing.assert_array_almost_equal(valid["A"].values, 5.0)

    def test_ema_weights_recent_more(self, factor):
        """EMA 应对近期数据赋予更高权重。

        前 20 天为 0, 后 5 天为 100 -> 最后一天 EMA 在 0 和 100 之间。
        """
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        values = [0.0] * 20 + [100.0] * 5
        daily_ask_depth = pd.DataFrame(values, index=dates, columns=stocks)

        result = factor.compute(daily_ask_depth=daily_ask_depth, T=20)

        last_val = result.iloc[-1, 0]
        assert 0 < last_val < 100

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算 EMA。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A", "B"]
        daily_ask_depth = pd.DataFrame(
            {"A": [10.0] * 25, "B": [20.0] * 25}, index=dates
        )

        result = factor.compute(daily_ask_depth=daily_ask_depth, T=20)

        valid = result.dropna()
        np.testing.assert_array_almost_equal(valid["A"].values, 10.0)
        np.testing.assert_array_almost_equal(valid["B"].values, 20.0)

    def test_ema_manual_T3(self, factor):
        """T=3, 手动验证 EMA 值。

        ewm(span=3) -> alpha = 2/(3+1) = 0.5
        data = [10, 20, 30, 40]
        min_periods=3, 所以前 2 行为 NaN。
        row 2: EMA 初始值 = mean([10, 20, 30]) 的 ewm 计算
        pandas ewm 从第一个非 NaN 开始递推:
          ema_0 = 10
          ema_1 = 0.5 * 20 + 0.5 * 10 = 15
          ema_2 = 0.5 * 30 + 0.5 * 15 = 22.5
          ema_3 = 0.5 * 40 + 0.5 * 22.5 = 31.25
        但 min_periods=3, 所以 row 0, 1 为 NaN。
        """
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        stocks = ["A"]
        daily_ask_depth = pd.DataFrame(
            [10.0, 20.0, 30.0, 40.0], index=dates, columns=stocks
        )

        result = factor.compute(daily_ask_depth=daily_ask_depth, T=3)

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(24.285714285714285, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(32.666666666666667, rel=1e-10)


class TestAskDepthEdgeCases:
    def test_nan_in_input(self, factor):
        """输入含 NaN 时, EMA 应正确处理 (pandas ewm 跳过 NaN)。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        values = np.ones(25) * 10.0
        values[10] = np.nan
        daily_ask_depth = pd.DataFrame(values, index=dates, columns=stocks)

        result = factor.compute(daily_ask_depth=daily_ask_depth, T=20)
        assert result.shape == (25, 1)

    def test_all_nan(self, factor):
        """全 NaN 输入时, 结果应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily_ask_depth = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_ask_depth=daily_ask_depth, T=20)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        """全零输入时, EMA 应全为 0。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily_ask_depth = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_ask_depth=daily_ask_depth, T=20)
        valid = result.dropna()
        for val in valid["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)


class TestAskDepthOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily_ask_depth = pd.DataFrame(
            np.random.uniform(1e5, 1e6, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_ask_depth=daily_ask_depth, T=20)

        assert result.shape == daily_ask_depth.shape
        assert list(result.columns) == list(daily_ask_depth.columns)
        assert list(result.index) == list(daily_ask_depth.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily_ask_depth = pd.DataFrame([1.0] * 25, index=dates, columns=stocks)

        result = factor.compute(daily_ask_depth=daily_ask_depth, T=20)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        """min_periods=T, 前 T-1 行应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        T = 20
        daily_ask_depth = pd.DataFrame(
            np.random.uniform(1, 100, (30, 2)), index=dates, columns=stocks
        )

        result = factor.compute(daily_ask_depth=daily_ask_depth, T=T)

        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
