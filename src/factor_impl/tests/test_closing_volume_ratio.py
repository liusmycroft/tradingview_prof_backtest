import numpy as np
import pandas as pd
import pytest

from factors.closing_volume_ratio import ClosingVolumeRatioFactor


@pytest.fixture
def factor():
    return ClosingVolumeRatioFactor()


class TestClosingVolumeRatioMetadata:
    def test_name(self, factor):
        assert factor.name == "CLOSING_VOLUME_RATIO"

    def test_category(self, factor):
        assert factor.category == "高频成交分布"

    def test_repr(self, factor):
        assert "CLOSING_VOLUME_RATIO" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "CLOSING_VOLUME_RATIO"
        assert meta["category"] == "高频成交分布"


class TestClosingVolumeRatioHandCalculated:
    """用手算数据验证 (closing_volume / daily_volume).rolling(T).mean() 的正确性。"""

    def test_constant_ratio(self, factor):
        """尾盘 200, 全天 1000 -> 占比恒为 0.2, 均值也为 0.2。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        closing = pd.DataFrame(200.0, index=dates, columns=stocks)
        daily = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(closing_volume=closing, daily_volume=daily, T=20)
        assert result.iloc[-1, 0] == pytest.approx(0.2, rel=1e-6)

    def test_varying_ratio_T3(self, factor):
        """T=3, 手动验证不同占比的滚动均值。

        closing = [100, 200, 300, 400, 500]
        daily   = [1000, 1000, 1000, 1000, 1000]
        ratio   = [0.1, 0.2, 0.3, 0.4, 0.5]

        T=3:
          row 0,1: NaN
          row 2: mean(0.1, 0.2, 0.3) = 0.2
          row 3: mean(0.2, 0.3, 0.4) = 0.3
          row 4: mean(0.3, 0.4, 0.5) = 0.4
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        closing = pd.DataFrame(
            [100.0, 200.0, 300.0, 400.0, 500.0], index=dates, columns=stocks
        )
        daily = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(closing_volume=closing, daily_volume=daily, T=3)

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(0.2, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(0.3, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(0.4, rel=1e-10)

    def test_mixed_ratio_T20(self, factor):
        """前 10 天占比 0.2, 后 10 天占比 0.6, T=20 均值 = 0.4。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        closing_vals = [100.0] * 10 + [300.0] * 10
        closing = pd.DataFrame(closing_vals, index=dates, columns=stocks)
        daily = pd.DataFrame(500.0, index=dates, columns=stocks)

        result = factor.compute(closing_volume=closing, daily_volume=daily, T=20)
        expected = (0.2 * 10 + 0.6 * 10) / 20
        assert result.iloc[-1, 0] == pytest.approx(expected, rel=1e-6)

    def test_two_stocks(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A", "B"]
        closing = pd.DataFrame(
            {"A": [200.0] * 25, "B": [400.0] * 25}, index=dates
        )
        daily = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(closing_volume=closing, daily_volume=daily, T=20)
        assert result.iloc[-1, 0] == pytest.approx(0.2, rel=1e-6)
        assert result.iloc[-1, 1] == pytest.approx(0.4, rel=1e-6)


class TestClosingVolumeRatioEdgeCases:
    def test_zero_daily_volume(self, factor):
        """daily_volume 为 0 时产生 inf/NaN, 不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        closing = pd.DataFrame([100.0] * 5, index=dates, columns=stocks)
        daily = pd.DataFrame([0.0, 1000.0, 1000.0, 1000.0, 1000.0], index=dates, columns=stocks)

        result = factor.compute(closing_volume=closing, daily_volume=daily, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_nan_in_input(self, factor):
        """输入含 NaN 时, 不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        closing = pd.DataFrame([100.0, np.nan, 100.0, 100.0, 100.0], index=dates, columns=stocks)
        daily = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(closing_volume=closing, daily_volume=daily, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_insufficient_window_returns_nan(self, factor):
        """数据不足 T 天时应返回 NaN。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        closing = pd.DataFrame(200.0, index=dates, columns=stocks)
        daily = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(closing_volume=closing, daily_volume=daily, T=20)
        assert result.isna().all().all()


class TestClosingVolumeRatioOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        closing = pd.DataFrame(
            np.random.uniform(100, 500, (30, 3)), index=dates, columns=stocks
        )
        daily = pd.DataFrame(
            np.random.uniform(500, 2000, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(closing_volume=closing, daily_volume=daily, T=20)

        assert result.shape == closing.shape
        assert list(result.columns) == list(closing.columns)
        assert list(result.index) == list(closing.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        closing = pd.DataFrame(200.0, index=dates, columns=stocks)
        daily = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(closing_volume=closing, daily_volume=daily, T=20)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        """前 T-1 行应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        T = 20
        closing = pd.DataFrame(200.0, index=dates, columns=stocks)
        daily = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(closing_volume=closing, daily_volume=daily, T=T)

        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
