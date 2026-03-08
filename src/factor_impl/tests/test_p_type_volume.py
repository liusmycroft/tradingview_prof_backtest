import numpy as np
import pandas as pd
import pytest

from factors.p_type_volume import PTypeVolumeFactor


@pytest.fixture
def factor():
    return PTypeVolumeFactor()


class TestPTypeVolumeMetadata:
    def test_name(self, factor):
        assert factor.name == "P_TYPE_VOLUME"

    def test_category(self, factor):
        assert factor.category == "高频成交分布"

    def test_repr(self, factor):
        assert "P_TYPE_VOLUME" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "P_TYPE_VOLUME"
        assert meta["category"] == "高频成交分布"


class TestPTypeVolumeHandCalculated:
    """用手算数据验证P型成交量分布因子。"""

    def test_constant_position(self, factor):
        """常数: vsa_low=9.5, high=11, low=9 => position=(9.5-9)/(11-9)=0.25。

        T日滚动均值 = 0.25。
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        vsa_low = pd.DataFrame(9.5, index=dates, columns=stocks)
        daily_high = pd.DataFrame(11.0, index=dates, columns=stocks)
        daily_low = pd.DataFrame(9.0, index=dates, columns=stocks)

        result = factor.compute(
            vsa_low=vsa_low, daily_high=daily_high, daily_low=daily_low, T=3
        )

        # min_periods=T=3, 前2行NaN
        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(0.25, rel=1e-10)

    def test_vsa_at_low(self, factor):
        """VSA_Low == Daily_Low => position = 0。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        vsa_low = pd.DataFrame(9.0, index=dates, columns=stocks)
        daily_high = pd.DataFrame(11.0, index=dates, columns=stocks)
        daily_low = pd.DataFrame(9.0, index=dates, columns=stocks)

        result = factor.compute(
            vsa_low=vsa_low, daily_high=daily_high, daily_low=daily_low, T=3
        )

        assert result.iloc[2, 0] == pytest.approx(0.0, abs=1e-15)

    def test_vsa_at_high(self, factor):
        """VSA_Low == Daily_High => position = 1。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        vsa_low = pd.DataFrame(11.0, index=dates, columns=stocks)
        daily_high = pd.DataFrame(11.0, index=dates, columns=stocks)
        daily_low = pd.DataFrame(9.0, index=dates, columns=stocks)

        result = factor.compute(
            vsa_low=vsa_low, daily_high=daily_high, daily_low=daily_low, T=3
        )

        assert result.iloc[2, 0] == pytest.approx(1.0, rel=1e-10)

    def test_midpoint(self, factor):
        """VSA_Low 在中点 => position = 0.5。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        vsa_low = pd.DataFrame(10.0, index=dates, columns=stocks)
        daily_high = pd.DataFrame(11.0, index=dates, columns=stocks)
        daily_low = pd.DataFrame(9.0, index=dates, columns=stocks)

        result = factor.compute(
            vsa_low=vsa_low, daily_high=daily_high, daily_low=daily_low, T=3
        )

        assert result.iloc[2, 0] == pytest.approx(0.5, rel=1e-10)

    def test_varying_position_T3(self, factor):
        """T=3, 变化的位置。

        daily_position = [0.25, 0.5, 0.75]
        rolling(3, min_periods=3):
          day2: mean([0.25, 0.5, 0.75]) = 0.5
        """
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        # position = (vsa_low - low) / (high - low)
        # high=11, low=9, range=2
        # day0: vsa_low=9.5 => 0.25
        # day1: vsa_low=10.0 => 0.5
        # day2: vsa_low=10.5 => 0.75
        vsa_low = pd.DataFrame([9.5, 10.0, 10.5], index=dates, columns=stocks)
        daily_high = pd.DataFrame(11.0, index=dates, columns=stocks)
        daily_low = pd.DataFrame(9.0, index=dates, columns=stocks)

        result = factor.compute(
            vsa_low=vsa_low, daily_high=daily_high, daily_low=daily_low, T=3
        )

        assert result.iloc[2, 0] == pytest.approx(0.5, rel=1e-10)

    def test_two_stocks(self, factor):
        """两只股票并行计算。

        Stock A: position = 0.25 (constant)
        Stock B: position = 0.75 (constant)
        """
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A", "B"]

        vsa_low = pd.DataFrame(
            {"A": [9.5] * 3, "B": [10.5] * 3}, index=dates
        )
        daily_high = pd.DataFrame(11.0, index=dates, columns=stocks)
        daily_low = pd.DataFrame(9.0, index=dates, columns=stocks)

        result = factor.compute(
            vsa_low=vsa_low, daily_high=daily_high, daily_low=daily_low, T=3
        )

        assert result.loc[dates[2], "A"] == pytest.approx(0.25, rel=1e-10)
        assert result.loc[dates[2], "B"] == pytest.approx(0.75, rel=1e-10)


class TestPTypeVolumeEdgeCases:
    def test_zero_price_range(self, factor):
        """Daily_High == Daily_Low 时, 除零产生 NaN/inf。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        vsa_low = pd.DataFrame(10.0, index=dates, columns=stocks)
        daily_high = pd.DataFrame(10.0, index=dates, columns=stocks)
        daily_low = pd.DataFrame(10.0, index=dates, columns=stocks)

        result = factor.compute(
            vsa_low=vsa_low, daily_high=daily_high, daily_low=daily_low, T=3
        )
        # 0/0 = NaN
        assert np.isnan(result.iloc[2, 0])

    def test_nan_in_vsa_low(self, factor):
        """vsa_low 中含 NaN 时, 不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        vsa_low = pd.DataFrame([9.5, np.nan, 10.5], index=dates, columns=stocks)
        daily_high = pd.DataFrame(11.0, index=dates, columns=stocks)
        daily_low = pd.DataFrame(9.0, index=dates, columns=stocks)

        result = factor.compute(
            vsa_low=vsa_low, daily_high=daily_high, daily_low=daily_low, T=3
        )
        # 窗口内有 NaN, rolling mean 结果为 NaN
        assert np.isnan(result.iloc[2, 0])

    def test_vsa_outside_range(self, factor):
        """VSA_Low 超出 [low, high] 范围时, position > 1 或 < 0。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        vsa_low = pd.DataFrame(12.0, index=dates, columns=stocks)
        daily_high = pd.DataFrame(11.0, index=dates, columns=stocks)
        daily_low = pd.DataFrame(9.0, index=dates, columns=stocks)

        result = factor.compute(
            vsa_low=vsa_low, daily_high=daily_high, daily_low=daily_low, T=3
        )
        # (12-9)/(11-9) = 1.5
        assert result.iloc[2, 0] == pytest.approx(1.5, rel=1e-10)

    def test_insufficient_data(self, factor):
        """数据不足 T 天时, 全部为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        vsa_low = pd.DataFrame(9.5, index=dates, columns=stocks)
        daily_high = pd.DataFrame(11.0, index=dates, columns=stocks)
        daily_low = pd.DataFrame(9.0, index=dates, columns=stocks)

        result = factor.compute(
            vsa_low=vsa_low, daily_high=daily_high, daily_low=daily_low, T=20
        )
        assert result.isna().all().all()


class TestPTypeVolumeOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        stocks = ["A", "B", "C"]

        daily_low = pd.DataFrame(
            np.random.uniform(5, 10, (50, 3)), index=dates, columns=stocks
        )
        daily_high = daily_low + np.random.uniform(0.5, 2, (50, 3))
        vsa_low = daily_low + np.random.uniform(0, 1, (50, 3))

        result = factor.compute(
            vsa_low=vsa_low, daily_high=daily_high, daily_low=daily_low, T=20
        )

        assert result.shape == daily_low.shape
        assert list(result.columns) == list(daily_low.columns)
        assert list(result.index) == list(daily_low.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        vsa_low = pd.DataFrame(9.5, index=dates, columns=stocks)
        daily_high = pd.DataFrame(11.0, index=dates, columns=stocks)
        daily_low = pd.DataFrame(9.0, index=dates, columns=stocks)

        result = factor.compute(
            vsa_low=vsa_low, daily_high=daily_high, daily_low=daily_low, T=3
        )
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        """前 T-1 行应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        T = 5

        vsa_low = pd.DataFrame(9.5, index=dates, columns=stocks)
        daily_high = pd.DataFrame(11.0, index=dates, columns=stocks)
        daily_low = pd.DataFrame(9.0, index=dates, columns=stocks)

        result = factor.compute(
            vsa_low=vsa_low, daily_high=daily_high, daily_low=daily_low, T=T
        )

        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
