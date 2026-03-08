import numpy as np
import pandas as pd
import pytest

from factors.net_support_volume import NetSupportVolumeFactor


@pytest.fixture
def factor():
    return NetSupportVolumeFactor()


class TestNetSupportVolumeMetadata:
    def test_name(self, factor):
        assert factor.name == "NET_SUPPORT_VOLUME"

    def test_category(self, factor):
        assert factor.category == "量价因子改进"

    def test_repr(self, factor):
        assert "NET_SUPPORT_VOLUME" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "NET_SUPPORT_VOLUME"
        assert meta["category"] == "量价因子改进"


class TestNetSupportVolumeHandCalculated:
    def test_constant_input(self, factor):
        """常数输入时, 滚动均值应等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(0.05, index=dates, columns=stocks)

        result = factor.compute(daily_net_support_volume=daily, T=3)

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(0.05, rel=1e-10)

    def test_varying_T3(self, factor):
        """T=3, 变化的值。

        vals = [0.01, 0.02, 0.03]
        rolling(3): mean = 0.02
        """
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame([0.01, 0.02, 0.03], index=dates, columns=stocks)

        result = factor.compute(daily_net_support_volume=daily, T=3)
        assert result.iloc[2, 0] == pytest.approx(0.02, rel=1e-10)

    def test_negative_values(self, factor):
        """负值（阻力大于支撑）应正确计算。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(-0.03, index=dates, columns=stocks)

        result = factor.compute(daily_net_support_volume=daily, T=3)
        assert result.iloc[2, 0] == pytest.approx(-0.03, rel=1e-10)

    def test_rolling_window_slides(self, factor):
        """验证滚动窗口正确滑动 (T=3)。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0], index=dates, columns=stocks)

        result = factor.compute(daily_net_support_volume=daily, T=3)
        assert result.iloc[2, 0] == pytest.approx(2.0, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(3.0, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(4.0, rel=1e-10)

    def test_two_stocks(self, factor):
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A", "B"]
        daily = pd.DataFrame(
            {"A": [0.01, 0.02, 0.03], "B": [-0.01, -0.02, -0.03]}, index=dates
        )

        result = factor.compute(daily_net_support_volume=daily, T=3)
        assert result.loc[dates[2], "A"] == pytest.approx(0.02, rel=1e-10)
        assert result.loc[dates[2], "B"] == pytest.approx(-0.02, rel=1e-10)


class TestNetSupportVolumeEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame([0.01, np.nan, 0.03], index=dates, columns=stocks)

        result = factor.compute(daily_net_support_volume=daily, T=3)
        assert np.isnan(result.iloc[2, 0])

    def test_all_zero(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_net_support_volume=daily, T=3)
        assert result.iloc[2, 0] == pytest.approx(0.0, abs=1e-15)

    def test_insufficient_data(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(0.01, index=dates, columns=stocks)

        result = factor.compute(daily_net_support_volume=daily, T=20)
        assert result.isna().all().all()


class TestNetSupportVolumeOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        stocks = ["A", "B", "C"]
        daily = pd.DataFrame(
            np.random.uniform(-0.1, 0.1, (50, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_net_support_volume=daily, T=20)
        assert result.shape == daily.shape
        assert list(result.columns) == list(daily.columns)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(0.01, index=dates, columns=stocks)

        result = factor.compute(daily_net_support_volume=daily, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        T = 5
        daily = pd.DataFrame(
            np.random.uniform(-0.1, 0.1, (10, 2)), index=dates, columns=stocks
        )

        result = factor.compute(daily_net_support_volume=daily, T=T)
        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
