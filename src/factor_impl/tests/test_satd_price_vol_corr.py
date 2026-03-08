import numpy as np
import pandas as pd
import pytest

from factors.satd_price_vol_corr import SatdPriceVolCorrFactor


@pytest.fixture
def factor():
    return SatdPriceVolCorrFactor()


class TestSatdPriceVolCorrMetadata:
    def test_name(self, factor):
        assert factor.name == "SATD_PRICE_VOL_CORR"

    def test_category(self, factor):
        assert factor.category == "高频成交分布"

    def test_repr(self, factor):
        assert "SATD_PRICE_VOL_CORR" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "SATD_PRICE_VOL_CORR"
        assert meta["category"] == "高频成交分布"


class TestSatdPriceVolCorrHandCalculated:
    """用手算数据验证 EWM(span=T, min_periods=1) 计算的正确性。"""

    def test_constant_input(self, factor):
        """常数输入时, EMA 应等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        daily_satd = pd.DataFrame(1.2, index=dates, columns=stocks)

        result = factor.compute(daily_satd=daily_satd, T=20)

        np.testing.assert_array_almost_equal(result["A"].values, 1.2)

    def test_ema_manual_T3(self, factor):
        """T=3, 手动验证 EMA 值。

        ewm(span=3, adjust=True) alpha = 2/(3+1) = 0.5
        data = [1.0, 1.5, 2.0, 2.5]
        ema_0 = 1.0
        ema_1 = (0.5*1.0 + 1.0*1.5) / (0.5+1.0) = 2.0/1.5 = 4/3
        ema_2 = (0.25*1.0 + 0.5*1.5 + 1.0*2.0) / (0.25+0.5+1.0) = 3.0/1.75 = 12/7
        ema_3 = (0.125*1.0 + 0.25*1.5 + 0.5*2.0 + 1.0*2.5) / (0.125+0.25+0.5+1.0) = 4.0/1.875 = 32/15
        """
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        stocks = ["A"]
        daily_satd = pd.DataFrame(
            [1.0, 1.5, 2.0, 2.5], index=dates, columns=stocks
        )

        result = factor.compute(daily_satd=daily_satd, T=3)

        assert result.iloc[0, 0] == pytest.approx(1.0, rel=1e-6)
        assert result.iloc[1, 0] == pytest.approx(4.0 / 3, rel=1e-6)
        assert result.iloc[2, 0] == pytest.approx(12.0 / 7, rel=1e-6)
        assert result.iloc[3, 0] == pytest.approx(32.0 / 15, rel=1e-6)

    def test_ema_recent_weight(self, factor):
        """EMA 应赋予近期数据更高权重。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        vals = [0.8] * 5 + [1.5] * 5
        daily_satd = pd.DataFrame(vals, index=dates, columns=stocks)

        result = factor.compute(daily_satd=daily_satd, T=5)
        assert result.iloc[-1, 0] > 1.15

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        daily_satd = pd.DataFrame(
            {"A": [1.0] * 10, "B": [2.0] * 10}, index=dates
        )

        result = factor.compute(daily_satd=daily_satd, T=5)

        np.testing.assert_array_almost_equal(result["A"].values, 1.0)
        np.testing.assert_array_almost_equal(result["B"].values, 2.0)


class TestSatdPriceVolCorrEdgeCases:
    def test_single_value(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A"]
        daily_satd = pd.DataFrame([1.3], index=dates, columns=stocks)

        result = factor.compute(daily_satd=daily_satd, T=20)
        assert result.iloc[0, 0] == pytest.approx(1.3, rel=1e-10)

    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        values = np.ones(10) * 1.1
        values[3] = np.nan
        daily_satd = pd.DataFrame(values, index=dates, columns=stocks)

        result = factor.compute(daily_satd=daily_satd, T=5)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily_satd = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_satd=daily_satd, T=5)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily_satd = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_satd=daily_satd, T=5)
        for val in result["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)


class TestSatdPriceVolCorrOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily_satd = pd.DataFrame(
            np.random.uniform(0.5, 2.0, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_satd=daily_satd, T=20)

        assert result.shape == daily_satd.shape
        assert list(result.columns) == list(daily_satd.columns)
        assert list(result.index) == list(daily_satd.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_satd = pd.DataFrame([1.0, 1.1, 1.2, 1.3, 1.4], index=dates, columns=stocks)

        result = factor.compute(daily_satd=daily_satd, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_no_leading_nan_min_periods_1(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        daily_satd = pd.DataFrame(
            np.random.uniform(0.5, 2.0, (10, 2)), index=dates, columns=stocks
        )

        result = factor.compute(daily_satd=daily_satd, T=20)
        assert result.iloc[0].notna().all()
        assert result.notna().all().all()
