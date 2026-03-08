import numpy as np
import pandas as pd
import pytest

from factors.peak_vol_sell_amt import PeakVolSellAmtFactor


@pytest.fixture
def factor():
    return PeakVolSellAmtFactor()


class TestPeakVolSellAmtMetadata:
    def test_name(self, factor):
        assert factor.name == "PEAK_VOL_SELL_AMT"

    def test_category(self, factor):
        assert factor.category == "高频成交分布"

    def test_repr(self, factor):
        assert "PEAK_VOL_SELL_AMT" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "PEAK_VOL_SELL_AMT"
        assert meta["category"] == "高频成交分布"


class TestPeakVolSellAmtHandCalculated:
    """手算验证 rolling(T).mean()。"""

    def test_constant_input(self, factor):
        """常数输入时, 均值等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(50000.0, index=dates, columns=stocks)

        result = factor.compute(daily_peak_sell_amt=daily, T=20)
        assert result.iloc[-1, 0] == pytest.approx(50000.0, rel=1e-10)

    def test_manual_T3(self, factor):
        """T=3, 手动验证。

        data = [10000, 20000, 30000, 40000, 50000]
        T=3:
          row 2: mean(10000, 20000, 30000) = 20000
          row 3: mean(20000, 30000, 40000) = 30000
          row 4: mean(30000, 40000, 50000) = 40000
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(
            [10000.0, 20000.0, 30000.0, 40000.0, 50000.0], index=dates, columns=stocks
        )

        result = factor.compute(daily_peak_sell_amt=daily, T=3)

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(20000.0, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(30000.0, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(40000.0, rel=1e-10)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        daily = pd.DataFrame(
            {"A": [10000.0] * 25, "B": [50000.0] * 25}, index=dates
        )

        result = factor.compute(daily_peak_sell_amt=daily, T=20)
        assert result.iloc[-1, 0] == pytest.approx(10000.0, rel=1e-10)
        assert result.iloc[-1, 1] == pytest.approx(50000.0, rel=1e-10)


class TestPeakVolSellAmtEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(np.ones(10) * 50000, index=dates, columns=stocks)
        daily.iloc[3, 0] = np.nan

        result = factor.compute(daily_peak_sell_amt=daily, T=5)
        assert isinstance(result, pd.DataFrame)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_peak_sell_amt=daily, T=5)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_peak_sell_amt=daily, T=20)
        for val in result.iloc[19:]["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)

    def test_insufficient_window(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(50000.0, index=dates, columns=stocks)

        result = factor.compute(daily_peak_sell_amt=daily, T=20)
        assert result.isna().all().all()


class TestPeakVolSellAmtOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily = pd.DataFrame(
            np.random.uniform(1e4, 1e5, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_peak_sell_amt=daily, T=20)

        assert result.shape == daily.shape
        assert list(result.columns) == list(daily.columns)
        assert list(result.index) == list(daily.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(50000.0, index=dates, columns=stocks)

        result = factor.compute(daily_peak_sell_amt=daily, T=20)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        T = 20
        daily = pd.DataFrame(50000.0, index=dates, columns=stocks)

        result = factor.compute(daily_peak_sell_amt=daily, T=T)

        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
