import numpy as np
import pandas as pd
import pytest

from factors.smart_money import SmartMoneyFactor


@pytest.fixture
def factor():
    return SmartMoneyFactor()


class TestSmartMoneyMetadata:
    def test_name(self, factor):
        assert factor.name == "SMART_MONEY"

    def test_category(self, factor):
        assert factor.category == "高频量价相关性"

    def test_repr(self, factor):
        assert "SMART_MONEY" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "SMART_MONEY"
        assert meta["category"] == "高频量价相关性"


class TestSmartMoneyHandCalculated:
    def test_constant_ratio(self, factor):
        """When VWAP_smart == VWAP_all, ratio=1, EMA should be 1."""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        vwap_smart = pd.DataFrame(10.0, index=dates, columns=stocks)
        vwap_all = pd.DataFrame(10.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_vwap_smart=vwap_smart, daily_vwap_all=vwap_all, T=20
        )
        np.testing.assert_array_almost_equal(result["A"].values, 1.0)

    def test_smart_higher_than_all(self, factor):
        """When smart VWAP > all VWAP, ratio > 1."""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        vwap_smart = pd.DataFrame(11.0, index=dates, columns=stocks)
        vwap_all = pd.DataFrame(10.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_vwap_smart=vwap_smart, daily_vwap_all=vwap_all, T=20
        )
        np.testing.assert_array_almost_equal(result["A"].values, 1.1)

    def test_ema_manual_T3(self, factor):
        """T=3 EMA hand-calculated verification.

        ratio = [1.0, 1.1, 1.2, 0.9]
        ewm(span=3, adjust=True), alpha = 2/(3+1) = 0.5
          ema_0 = 1.0
          ema_1 = (0.5*1.0 + 1.0*1.1) / (0.5+1.0) = 1.6/1.5
          ema_2 = (0.25*1.0 + 0.5*1.1 + 1.0*1.2) / 1.75 = 2.0/1.75
          ema_3 = (0.125*1.0 + 0.25*1.1 + 0.5*1.2 + 1.0*0.9) / 1.875 = 1.9/1.875
        """
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        stocks = ["A"]
        vwap_smart = pd.DataFrame([10.0, 11.0, 12.0, 9.0], index=dates, columns=stocks)
        vwap_all = pd.DataFrame(10.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_vwap_smart=vwap_smart, daily_vwap_all=vwap_all, T=3
        )

        assert result.iloc[0, 0] == pytest.approx(1.0, rel=1e-6)
        assert result.iloc[1, 0] == pytest.approx(1.6 / 1.5, rel=1e-6)
        assert result.iloc[2, 0] == pytest.approx(2.0 / 1.75, rel=1e-6)
        assert result.iloc[3, 0] == pytest.approx(1.9 / 1.875, rel=1e-6)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        vwap_smart = pd.DataFrame({"A": [10.0] * 20, "B": [20.0] * 20}, index=dates)
        vwap_all = pd.DataFrame({"A": [10.0] * 20, "B": [10.0] * 20}, index=dates)

        result = factor.compute(
            daily_vwap_smart=vwap_smart, daily_vwap_all=vwap_all, T=20
        )
        np.testing.assert_array_almost_equal(result["A"].values, 1.0)
        np.testing.assert_array_almost_equal(result["B"].values, 2.0)


class TestSmartMoneyEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        smart_vals = np.ones(10) * 10.0
        smart_vals[3] = np.nan
        vwap_smart = pd.DataFrame(smart_vals, index=dates, columns=stocks)
        vwap_all = pd.DataFrame(10.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_vwap_smart=vwap_smart, daily_vwap_all=vwap_all, T=5
        )
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        vwap_smart = pd.DataFrame(np.nan, index=dates, columns=stocks)
        vwap_all = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(
            daily_vwap_smart=vwap_smart, daily_vwap_all=vwap_all, T=5
        )
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        """Zero denominator produces NaN/inf, should not raise."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        vwap_smart = pd.DataFrame(0.0, index=dates, columns=stocks)
        vwap_all = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_vwap_smart=vwap_smart, daily_vwap_all=vwap_all, T=5
        )
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)


class TestSmartMoneyOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        vwap_smart = pd.DataFrame(
            np.random.uniform(9, 11, (30, 3)), index=dates, columns=stocks
        )
        vwap_all = pd.DataFrame(
            np.random.uniform(9, 11, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(
            daily_vwap_smart=vwap_smart, daily_vwap_all=vwap_all, T=20
        )
        assert result.shape == vwap_smart.shape
        assert list(result.columns) == list(vwap_smart.columns)
        assert list(result.index) == list(vwap_smart.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        vwap_smart = pd.DataFrame(
            [10.0, 11.0, 10.5, 10.2, 10.8], index=dates, columns=stocks
        )
        vwap_all = pd.DataFrame(10.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_vwap_smart=vwap_smart, daily_vwap_all=vwap_all, T=3
        )
        assert isinstance(result, pd.DataFrame)
