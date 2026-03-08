import numpy as np
import pandas as pd
import pytest

from factors.liquidity_premium_ew import LiquidityPremiumEWFactor


@pytest.fixture
def factor():
    return LiquidityPremiumEWFactor()


class TestLiquidityPremiumEWMetadata:
    def test_name(self, factor):
        assert factor.name == "LIQUIDITY_PREMIUM_EW"

    def test_category(self, factor):
        assert factor.category == "高频流动性"

    def test_repr(self, factor):
        assert "LIQUIDITY_PREMIUM_EW" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "LIQUIDITY_PREMIUM_EW"
        assert meta["category"] == "高频流动性"


class TestLiquidityPremiumEWHandCalculated:
    def test_equal_cap(self, factor):
        """need == actual 时, ratio=1, mean-1=0。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        cap_need = pd.DataFrame(100.0, index=dates, columns=["A"])
        cap_actual = pd.DataFrame(100.0, index=dates, columns=["A"])

        result = factor.compute(daily_cap_need=cap_need, daily_cap_actual=cap_actual, T=3)
        assert result.iloc[2, 0] == pytest.approx(0.0, abs=1e-10)

    def test_constant_ratio(self, factor):
        """need/actual = 1.1 时, mean-1 = 0.1。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        cap_need = pd.DataFrame(110.0, index=dates, columns=["A"])
        cap_actual = pd.DataFrame(100.0, index=dates, columns=["A"])

        result = factor.compute(daily_cap_need=cap_need, daily_cap_actual=cap_actual, T=3)
        assert result.iloc[2, 0] == pytest.approx(0.1, rel=1e-6)

    def test_varying_ratio_T3(self, factor):
        """T=3, ratio=[1.1, 1.2, 1.3] => mean=1.2, result=0.2。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        cap_need = pd.DataFrame([110.0, 120.0, 130.0], index=dates, columns=["A"])
        cap_actual = pd.DataFrame(100.0, index=dates, columns=["A"])

        result = factor.compute(daily_cap_need=cap_need, daily_cap_actual=cap_actual, T=3)
        assert result.iloc[2, 0] == pytest.approx(0.2, rel=1e-6)


class TestLiquidityPremiumEWEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        cap_need = pd.DataFrame([110.0, np.nan, 130.0, 140.0, 150.0], index=dates, columns=["A"])
        cap_actual = pd.DataFrame(100.0, index=dates, columns=["A"])

        result = factor.compute(daily_cap_need=cap_need, daily_cap_actual=cap_actual, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        cap_need = pd.DataFrame(np.nan, index=dates, columns=["A"])
        cap_actual = pd.DataFrame(np.nan, index=dates, columns=["A"])

        result = factor.compute(daily_cap_need=cap_need, daily_cap_actual=cap_actual, T=5)
        assert result.isna().all().all()

    def test_zero_actual(self, factor):
        """actual=0 时, ratio=inf, 结果应为 inf 或 NaN。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        cap_need = pd.DataFrame(100.0, index=dates, columns=["A"])
        cap_actual = pd.DataFrame(0.0, index=dates, columns=["A"])

        result = factor.compute(daily_cap_need=cap_need, daily_cap_actual=cap_actual, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_insufficient_data(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        cap_need = pd.DataFrame(110.0, index=dates, columns=["A"])
        cap_actual = pd.DataFrame(100.0, index=dates, columns=["A"])

        result = factor.compute(daily_cap_need=cap_need, daily_cap_actual=cap_actual, T=20)
        assert result.isna().all().all()


class TestLiquidityPremiumEWOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        stocks = ["A", "B", "C"]
        cap_need = pd.DataFrame(
            np.random.uniform(100, 200, (50, 3)), index=dates, columns=stocks
        )
        cap_actual = pd.DataFrame(
            np.random.uniform(80, 150, (50, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_cap_need=cap_need, daily_cap_actual=cap_actual, T=20)
        assert result.shape == cap_need.shape
        assert list(result.columns) == list(cap_need.columns)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        cap_need = pd.DataFrame(110.0, index=dates, columns=["A"])
        cap_actual = pd.DataFrame(100.0, index=dates, columns=["A"])

        result = factor.compute(daily_cap_need=cap_need, daily_cap_actual=cap_actual, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        T = 5
        cap_need = pd.DataFrame(
            np.random.uniform(100, 200, (10, 2)), index=dates, columns=stocks
        )
        cap_actual = pd.DataFrame(
            np.random.uniform(80, 150, (10, 2)), index=dates, columns=stocks
        )

        result = factor.compute(daily_cap_need=cap_need, daily_cap_actual=cap_actual, T=T)
        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
