import numpy as np
import pandas as pd
import pytest

from factors.corrected_ambiguity_spread import CorrectedAmbiguitySpreadFactor


@pytest.fixture
def factor():
    return CorrectedAmbiguitySpreadFactor()


class TestCorrectedAmbiguitySpreadMetadata:
    def test_name(self, factor):
        assert factor.name == "CORRECTED_AMBIGUITY_SPREAD"

    def test_category(self, factor):
        assert factor.category == "高频成交分布"

    def test_repr(self, factor):
        assert "CORRECTED_AMBIGUITY_SPREAD" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "CORRECTED_AMBIGUITY_SPREAD"
        assert meta["category"] == "高频成交分布"


class TestCorrectedAmbiguitySpreadCompute:
    def test_constant_input(self, factor):
        """常数输入验证。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        spread = pd.DataFrame(0.05, index=dates, columns=stocks)
        ret = pd.DataFrame(0.01, index=dates, columns=stocks)

        result = factor.compute(daily_ambiguity_spread=spread, daily_return=ret, T=5)
        # 0.05 - |0.01| = 0.04
        np.testing.assert_array_almost_equal(result["A"].values, 0.04)

    def test_negative_return_uses_abs(self, factor):
        """负收益率也应取绝对值。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        spread = pd.DataFrame(0.05, index=dates, columns=stocks)
        ret = pd.DataFrame(-0.02, index=dates, columns=stocks)

        result = factor.compute(daily_ambiguity_spread=spread, daily_return=ret, T=3)
        # 0.05 - |-0.02| = 0.03
        np.testing.assert_array_almost_equal(result["A"].values, 0.03)

    def test_manual_rolling_T3(self, factor):
        """T=3 手动验证滚动均值。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        spread = pd.DataFrame([0.05, 0.06, 0.04, 0.07, 0.03], index=dates, columns=stocks)
        ret = pd.DataFrame([0.01, -0.02, 0.03, -0.01, 0.02], index=dates, columns=stocks)

        result = factor.compute(daily_ambiguity_spread=spread, daily_return=ret, T=3)

        # corrected: [0.05-0.01, 0.06-0.02, 0.04-0.03, 0.07-0.01, 0.03-0.02]
        #          = [0.04, 0.04, 0.01, 0.06, 0.01]
        # rolling mean T=3, min_periods=1:
        #   [0.04, 0.04, 0.03, 0.03667, 0.02667]
        assert result.iloc[0, 0] == pytest.approx(0.04, rel=1e-6)
        assert result.iloc[1, 0] == pytest.approx(0.04, rel=1e-6)
        assert result.iloc[2, 0] == pytest.approx(0.03, rel=1e-6)
        assert result.iloc[3, 0] == pytest.approx(0.11 / 3, rel=1e-6)
        assert result.iloc[4, 0] == pytest.approx(0.08 / 3, rel=1e-6)

    def test_large_return_makes_negative(self, factor):
        """收益率绝对值大于价差时，修正后为负。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        spread = pd.DataFrame(0.01, index=dates, columns=stocks)
        ret = pd.DataFrame(0.05, index=dates, columns=stocks)

        result = factor.compute(daily_ambiguity_spread=spread, daily_return=ret, T=3)
        # 0.01 - 0.05 = -0.04
        np.testing.assert_array_almost_equal(result["A"].values, -0.04)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        spread = pd.DataFrame({"A": [0.05] * 10, "B": [0.10] * 10}, index=dates)
        ret = pd.DataFrame({"A": [0.01] * 10, "B": [0.03] * 10}, index=dates)

        result = factor.compute(daily_ambiguity_spread=spread, daily_return=ret, T=5)
        np.testing.assert_array_almost_equal(result["A"].values, 0.04)
        np.testing.assert_array_almost_equal(result["B"].values, 0.07)


class TestCorrectedAmbiguitySpreadEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        spread = pd.DataFrame([0.05, np.nan, 0.04, 0.06, 0.03], index=dates, columns=stocks)
        ret = pd.DataFrame(0.01, index=dates, columns=stocks)

        result = factor.compute(daily_ambiguity_spread=spread, daily_return=ret, T=3)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (5, 1)

    def test_zero_spread_and_return(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        spread = pd.DataFrame(0.0, index=dates, columns=stocks)
        ret = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_ambiguity_spread=spread, daily_return=ret, T=3)
        np.testing.assert_array_almost_equal(result["A"].values, 0.0)


class TestCorrectedAmbiguitySpreadOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        spread = pd.DataFrame(np.random.uniform(0.01, 0.1, (30, 3)), index=dates, columns=stocks)
        ret = pd.DataFrame(np.random.randn(30, 3) * 0.02, index=dates, columns=stocks)

        result = factor.compute(daily_ambiguity_spread=spread, daily_return=ret)
        assert result.shape == spread.shape
        assert list(result.columns) == list(spread.columns)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        spread = pd.DataFrame(0.05, index=dates, columns=stocks)
        ret = pd.DataFrame(0.01, index=dates, columns=stocks)

        result = factor.compute(daily_ambiguity_spread=spread, daily_return=ret, T=3)
        assert isinstance(result, pd.DataFrame)
