import numpy as np
import pandas as pd
import pytest

from factors.abnormal_return_avg import AbnormalReturnAvgFactor


@pytest.fixture
def factor():
    return AbnormalReturnAvgFactor()


class TestAbnormalReturnAvgMetadata:
    def test_name(self, factor):
        assert factor.name == "ABNRETAVG"

    def test_category(self, factor):
        assert factor.category == "行为金融-投资者注意力"

    def test_repr(self, factor):
        assert "ABNRETAVG" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "ABNRETAVG"
        assert meta["category"] == "行为金融-投资者注意力"


class TestAbnormalReturnAvgCompute:
    def test_constant_abnormal_return(self, factor):
        """异常收益恒定时，因子值应等于其平方。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily_return = pd.DataFrame(0.05, index=dates, columns=stocks)
        market_return = pd.DataFrame(0.02, index=dates, columns=stocks)

        result = factor.compute(daily_return=daily_return, market_return=market_return, T=5)
        # (0.05 - 0.02)^2 = 0.0009
        np.testing.assert_array_almost_equal(result["A"].values, 0.0009)

    def test_zero_abnormal_return(self, factor):
        """个股收益等于市场收益时，因子值应为 0。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily_return = pd.DataFrame(0.01, index=dates, columns=stocks)
        market_return = pd.DataFrame(0.01, index=dates, columns=stocks)

        result = factor.compute(daily_return=daily_return, market_return=market_return, T=5)
        np.testing.assert_array_almost_equal(result["A"].values, 0.0)

    def test_manual_rolling_T3(self, factor):
        """T=3 手动验证滚动均值。"""
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        stocks = ["A"]
        daily_return = pd.DataFrame([0.05, -0.03, 0.02, 0.06], index=dates, columns=stocks)
        market_return = pd.DataFrame([0.01, 0.01, 0.01, 0.01], index=dates, columns=stocks)

        result = factor.compute(daily_return=daily_return, market_return=market_return, T=3)

        # abnormal: [0.04, -0.04, 0.01, 0.05]
        # squared:  [0.0016, 0.0016, 0.0001, 0.0025]
        # rolling mean T=3, min_periods=1:
        #   [0.0016, 0.0016, 0.0011, 0.0014]
        assert result.iloc[0, 0] == pytest.approx(0.0016, rel=1e-6)
        assert result.iloc[1, 0] == pytest.approx(0.0016, rel=1e-6)
        assert result.iloc[2, 0] == pytest.approx((0.0016 + 0.0016 + 0.0001) / 3, rel=1e-6)
        assert result.iloc[3, 0] == pytest.approx((0.0016 + 0.0001 + 0.0025) / 3, rel=1e-6)

    def test_negative_abnormal_squared_is_positive(self, factor):
        """负异常收益的平方也应为正。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_return = pd.DataFrame(-0.05, index=dates, columns=stocks)
        market_return = pd.DataFrame(0.01, index=dates, columns=stocks)

        result = factor.compute(daily_return=daily_return, market_return=market_return, T=3)
        # (-0.06)^2 = 0.0036
        np.testing.assert_array_almost_equal(result["A"].values, 0.0036)
        assert (result.values >= 0).all()

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        daily_return = pd.DataFrame(
            {"A": [0.05] * 10, "B": [-0.03] * 10}, index=dates
        )
        market_return = pd.DataFrame(
            {"A": [0.01] * 10, "B": [0.01] * 10}, index=dates
        )

        result = factor.compute(daily_return=daily_return, market_return=market_return, T=5)
        np.testing.assert_array_almost_equal(result["A"].values, 0.04**2)
        np.testing.assert_array_almost_equal(result["B"].values, 0.04**2)


class TestAbnormalReturnAvgEdgeCases:
    def test_single_value(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A"]
        daily_return = pd.DataFrame([0.05], index=dates, columns=stocks)
        market_return = pd.DataFrame([0.02], index=dates, columns=stocks)

        result = factor.compute(daily_return=daily_return, market_return=market_return, T=20)
        assert result.iloc[0, 0] == pytest.approx(0.03**2, rel=1e-10)

    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        vals = np.ones(10) * 0.05
        vals[3] = np.nan
        daily_return = pd.DataFrame(vals, index=dates, columns=stocks)
        market_return = pd.DataFrame(0.01, index=dates, columns=stocks)

        result = factor.compute(daily_return=daily_return, market_return=market_return, T=5)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily_return = pd.DataFrame(np.nan, index=dates, columns=stocks)
        market_return = pd.DataFrame(0.01, index=dates, columns=stocks)

        result = factor.compute(daily_return=daily_return, market_return=market_return, T=5)
        assert result.isna().all().all()

    def test_result_always_non_negative(self, factor):
        """平方值应始终非负。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A", "B"]
        np.random.seed(42)
        daily_return = pd.DataFrame(
            np.random.randn(20, 2) * 0.05, index=dates, columns=stocks
        )
        market_return = pd.DataFrame(
            np.random.randn(20, 2) * 0.02, index=dates, columns=stocks
        )

        result = factor.compute(daily_return=daily_return, market_return=market_return, T=5)
        valid = result.dropna()
        assert (valid.values >= -1e-15).all()


class TestAbnormalReturnAvgOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily_return = pd.DataFrame(
            np.random.randn(30, 3) * 0.02, index=dates, columns=stocks
        )
        market_return = pd.DataFrame(
            np.random.randn(30, 3) * 0.01, index=dates, columns=stocks
        )

        result = factor.compute(daily_return=daily_return, market_return=market_return, T=20)
        assert result.shape == daily_return.shape
        assert list(result.columns) == list(daily_return.columns)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_return = pd.DataFrame([0.01, 0.02, 0.03, 0.04, 0.05], index=dates, columns=stocks)
        market_return = pd.DataFrame(0.01, index=dates, columns=stocks)

        result = factor.compute(daily_return=daily_return, market_return=market_return, T=3)
        assert isinstance(result, pd.DataFrame)
