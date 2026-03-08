import numpy as np
import pandas as pd
import pytest

from factors.intraday_ret_vol_ratio import IntradayReturnVolRatioFactor


@pytest.fixture
def factor():
    return IntradayReturnVolRatioFactor()


class TestIntradayRetVolRatioMetadata:
    def test_name(self, factor):
        assert factor.name == "INTRADAY_RET_VOL_RATIO"

    def test_category(self, factor):
        assert factor.category == "高频波动跳跃"

    def test_repr(self, factor):
        assert "INTRADAY_RET_VOL_RATIO" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "INTRADAY_RET_VOL_RATIO"
        assert meta["category"] == "高频波动跳跃"


class TestIntradayRetVolRatioHandCalculated:
    def test_constant_ratio(self, factor):
        """ret=0.02, vol=0.01 -> ratio=2.0, rolling mean of constant=2.0."""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        ret = pd.DataFrame(0.02, index=dates, columns=stocks)
        vol = pd.DataFrame(0.01, index=dates, columns=stocks)

        result = factor.compute(
            daily_intraday_return=ret, daily_intraday_volatility=vol, T=20
        )
        np.testing.assert_allclose(result["A"].values, 2.0, atol=1e-10)

    def test_simple_mean_T3(self, factor):
        """T=3 rolling mean of ratio with min_periods=1."""
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        stocks = ["A"]
        ret = pd.DataFrame([0.01, 0.02, 0.03, 0.04], index=dates, columns=stocks)
        vol = pd.DataFrame([0.01, 0.01, 0.01, 0.01], index=dates, columns=stocks)
        # ratios: [1, 2, 3, 4]

        result = factor.compute(
            daily_intraday_return=ret, daily_intraday_volatility=vol, T=3
        )
        assert result.iloc[0, 0] == pytest.approx(1.0, rel=1e-10)
        assert result.iloc[1, 0] == pytest.approx(1.5, rel=1e-10)
        assert result.iloc[2, 0] == pytest.approx(2.0, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(3.0, rel=1e-10)

    def test_negative_return(self, factor):
        """Negative returns produce negative ratios."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        ret = pd.DataFrame(-0.02, index=dates, columns=stocks)
        vol = pd.DataFrame(0.01, index=dates, columns=stocks)

        result = factor.compute(
            daily_intraday_return=ret, daily_intraday_volatility=vol, T=5
        )
        np.testing.assert_allclose(result["A"].values, -2.0, atol=1e-10)


class TestIntradayRetVolRatioEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        ret = pd.DataFrame(np.ones(10) * 0.02, index=dates, columns=stocks)
        ret.iloc[3, 0] = np.nan
        vol = pd.DataFrame(np.ones(10) * 0.01, index=dates, columns=stocks)

        result = factor.compute(
            daily_intraday_return=ret, daily_intraday_volatility=vol, T=5
        )
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        ret = pd.DataFrame(np.nan, index=dates, columns=stocks)
        vol = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(
            daily_intraday_return=ret, daily_intraday_volatility=vol, T=5
        )
        assert result.isna().all().all()

    def test_zero_volatility(self, factor):
        """Zero volatility -> inf ratio -> rolling mean is inf."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        ret = pd.DataFrame(0.02, index=dates, columns=stocks)
        vol = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_intraday_return=ret, daily_intraday_volatility=vol, T=5
        )
        assert isinstance(result, pd.DataFrame)


class TestIntradayRetVolRatioOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        np.random.seed(42)
        ret = pd.DataFrame(
            np.random.randn(30, 3) * 0.02, index=dates, columns=stocks
        )
        vol = pd.DataFrame(
            np.random.rand(30, 3) * 0.01 + 0.001, index=dates, columns=stocks
        )

        result = factor.compute(
            daily_intraday_return=ret, daily_intraday_volatility=vol, T=20
        )
        assert result.shape == ret.shape
        assert list(result.columns) == list(ret.columns)
        assert list(result.index) == list(ret.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        ret = pd.DataFrame([0.01, 0.02, 0.03, 0.02, 0.01], index=dates, columns=stocks)
        vol = pd.DataFrame([0.01, 0.01, 0.01, 0.01, 0.01], index=dates, columns=stocks)

        result = factor.compute(
            daily_intraday_return=ret, daily_intraday_volatility=vol, T=3
        )
        assert isinstance(result, pd.DataFrame)

    def test_no_leading_nan_min_periods_1(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        ret = pd.DataFrame(
            np.random.randn(10, 2) * 0.02, index=dates, columns=stocks
        )
        vol = pd.DataFrame(
            np.random.rand(10, 2) * 0.01 + 0.001, index=dates, columns=stocks
        )

        result = factor.compute(
            daily_intraday_return=ret, daily_intraday_volatility=vol, T=20
        )
        assert result.iloc[0].notna().all()
