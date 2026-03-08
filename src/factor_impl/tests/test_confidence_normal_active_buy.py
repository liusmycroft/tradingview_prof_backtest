import numpy as np
import pandas as pd
import pytest

from factors.confidence_normal_active_buy import ConfidenceNormalActiveBuyFactor


@pytest.fixture
def factor():
    return ConfidenceNormalActiveBuyFactor()


class TestConfidenceNormalActiveBuyMetadata:
    def test_name(self, factor):
        assert factor.name == "CONFIDENCE_NORMAL_ACTIVE_BUY"

    def test_category(self, factor):
        assert factor.category == "高频资金流"

    def test_repr(self, factor):
        assert "CONFIDENCE_NORMAL_ACTIVE_BUY" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "CONFIDENCE_NORMAL_ACTIVE_BUY"
        assert meta["category"] == "高频资金流"


class TestConfidenceNormalActiveBuyCompute:
    def test_zero_returns(self, factor):
        """收益率为0时，N(0)=0.5，因子应为0.5。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        cols = list(range(48))
        minute_returns = pd.DataFrame(0.0, index=dates, columns=cols)
        minute_amount = pd.DataFrame(1000.0, index=dates, columns=cols)

        result = factor.compute(minute_returns=minute_returns, minute_amount=minute_amount)
        np.testing.assert_array_almost_equal(result["factor"].values, 0.5)

    def test_positive_returns_above_half(self, factor):
        """正收益率时，N(z)>0.5，因子应>0.5。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        cols = list(range(48))
        minute_returns = pd.DataFrame(0.05, index=dates, columns=cols)
        minute_amount = pd.DataFrame(1000.0, index=dates, columns=cols)

        result = factor.compute(minute_returns=minute_returns, minute_amount=minute_amount)
        assert (result["factor"] > 0.5).all()

    def test_negative_returns_below_half(self, factor):
        """负收益率时，N(z)<0.5，因子应<0.5。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        cols = list(range(48))
        minute_returns = pd.DataFrame(-0.05, index=dates, columns=cols)
        minute_amount = pd.DataFrame(1000.0, index=dates, columns=cols)

        result = factor.compute(minute_returns=minute_returns, minute_amount=minute_amount)
        assert (result["factor"] < 0.5).all()

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        cols = list(range(48))
        minute_returns = pd.DataFrame(
            np.random.randn(5, 48) * 0.01, index=dates, columns=cols
        )
        minute_amount = pd.DataFrame(
            np.random.uniform(100, 1000, (5, 48)), index=dates, columns=cols
        )

        result = factor.compute(minute_returns=minute_returns, minute_amount=minute_amount)
        assert result.shape == (5, 1)

    def test_factor_bounded(self, factor):
        """因子值应在 [0, 1] 之间。"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        cols = list(range(48))
        minute_returns = pd.DataFrame(
            np.random.randn(10, 48) * 0.05, index=dates, columns=cols
        )
        minute_amount = pd.DataFrame(
            np.random.uniform(100, 1000, (10, 48)), index=dates, columns=cols
        )

        result = factor.compute(minute_returns=minute_returns, minute_amount=minute_amount)
        assert (result["factor"] >= 0).all()
        assert (result["factor"] <= 1).all()


class TestConfidenceNormalActiveBuyEdgeCases:
    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        cols = list(range(10))
        minute_returns = pd.DataFrame(0.01, index=dates, columns=cols)
        minute_amount = pd.DataFrame(100.0, index=dates, columns=cols)

        result = factor.compute(minute_returns=minute_returns, minute_amount=minute_amount)
        assert isinstance(result, pd.DataFrame)
