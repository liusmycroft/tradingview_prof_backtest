import numpy as np
import pandas as pd
import pytest

from factors.attention_abnormal_return import AttentionAbnormalReturnFactor


@pytest.fixture
def factor():
    return AttentionAbnormalReturnFactor()


class TestAttentionAbnormalReturnMetadata:
    def test_name(self, factor):
        assert factor.name == "ATTENTION_ABNORMAL_RETURN"

    def test_category(self, factor):
        assert factor.category == "行为金融-投资者注意力"

    def test_repr(self, factor):
        assert "ATTENTION_ABNORMAL_RETURN" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "ATTENTION_ABNORMAL_RETURN"
        assert meta["category"] == "行为金融-投资者注意力"


class TestAttentionAbnormalReturnCompute:
    def test_constant_abnormal_return(self, factor):
        """异常收益恒定时，因子值应等于其绝对值。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily_return = pd.DataFrame(0.05, index=dates, columns=stocks)
        market_return = pd.DataFrame(0.02, index=dates, columns=stocks)

        result = factor.compute(daily_return=daily_return, market_return=market_return, T=5)
        # |0.05 - 0.02| = 0.03
        np.testing.assert_array_almost_equal(result["A"].values, 0.03)

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
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_return = pd.DataFrame([0.05, -0.03, 0.02, 0.04, -0.01], index=dates, columns=stocks)
        market_return = pd.DataFrame([0.01, 0.01, 0.01, 0.01, 0.01], index=dates, columns=stocks)

        result = factor.compute(daily_return=daily_return, market_return=market_return, T=3)

        # abs abnormal: [0.04, 0.04, 0.01, 0.03, 0.02]
        # rolling mean T=3, min_periods=1:
        #   [0.04, 0.04, 0.03, 0.02667, 0.02]
        assert result.iloc[0, 0] == pytest.approx(0.04, rel=1e-6)
        assert result.iloc[1, 0] == pytest.approx(0.04, rel=1e-6)
        assert result.iloc[2, 0] == pytest.approx((0.04 + 0.04 + 0.01) / 3, rel=1e-6)
        assert result.iloc[3, 0] == pytest.approx((0.04 + 0.01 + 0.03) / 3, rel=1e-6)
        assert result.iloc[4, 0] == pytest.approx((0.01 + 0.03 + 0.02) / 3, rel=1e-6)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        daily_return = pd.DataFrame(
            {"A": [0.05] * 10, "B": [-0.03] * 10}, index=dates
        )
        market_return = pd.DataFrame(
            {"A": [0.01] * 10, "B": [0.01] * 10}, index=dates
        )

        result = factor.compute(daily_return=daily_return, market_return=market_return, T=5)
        np.testing.assert_array_almost_equal(result["A"].values, 0.04)
        np.testing.assert_array_almost_equal(result["B"].values, 0.04)

    def test_negative_abnormal_uses_abs(self, factor):
        """负异常收益也应取绝对值。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_return = pd.DataFrame(-0.05, index=dates, columns=stocks)
        market_return = pd.DataFrame(0.01, index=dates, columns=stocks)

        result = factor.compute(daily_return=daily_return, market_return=market_return, T=3)
        np.testing.assert_array_almost_equal(result["A"].values, 0.06)


class TestAttentionAbnormalReturnEdgeCases:
    def test_single_value(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A"]
        daily_return = pd.DataFrame([0.05], index=dates, columns=stocks)
        market_return = pd.DataFrame([0.02], index=dates, columns=stocks)

        result = factor.compute(daily_return=daily_return, market_return=market_return, T=20)
        assert result.iloc[0, 0] == pytest.approx(0.03, rel=1e-10)

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


class TestAttentionAbnormalReturnOutputShape:
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
