import numpy as np
import pandas as pd
import pytest

from factors.vcvar import VCVaRFactor


@pytest.fixture
def factor():
    return VCVaRFactor()


class TestMetadata:
    def test_name(self, factor):
        assert factor.name == "VCVaR"

    def test_category(self, factor):
        assert factor.category == "高频收益分布"

    def test_repr(self, factor):
        assert "VCVaR" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "VCVaR"
        assert meta["category"] == "高频收益分布"


class TestHandCalculated:
    """用手算数据验证 EWM(span=T, min_periods=1) 计算的正确性。"""

    def test_constant_input(self, factor):
        """常数输入时, EMA 应等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        daily_vcvar = pd.DataFrame(-0.03, index=dates, columns=stocks)

        result = factor.compute(daily_vcvar=daily_vcvar, T=20)

        np.testing.assert_array_almost_equal(result["A"].values, -0.03)

    def test_ema_manual_T3(self, factor):
        """T=3, 手动验证 EMA 值。

        ewm(span=3, adjust=True) alpha = 2/(3+1) = 0.5
        data = [10, 20, 30, 40]
          ema_0 = 10.0
          ema_1 = (0.5*10 + 1.0*20) / (0.5+1.0) = 16.6667
          ema_2 = (0.25*10 + 0.5*20 + 1.0*30) / (0.25+0.5+1.0) = 24.2857
          ema_3 = (0.125*10 + 0.25*20 + 0.5*30 + 1.0*40) / (0.125+0.25+0.5+1.0) = 32.6667
        """
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        stocks = ["A"]
        daily_vcvar = pd.DataFrame(
            [10.0, 20.0, 30.0, 40.0], index=dates, columns=stocks
        )

        result = factor.compute(daily_vcvar=daily_vcvar, T=3)

        assert result.iloc[0, 0] == pytest.approx(10.0, rel=1e-6)
        assert result.iloc[1, 0] == pytest.approx(50 / 3, rel=1e-6)
        assert result.iloc[2, 0] == pytest.approx(170 / 7, rel=1e-6)
        assert result.iloc[3, 0] == pytest.approx(490 / 15, rel=1e-6)

    def test_ema_recent_weight(self, factor):
        """EMA 应赋予近期数据更高权重。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        vals = [-0.05] * 5 + [-0.01] * 5
        daily_vcvar = pd.DataFrame(vals, index=dates, columns=stocks)

        result = factor.compute(daily_vcvar=daily_vcvar, T=5)
        # EMA 应偏向 -0.01（近期值）
        assert result.iloc[-1, 0] > -0.03

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        daily_vcvar = pd.DataFrame(
            {"A": [-0.02] * 10, "B": [-0.05] * 10}, index=dates
        )

        result = factor.compute(daily_vcvar=daily_vcvar, T=5)

        np.testing.assert_array_almost_equal(result["A"].values, -0.02)
        np.testing.assert_array_almost_equal(result["B"].values, -0.05)


class TestEdgeCases:
    def test_single_value(self, factor):
        """单个数据点的 EMA 应等于该值 (min_periods=1)。"""
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A"]
        daily_vcvar = pd.DataFrame([-0.04], index=dates, columns=stocks)

        result = factor.compute(daily_vcvar=daily_vcvar, T=20)
        assert result.iloc[0, 0] == pytest.approx(-0.04, rel=1e-10)

    def test_nan_in_input(self, factor):
        """输入含 NaN 时, 不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        values = np.ones(10) * -0.02
        values[3] = np.nan
        daily_vcvar = pd.DataFrame(values, index=dates, columns=stocks)

        result = factor.compute(daily_vcvar=daily_vcvar, T=5)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        """全 NaN 输入时, 结果应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily_vcvar = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_vcvar=daily_vcvar, T=5)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        """全零输入时, EMA 应全为 0。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily_vcvar = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_vcvar=daily_vcvar, T=5)
        for val in result["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)


class TestOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily_vcvar = pd.DataFrame(
            np.random.uniform(-0.05, 0.0, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_vcvar=daily_vcvar, T=20)

        assert result.shape == daily_vcvar.shape
        assert list(result.columns) == list(daily_vcvar.columns)
        assert list(result.index) == list(daily_vcvar.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_vcvar = pd.DataFrame([-0.01, -0.02, -0.03, -0.04, -0.05], index=dates, columns=stocks)

        result = factor.compute(daily_vcvar=daily_vcvar, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_no_leading_nan_min_periods_1(self, factor):
        """min_periods=1, 所以第一行就有值, 不存在前导 NaN。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        daily_vcvar = pd.DataFrame(
            np.random.uniform(-0.05, 0.0, (10, 2)), index=dates, columns=stocks
        )

        result = factor.compute(daily_vcvar=daily_vcvar, T=20)
        assert result.iloc[0].notna().all()
        assert result.notna().all().all()
