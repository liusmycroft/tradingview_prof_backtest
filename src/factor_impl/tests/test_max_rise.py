import numpy as np
import pandas as pd
import pytest

from factors.max_rise import MaxRiseFactor


@pytest.fixture
def factor():
    return MaxRiseFactor()


class TestMaxRiseMetadata:
    def test_name(self, factor):
        assert factor.name == "MAX_RISE"

    def test_category(self, factor):
        assert factor.category == "高频动量反转"

    def test_repr(self, factor):
        assert "MAX_RISE" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "MAX_RISE"
        assert meta["category"] == "高频动量反转"


class TestMaxRiseCompute:
    def test_constant_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        daily = pd.DataFrame(1.05, index=dates, columns=["A"])

        result = factor.compute(daily_max_rise=daily, T=20)
        np.testing.assert_array_almost_equal(result["A"].values, 1.05)

    def test_rolling_mean(self, factor):
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        daily = pd.DataFrame([1.01, 1.02, 1.03, 1.04], index=dates, columns=["A"])

        result = factor.compute(daily_max_rise=daily, T=2)
        assert result.iloc[0, 0] == pytest.approx(1.01)
        assert result.iloc[1, 0] == pytest.approx(1.015)
        assert result.iloc[2, 0] == pytest.approx(1.025)
        assert result.iloc[3, 0] == pytest.approx(1.035)

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        daily = pd.DataFrame(np.random.uniform(1.0, 1.1, (30, 2)),
                             index=dates, columns=stocks)

        result = factor.compute(daily_max_rise=daily, T=20)
        assert result.shape == daily.shape


class TestMaxRiseDaily:
    def test_compute_daily_basic(self):
        """验证 compute_daily 基本逻辑。"""
        # 10个分钟收益率，前10%即最大的1个
        returns = pd.Series([0.001, 0.002, 0.003, 0.004, 0.005,
                             0.006, 0.007, 0.008, 0.009, 0.01])
        result = MaxRiseFactor.compute_daily(returns, quantile=0.9)
        # 前10%: 0.01
        assert result == pytest.approx(1.01)

    def test_compute_daily_all_positive(self):
        """所有收益率相同时，quantile(0.9)=该值，所有值都>=阈值，全部参与乘积。"""
        returns = pd.Series([0.01] * 100)
        result = MaxRiseFactor.compute_daily(returns, quantile=0.9)
        # 所有值相同，quantile(0.9)=0.01，全部100个都>=阈值
        expected = (1.01) ** 100
        assert result == pytest.approx(expected, rel=1e-6)

    def test_compute_daily_zero_returns(self):
        returns = pd.Series([0.0] * 100)
        result = MaxRiseFactor.compute_daily(returns, quantile=0.9)
        assert result == pytest.approx(1.0)


class TestMaxRiseEdgeCases:
    def test_single_value(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        daily = pd.DataFrame([1.05], index=dates, columns=["A"])

        result = factor.compute(daily_max_rise=daily, T=20)
        assert result.iloc[0, 0] == pytest.approx(1.05)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        daily = pd.DataFrame(np.nan, index=dates, columns=["A"])

        result = factor.compute(daily_max_rise=daily, T=5)
        assert result.isna().all().all()
