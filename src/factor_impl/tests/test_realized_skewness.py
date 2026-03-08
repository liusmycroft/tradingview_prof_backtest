import numpy as np
import pandas as pd
import pytest

from factors.realized_skewness import RealizedSkewnessFactor


@pytest.fixture
def factor():
    return RealizedSkewnessFactor()


class TestRealizedSkewnessMetadata:
    def test_name(self, factor):
        assert factor.name == "REALIZED_SKEWNESS"

    def test_category(self, factor):
        assert factor.category == "高频收益分布"

    def test_repr(self, factor):
        assert "REALIZED_SKEWNESS" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "REALIZED_SKEWNESS"
        assert meta["category"] == "高频收益分布"


class TestRealizedSkewnessHandCalculated:
    def test_rolling_mean_T3(self, factor):
        """T=3 滚动均值手算验证。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        data = pd.DataFrame([0.5, -0.3, 0.8, -0.1, 0.4], index=dates, columns=stocks)

        result = factor.compute(daily_rskew=data, T=3)

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])

        expected_2 = (0.5 + (-0.3) + 0.8) / 3.0
        assert result.iloc[2, 0] == pytest.approx(expected_2, rel=1e-10)

        expected_3 = ((-0.3) + 0.8 + (-0.1)) / 3.0
        assert result.iloc[3, 0] == pytest.approx(expected_3, rel=1e-10)

        expected_4 = (0.8 + (-0.1) + 0.4) / 3.0
        assert result.iloc[4, 0] == pytest.approx(expected_4, rel=1e-10)

    def test_constant_skewness(self, factor):
        """常数偏度时，滚动均值等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        data = pd.DataFrame([-0.5] * 5, index=dates, columns=stocks)

        result = factor.compute(daily_rskew=data, T=3)
        assert result.iloc[2, 0] == pytest.approx(-0.5, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(-0.5, rel=1e-10)

    def test_two_stocks(self, factor):
        """两只股票并行计算。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A", "B"]
        data = pd.DataFrame(
            [[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]],
            index=dates, columns=stocks,
        )

        result = factor.compute(daily_rskew=data, T=3)

        assert result.iloc[2, 0] == pytest.approx(2.0, rel=1e-10)
        assert result.iloc[2, 1] == pytest.approx(-2.0, rel=1e-10)


class TestRealizedSkewnessEdgeCases:
    def test_nan_propagation(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        data = pd.DataFrame([1.0, np.nan, 3.0, 4.0, 5.0], index=dates, columns=stocks)

        result = factor.compute(daily_rskew=data, T=3)
        assert np.isnan(result.iloc[2, 0])
        assert np.isnan(result.iloc[3, 0])
        assert not np.isnan(result.iloc[4, 0])

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        data = pd.DataFrame(
            np.random.normal(0, 1, (30, 2)), index=dates, columns=stocks
        )
        result = factor.compute(daily_rskew=data, T=20)
        assert result.shape == data.shape
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(np.ones(10), index=dates, columns=stocks)
        T = 5
        result = factor.compute(daily_rskew=data, T=T)
        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
