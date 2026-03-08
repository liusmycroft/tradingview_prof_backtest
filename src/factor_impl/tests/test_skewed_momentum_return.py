import numpy as np
import pandas as pd
import pytest

from factors.skewed_momentum_return import SkewedMomentumReturnFactor


@pytest.fixture
def factor():
    return SkewedMomentumReturnFactor()


class TestSkewedMomentumReturnMetadata:
    def test_name(self, factor):
        assert factor.name == "SKEWED_MOMENTUM_RETURN"

    def test_category(self, factor):
        assert factor.category == "高频动量反转"

    def test_repr(self, factor):
        assert "SKEWED_MOMENTUM_RETURN" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "SKEWED_MOMENTUM_RETURN"
        assert meta["category"] == "高频动量反转"


class TestSkewedMomentumReturnHandCalculated:
    def test_rolling_std_T3(self, factor):
        """T=3 滚动标准差手算验证。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        data = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0], index=dates, columns=stocks)

        result = factor.compute(daily_deviation_sum=data, T=3)

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        # std([1,2,3]) = 1.0 (ddof=1)
        assert result.iloc[2, 0] == pytest.approx(1.0, rel=1e-10)
        # std([2,3,4]) = 1.0
        assert result.iloc[3, 0] == pytest.approx(1.0, rel=1e-10)
        # std([3,4,5]) = 1.0
        assert result.iloc[4, 0] == pytest.approx(1.0, rel=1e-10)

    def test_constant_input(self, factor):
        """常数输入时，标准差为 0。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(5.0, index=dates, columns=stocks)

        result = factor.compute(daily_deviation_sum=data, T=5)
        for i in range(4, 10):
            assert result.iloc[i, 0] == pytest.approx(0.0, abs=1e-15)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame(
            {"A": [1.0, 2.0, 3.0, 4.0, 5.0],
             "B": [10.0, 20.0, 30.0, 40.0, 50.0]},
            index=dates,
        )

        result = factor.compute(daily_deviation_sum=data, T=3)
        assert result.iloc[2, 0] == pytest.approx(1.0, rel=1e-10)
        assert result.iloc[2, 1] == pytest.approx(10.0, rel=1e-10)


class TestSkewedMomentumReturnEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        values = np.ones(10) * 5.0
        values[3] = np.nan
        data = pd.DataFrame(values, index=dates, columns=stocks)

        result = factor.compute(daily_deviation_sum=data, T=5)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_deviation_sum=data, T=5)
        assert result.isna().all().all()

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        data = pd.DataFrame(
            np.random.uniform(-1, 1, (30, 3)), index=dates, columns=stocks
        )
        result = factor.compute(daily_deviation_sum=data, T=20)
        assert result.shape == data.shape

    def test_first_T_minus_1_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(np.arange(10, dtype=float), index=dates, columns=stocks)
        T = 5
        result = factor.compute(daily_deviation_sum=data, T=T)
        assert result.iloc[: T - 1].isna().all().all()
