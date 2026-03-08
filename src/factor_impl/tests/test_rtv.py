import numpy as np
import pandas as pd
import pytest

from factors.rtv import RTVFactor


@pytest.fixture
def factor():
    return RTVFactor()


class TestRTVMetadata:
    def test_name(self, factor):
        assert factor.name == "RTV"

    def test_category(self, factor):
        assert factor.category == "高频波动跳跃"

    def test_repr(self, factor):
        assert "RTV" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "RTV"
        assert meta["category"] == "高频波动跳跃"


class TestRTVHandCalculated:
    def test_constant_input(self, factor):
        """常数输入时, EMA 收敛到该常数。"""
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(0.05, index=dates, columns=stocks)

        result = factor.compute(daily_rtv=data, T=20)
        assert result.iloc[-1, 0] == pytest.approx(0.05, rel=1e-6)

    def test_ema_first_value(self, factor):
        """EMA 第一个值等于输入第一个值 (min_periods=1)。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        data = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0], index=dates, columns=stocks)

        result = factor.compute(daily_rtv=data, T=20)
        assert result.iloc[0, 0] == pytest.approx(1.0, rel=1e-10)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        data = pd.DataFrame({"A": [0.03] * 60, "B": [0.07] * 60}, index=dates)

        result = factor.compute(daily_rtv=data, T=20)
        assert result.iloc[-1, 0] == pytest.approx(0.03, rel=1e-6)
        assert result.iloc[-1, 1] == pytest.approx(0.07, rel=1e-6)


class TestRTVEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        values = np.ones(10) * 0.05
        values[3] = np.nan
        data = pd.DataFrame(values, index=dates, columns=stocks)

        result = factor.compute(daily_rtv=data, T=5)
        assert isinstance(result, pd.DataFrame)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_rtv=data, T=5)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_rtv=data, T=20)
        assert result.iloc[-1, 0] == pytest.approx(0.0, abs=1e-15)


class TestRTVOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        data = pd.DataFrame(
            np.random.uniform(0, 0.1, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_rtv=data, T=20)
        assert result.shape == data.shape
        assert list(result.columns) == list(data.columns)
        assert list(result.index) == list(data.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(0.05, index=dates, columns=stocks)

        result = factor.compute(daily_rtv=data, T=20)
        assert isinstance(result, pd.DataFrame)

    def test_no_nan_with_sufficient_data(self, factor):
        """EMA with min_periods=1 should produce no NaN for non-NaN input."""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        data = pd.DataFrame(
            np.random.uniform(0, 0.1, (30, 2)), index=dates, columns=stocks
        )

        result = factor.compute(daily_rtv=data, T=20)
        assert result.notna().all().all()
