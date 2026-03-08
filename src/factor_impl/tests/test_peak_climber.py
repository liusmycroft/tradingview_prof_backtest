import numpy as np
import pandas as pd
import pytest

from factors.peak_climber import PeakClimberFactor


@pytest.fixture
def factor():
    return PeakClimberFactor()


class TestPeakClimberMetadata:
    def test_name(self, factor):
        assert factor.name == "PEAK_CLIMBER"

    def test_category(self, factor):
        assert factor.category == "高频波动跳跃"

    def test_repr(self, factor):
        assert "PEAK_CLIMBER" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "PEAK_CLIMBER"
        assert meta["category"] == "高频波动跳跃"


class TestPeakClimberHandCalculated:
    def test_constant_input(self, factor):
        """常数输入时，滚动均值应等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(0.05, index=dates, columns=stocks)

        result = factor.compute(daily_peak_climber=data, T=10)

        np.testing.assert_array_almost_equal(result["A"].values, 0.05)

    def test_rolling_mean_T3(self, factor):
        """T=3 滚动均值手算验证 (min_periods=1)。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        data = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0], index=dates, columns=stocks)

        result = factor.compute(daily_peak_climber=data, T=3)

        # min_periods=1: 第一个值就有
        assert result.iloc[0, 0] == pytest.approx(1.0, rel=1e-10)
        assert result.iloc[1, 0] == pytest.approx(1.5, rel=1e-10)
        assert result.iloc[2, 0] == pytest.approx(2.0, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(3.0, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(4.0, rel=1e-10)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(
            {"A": [0.1] * 10, "B": [0.5] * 10}, index=dates
        )

        result = factor.compute(daily_peak_climber=data, T=5)

        np.testing.assert_array_almost_equal(result["A"].values, 0.1)
        np.testing.assert_array_almost_equal(result["B"].values, 0.5)


class TestPeakClimberEdgeCases:
    def test_single_value(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        data = pd.DataFrame([0.03], index=dates, columns=["A"])

        result = factor.compute(daily_peak_climber=data, T=20)
        assert result.iloc[0, 0] == pytest.approx(0.03, rel=1e-10)

    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        values = np.ones(10) * 0.1
        values[3] = np.nan
        data = pd.DataFrame(values, index=dates, columns=["A"])

        result = factor.compute(daily_peak_climber=data, T=5)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(np.nan, index=dates, columns=["A"])

        result = factor.compute(daily_peak_climber=data, T=5)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(0.0, index=dates, columns=["A"])

        result = factor.compute(daily_peak_climber=data, T=5)
        for val in result["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)


class TestPeakClimberOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        data = pd.DataFrame(
            np.random.uniform(-0.1, 0.1, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_peak_climber=data, T=20)

        assert result.shape == data.shape
        assert list(result.columns) == list(data.columns)
        assert list(result.index) == list(data.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame([0.1, 0.2, 0.3, 0.4, 0.5], index=dates, columns=["A"])

        result = factor.compute(daily_peak_climber=data, T=3)
        assert isinstance(result, pd.DataFrame)
