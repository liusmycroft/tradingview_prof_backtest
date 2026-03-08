import numpy as np
import pandas as pd
import pytest

from factors.minute_ideal_amplitude import MinuteIdealAmplitudeFactor


@pytest.fixture
def factor():
    return MinuteIdealAmplitudeFactor()


class TestMinuteIdealAmplitudeMetadata:
    def test_name(self, factor):
        assert factor.name == "MINUTE_IDEAL_AMPLITUDE"

    def test_category(self, factor):
        assert factor.category == "高频波动跳跃"

    def test_repr(self, factor):
        assert "MINUTE_IDEAL_AMPLITUDE" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "MINUTE_IDEAL_AMPLITUDE"
        assert meta["category"] == "高频波动跳跃"


class TestMinuteIdealAmplitudeHandCalculated:
    """手算验证 V(lambda) = V_high - V_low"""

    def test_simple_values(self, factor):
        """简单数值验证。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        v_high = pd.DataFrame([0.05, 0.06, 0.07], index=dates, columns=stocks)
        v_low = pd.DataFrame([0.02, 0.03, 0.04], index=dates, columns=stocks)

        result = factor.compute(daily_v_high=v_high, daily_v_low=v_low)

        assert result.iloc[0, 0] == pytest.approx(0.03, rel=1e-10)
        assert result.iloc[1, 0] == pytest.approx(0.03, rel=1e-10)
        assert result.iloc[2, 0] == pytest.approx(0.03, rel=1e-10)

    def test_equal_amplitudes(self, factor):
        """V_high == V_low 时结果应为 0。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        v = pd.DataFrame(0.04, index=dates, columns=stocks)

        result = factor.compute(daily_v_high=v, daily_v_low=v)
        np.testing.assert_array_almost_equal(result.values, 0.0)

    def test_negative_result(self, factor):
        """V_high < V_low 时结果为负。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        v_high = pd.DataFrame([0.02, 0.03, 0.01], index=dates, columns=stocks)
        v_low = pd.DataFrame([0.05, 0.06, 0.04], index=dates, columns=stocks)

        result = factor.compute(daily_v_high=v_high, daily_v_low=v_low)

        assert result.iloc[0, 0] == pytest.approx(-0.03, rel=1e-10)
        assert result.iloc[1, 0] == pytest.approx(-0.03, rel=1e-10)
        assert result.iloc[2, 0] == pytest.approx(-0.03, rel=1e-10)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A", "B"]

        v_high = pd.DataFrame({"A": [0.05, 0.06, 0.07], "B": [0.10, 0.12, 0.14]}, index=dates)
        v_low = pd.DataFrame({"A": [0.02, 0.03, 0.04], "B": [0.08, 0.09, 0.10]}, index=dates)

        result = factor.compute(daily_v_high=v_high, daily_v_low=v_low)

        assert result.iloc[0, 0] == pytest.approx(0.03, rel=1e-10)
        assert result.iloc[0, 1] == pytest.approx(0.02, rel=1e-10)


class TestMinuteIdealAmplitudeEdgeCases:
    def test_nan_in_input(self, factor):
        """输入含 NaN 时不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        v_high = pd.DataFrame([0.05, np.nan, 0.07, 0.08, 0.09], index=dates, columns=stocks)
        v_low = pd.DataFrame([0.02, 0.03, 0.04, 0.05, 0.06], index=dates, columns=stocks)

        result = factor.compute(daily_v_high=v_high, daily_v_low=v_low)
        assert isinstance(result, pd.DataFrame)
        assert np.isnan(result.iloc[1, 0])

    def test_all_nan(self, factor):
        """全 NaN 输入时结果应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        nans = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_v_high=nans, daily_v_low=nans)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        """全零输入时结果应全为 0。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        zeros = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_v_high=zeros, daily_v_low=zeros)
        np.testing.assert_array_almost_equal(result.values, 0.0)


class TestMinuteIdealAmplitudeOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        v_high = pd.DataFrame(np.random.uniform(0.03, 0.1, (30, 3)), index=dates, columns=stocks)
        v_low = pd.DataFrame(np.random.uniform(0.01, 0.05, (30, 3)), index=dates, columns=stocks)

        result = factor.compute(daily_v_high=v_high, daily_v_low=v_low)
        assert result.shape == v_high.shape
        assert list(result.columns) == list(v_high.columns)
        assert list(result.index) == list(v_high.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        v_high = pd.DataFrame([0.05] * 5, index=dates, columns=stocks)
        v_low = pd.DataFrame([0.02] * 5, index=dates, columns=stocks)

        result = factor.compute(daily_v_high=v_high, daily_v_low=v_low)
        assert isinstance(result, pd.DataFrame)

    def test_no_nan_in_output(self, factor):
        """正常输入时输出不应有 NaN。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        v_high = pd.DataFrame(np.random.uniform(0.03, 0.1, (10, 2)), index=dates, columns=stocks)
        v_low = pd.DataFrame(np.random.uniform(0.01, 0.05, (10, 2)), index=dates, columns=stocks)

        result = factor.compute(daily_v_high=v_high, daily_v_low=v_low)
        assert result.notna().all().all()
