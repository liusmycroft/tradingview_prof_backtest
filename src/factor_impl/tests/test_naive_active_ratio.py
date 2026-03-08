import numpy as np
import pandas as pd
import pytest

from factors.naive_active_ratio import NaiveActiveRatioFactor


@pytest.fixture
def factor():
    return NaiveActiveRatioFactor()


class TestNaiveActiveRatioMetadata:
    def test_name(self, factor):
        assert factor.name == "NAIVE_ACTIVE_RATIO"

    def test_category(self, factor):
        assert factor.category == "高频资金流"

    def test_repr(self, factor):
        assert "NAIVE_ACTIVE_RATIO" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "NAIVE_ACTIVE_RATIO"
        assert meta["category"] == "高频资金流"


class TestNaiveActiveRatioHandCalculated:
    def test_constant_input(self, factor):
        """常数输入时，EMA 应等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        data = pd.DataFrame(0.55, index=dates, columns=["A"])

        result = factor.compute(daily_naive_active_ratio=data, T=20)

        np.testing.assert_array_almost_equal(result["A"].values, 0.55)

    def test_ema_manual_T3(self, factor):
        """T=3, 手动验证 EMA 值。

        ewm(span=3, adjust=True) alpha = 0.5
        data = [0.4, 0.5, 0.6, 0.7]
          ema_0 = 0.4
          ema_1 = (0.5*0.4 + 1.0*0.5) / 1.5 = 0.4667
          ema_2 = (0.25*0.4 + 0.5*0.5 + 1.0*0.6) / 1.75 = 0.5429
          ema_3 = (0.125*0.4 + 0.25*0.5 + 0.5*0.6 + 1.0*0.7) / 1.875 = 0.6333
        """
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        data = pd.DataFrame([0.4, 0.5, 0.6, 0.7], index=dates, columns=["A"])

        result = factor.compute(daily_naive_active_ratio=data, T=3)

        assert result.iloc[0, 0] == pytest.approx(0.4, rel=1e-6)
        assert result.iloc[1, 0] == pytest.approx(0.7 / 1.5, rel=1e-6)
        assert result.iloc[2, 0] == pytest.approx(0.95 / 1.75, rel=1e-6)
        assert result.iloc[3, 0] == pytest.approx(1.175 / 1.875, rel=1e-6)

    def test_ema_recent_weight(self, factor):
        """EMA 应赋予近期数据更高权重。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        vals = [0.3] * 5 + [0.8] * 5
        data = pd.DataFrame(vals, index=dates, columns=["A"])

        result = factor.compute(daily_naive_active_ratio=data, T=5)
        assert result.iloc[-1, 0] > 0.55

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(
            {"A": [0.4] * 10, "B": [0.6] * 10}, index=dates
        )

        result = factor.compute(daily_naive_active_ratio=data, T=5)

        np.testing.assert_array_almost_equal(result["A"].values, 0.4)
        np.testing.assert_array_almost_equal(result["B"].values, 0.6)


class TestNaiveActiveRatioEdgeCases:
    def test_single_value(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        data = pd.DataFrame([0.52], index=dates, columns=["A"])

        result = factor.compute(daily_naive_active_ratio=data, T=20)
        assert result.iloc[0, 0] == pytest.approx(0.52, rel=1e-10)

    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        values = np.ones(10) * 0.5
        values[3] = np.nan
        data = pd.DataFrame(values, index=dates, columns=["A"])

        result = factor.compute(daily_naive_active_ratio=data, T=5)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(np.nan, index=dates, columns=["A"])

        result = factor.compute(daily_naive_active_ratio=data, T=5)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(0.0, index=dates, columns=["A"])

        result = factor.compute(daily_naive_active_ratio=data, T=5)
        for val in result["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)


class TestNaiveActiveRatioOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        data = pd.DataFrame(
            np.random.uniform(0.3, 0.7, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_naive_active_ratio=data, T=20)

        assert result.shape == data.shape
        assert list(result.columns) == list(data.columns)
        assert list(result.index) == list(data.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame(
            [0.4, 0.5, 0.6, 0.5, 0.4], index=dates, columns=["A"]
        )

        result = factor.compute(daily_naive_active_ratio=data, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_no_leading_nan_min_periods_1(self, factor):
        """min_periods=1, 第一行就有值。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(
            np.random.uniform(0.3, 0.7, (10, 2)), index=dates, columns=["A", "B"]
        )

        result = factor.compute(daily_naive_active_ratio=data, T=20)
        assert result.iloc[0].notna().all()
        assert result.notna().all().all()
