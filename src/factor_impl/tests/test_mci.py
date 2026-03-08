import numpy as np
import pandas as pd
import pytest

from factors.mci import MCIFactor


@pytest.fixture
def factor():
    return MCIFactor()


class TestMCIMetadata:
    def test_name(self, factor):
        assert factor.name == "MCI"

    def test_category(self, factor):
        assert factor.category == "高频流动性"

    def test_repr(self, factor):
        assert "MCI" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "MCI"
        assert meta["category"] == "高频流动性"


class TestMCIHandCalculated:
    def test_constant_input(self, factor):
        """常数输入时，EMA 应等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(5.0, index=dates, columns=stocks)

        result = factor.compute(daily_mci=data, T=20)

        np.testing.assert_array_almost_equal(result["A"].values, 5.0)

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
        data = pd.DataFrame([10.0, 20.0, 30.0, 40.0], index=dates, columns=["A"])

        result = factor.compute(daily_mci=data, T=3)

        assert result.iloc[0, 0] == pytest.approx(10.0, rel=1e-6)
        assert result.iloc[1, 0] == pytest.approx(50 / 3, rel=1e-6)
        assert result.iloc[2, 0] == pytest.approx(170 / 7, rel=1e-6)
        assert result.iloc[3, 0] == pytest.approx(490 / 15, rel=1e-6)

    def test_ema_recent_weight(self, factor):
        """EMA 应赋予近期数据更高权重。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        vals = [1.0] * 5 + [10.0] * 5
        data = pd.DataFrame(vals, index=dates, columns=["A"])

        result = factor.compute(daily_mci=data, T=5)
        assert result.iloc[-1, 0] > 5.5

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(
            {"A": [2.0] * 10, "B": [8.0] * 10}, index=dates
        )

        result = factor.compute(daily_mci=data, T=5)

        np.testing.assert_array_almost_equal(result["A"].values, 2.0)
        np.testing.assert_array_almost_equal(result["B"].values, 8.0)


class TestMCIEdgeCases:
    def test_single_value(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        data = pd.DataFrame([3.5], index=dates, columns=["A"])

        result = factor.compute(daily_mci=data, T=20)
        assert result.iloc[0, 0] == pytest.approx(3.5, rel=1e-10)

    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        values = np.ones(10) * 5.0
        values[3] = np.nan
        data = pd.DataFrame(values, index=dates, columns=["A"])

        result = factor.compute(daily_mci=data, T=5)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(np.nan, index=dates, columns=["A"])

        result = factor.compute(daily_mci=data, T=5)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(0.0, index=dates, columns=["A"])

        result = factor.compute(daily_mci=data, T=5)
        for val in result["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)


class TestMCIOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        data = pd.DataFrame(
            np.random.uniform(0.1, 10.0, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_mci=data, T=20)

        assert result.shape == data.shape
        assert list(result.columns) == list(data.columns)
        assert list(result.index) == list(data.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0], index=dates, columns=["A"])

        result = factor.compute(daily_mci=data, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_no_leading_nan_min_periods_1(self, factor):
        """min_periods=1, 第一行就有值。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(
            np.random.uniform(1, 100, (10, 2)), index=dates, columns=["A", "B"]
        )

        result = factor.compute(daily_mci=data, T=20)
        assert result.iloc[0].notna().all()
        assert result.notna().all().all()
