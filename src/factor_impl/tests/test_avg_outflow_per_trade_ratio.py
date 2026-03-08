import numpy as np
import pandas as pd
import pytest

from factors.avg_outflow_per_trade_ratio import AvgOutflowPerTradeRatioFactor


@pytest.fixture
def factor():
    return AvgOutflowPerTradeRatioFactor()


class TestAvgOutflowPerTradeRatioMetadata:
    def test_name(self, factor):
        assert factor.name == "AVG_OUTFLOW_PER_TRADE_RATIO"

    def test_category(self, factor):
        assert factor.category == "高频资金流"

    def test_repr(self, factor):
        assert "AVG_OUTFLOW_PER_TRADE_RATIO" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "AVG_OUTFLOW_PER_TRADE_RATIO"
        assert meta["category"] == "高频资金流"


class TestAvgOutflowPerTradeRatioHandCalculated:
    def test_constant_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        data = pd.DataFrame(0.8, index=dates, columns=["A"])
        result = factor.compute(daily_outflow_ratio=data, T=20)
        assert result.iloc[19, 0] == pytest.approx(0.8)

    def test_rolling_mean_T3(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame([0.5, 0.6, 0.7, 0.8, 0.9], index=dates, columns=["A"])
        result = factor.compute(daily_outflow_ratio=data, T=3)
        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(0.6)
        assert result.iloc[3, 0] == pytest.approx(0.7)
        assert result.iloc[4, 0] == pytest.approx(0.8)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame({"A": [1.0] * 5, "B": [2.0] * 5}, index=dates)
        result = factor.compute(daily_outflow_ratio=data, T=3)
        assert result.iloc[2, 0] == pytest.approx(1.0)
        assert result.iloc[2, 1] == pytest.approx(2.0)


class TestAvgOutflowPerTradeRatioEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame([0.5, np.nan, 0.7, 0.8, 0.9], index=dates, columns=["A"])
        result = factor.compute(daily_outflow_ratio=data, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(np.nan, index=dates, columns=["A"])
        result = factor.compute(daily_outflow_ratio=data, T=5)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame(0.0, index=dates, columns=["A"])
        result = factor.compute(daily_outflow_ratio=data, T=3)
        assert result.iloc[2, 0] == pytest.approx(0.0, abs=1e-15)


class TestAvgOutflowPerTradeRatioOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        data = pd.DataFrame(
            np.random.uniform(0.5, 1.5, (30, 3)), index=dates, columns=stocks
        )
        result = factor.compute(daily_outflow_ratio=data, T=20)
        assert result.shape == data.shape

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame([0.5, 0.6, 0.7, 0.8, 0.9], index=dates, columns=["A"])
        result = factor.compute(daily_outflow_ratio=data, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        data = pd.DataFrame(
            np.random.uniform(0.5, 1.5, (25, 2)), index=dates, columns=["A", "B"]
        )
        result = factor.compute(daily_outflow_ratio=data, T=20)
        assert result.iloc[:19].isna().all().all()
        assert result.iloc[19:].notna().all().all()
