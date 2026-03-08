import numpy as np
import pandas as pd
import pytest

from factors.mlqs import MLQSFactor


@pytest.fixture
def factor():
    return MLQSFactor()


class TestMLQSMetadata:
    def test_name(self, factor):
        assert factor.name == "MLQS"

    def test_category(self, factor):
        assert factor.category == "高频流动性"

    def test_repr(self, factor):
        assert "MLQS" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "MLQS"
        assert meta["category"] == "高频流动性"


class TestMLQSHandCalculated:
    """用手算数据验证 EWM(span=T, min_periods=1) 计算的正确性。"""

    def test_constant_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        daily_mlqs = pd.DataFrame(0.005, index=dates, columns=stocks)

        result = factor.compute(daily_mlqs=daily_mlqs, T=20)

        np.testing.assert_array_almost_equal(result["A"].values, 0.005)

    def test_ema_manual_T3(self, factor):
        """T=3, 手动验证 EMA 值。"""
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        stocks = ["A"]
        daily_mlqs = pd.DataFrame(
            [10.0, 20.0, 30.0, 40.0], index=dates, columns=stocks
        )

        result = factor.compute(daily_mlqs=daily_mlqs, T=3)

        assert result.iloc[0, 0] == pytest.approx(10.0, rel=1e-6)
        assert result.iloc[1, 0] == pytest.approx(50 / 3, rel=1e-6)
        assert result.iloc[2, 0] == pytest.approx(170 / 7, rel=1e-6)
        assert result.iloc[3, 0] == pytest.approx(490 / 15, rel=1e-6)

    def test_ema_recent_weight(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        vals = [0.001] * 5 + [0.010] * 5
        daily_mlqs = pd.DataFrame(vals, index=dates, columns=stocks)

        result = factor.compute(daily_mlqs=daily_mlqs, T=5)
        assert result.iloc[-1, 0] > 0.0055

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        daily_mlqs = pd.DataFrame(
            {"A": [0.003] * 10, "B": [0.008] * 10}, index=dates
        )

        result = factor.compute(daily_mlqs=daily_mlqs, T=5)

        np.testing.assert_array_almost_equal(result["A"].values, 0.003)
        np.testing.assert_array_almost_equal(result["B"].values, 0.008)


class TestMLQSEdgeCases:
    def test_single_value(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A"]
        daily_mlqs = pd.DataFrame([0.004], index=dates, columns=stocks)

        result = factor.compute(daily_mlqs=daily_mlqs, T=20)
        assert result.iloc[0, 0] == pytest.approx(0.004, rel=1e-10)

    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        values = np.ones(10) * 0.005
        values[3] = np.nan
        daily_mlqs = pd.DataFrame(values, index=dates, columns=stocks)

        result = factor.compute(daily_mlqs=daily_mlqs, T=5)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily_mlqs = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_mlqs=daily_mlqs, T=5)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily_mlqs = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_mlqs=daily_mlqs, T=5)
        for val in result["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)


class TestMLQSOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily_mlqs = pd.DataFrame(
            np.random.uniform(0.001, 0.01, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_mlqs=daily_mlqs, T=20)

        assert result.shape == daily_mlqs.shape
        assert list(result.columns) == list(daily_mlqs.columns)
        assert list(result.index) == list(daily_mlqs.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_mlqs = pd.DataFrame(
            [0.001, 0.002, 0.003, 0.004, 0.005], index=dates, columns=stocks
        )

        result = factor.compute(daily_mlqs=daily_mlqs, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_no_leading_nan_min_periods_1(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        daily_mlqs = pd.DataFrame(
            np.random.uniform(0.001, 0.01, (10, 2)), index=dates, columns=stocks
        )

        result = factor.compute(daily_mlqs=daily_mlqs, T=20)
        assert result.iloc[0].notna().all()
        assert result.notna().all().all()
