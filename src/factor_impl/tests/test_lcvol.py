import numpy as np
import pandas as pd
import pytest

from factors.lcvol import LCVOLFactor


@pytest.fixture
def factor():
    return LCVOLFactor()


class TestLCVOLMetadata:
    def test_name(self, factor):
        assert factor.name == "LCVOL"

    def test_category(self, factor):
        assert factor.category == "高频成交分布"

    def test_repr(self, factor):
        assert "LCVOL" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "LCVOL"
        assert meta["category"] == "高频成交分布"


class TestLCVOLHandCalculated:
    """用手算数据验证 EWM(span=T, min_periods=1) 计算的正确性。"""

    def test_constant_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        daily_lcvol = pd.DataFrame(0.15, index=dates, columns=stocks)

        result = factor.compute(daily_lcvol=daily_lcvol, T=20)

        np.testing.assert_array_almost_equal(result["A"].values, 0.15)

    def test_ema_manual_T3(self, factor):
        """T=3, 手动验证 EMA 值。"""
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        stocks = ["A"]
        daily_lcvol = pd.DataFrame(
            [10.0, 20.0, 30.0, 40.0], index=dates, columns=stocks
        )

        result = factor.compute(daily_lcvol=daily_lcvol, T=3)

        assert result.iloc[0, 0] == pytest.approx(10.0, rel=1e-6)
        assert result.iloc[1, 0] == pytest.approx(50 / 3, rel=1e-6)
        assert result.iloc[2, 0] == pytest.approx(170 / 7, rel=1e-6)
        assert result.iloc[3, 0] == pytest.approx(490 / 15, rel=1e-6)

    def test_ema_recent_weight(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        vals = [0.05] * 5 + [0.25] * 5
        daily_lcvol = pd.DataFrame(vals, index=dates, columns=stocks)

        result = factor.compute(daily_lcvol=daily_lcvol, T=5)
        assert result.iloc[-1, 0] > 0.15

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        daily_lcvol = pd.DataFrame(
            {"A": [0.10] * 10, "B": [0.30] * 10}, index=dates
        )

        result = factor.compute(daily_lcvol=daily_lcvol, T=5)

        np.testing.assert_array_almost_equal(result["A"].values, 0.10)
        np.testing.assert_array_almost_equal(result["B"].values, 0.30)


class TestLCVOLEdgeCases:
    def test_single_value(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A"]
        daily_lcvol = pd.DataFrame([0.12], index=dates, columns=stocks)

        result = factor.compute(daily_lcvol=daily_lcvol, T=20)
        assert result.iloc[0, 0] == pytest.approx(0.12, rel=1e-10)

    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        values = np.ones(10) * 0.15
        values[3] = np.nan
        daily_lcvol = pd.DataFrame(values, index=dates, columns=stocks)

        result = factor.compute(daily_lcvol=daily_lcvol, T=5)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily_lcvol = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_lcvol=daily_lcvol, T=5)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily_lcvol = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_lcvol=daily_lcvol, T=5)
        for val in result["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)


class TestLCVOLOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily_lcvol = pd.DataFrame(
            np.random.uniform(0.0, 0.5, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_lcvol=daily_lcvol, T=20)

        assert result.shape == daily_lcvol.shape
        assert list(result.columns) == list(daily_lcvol.columns)
        assert list(result.index) == list(daily_lcvol.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_lcvol = pd.DataFrame(
            [0.1, 0.2, 0.3, 0.2, 0.1], index=dates, columns=stocks
        )

        result = factor.compute(daily_lcvol=daily_lcvol, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_no_leading_nan_min_periods_1(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        daily_lcvol = pd.DataFrame(
            np.random.uniform(0.0, 0.5, (10, 2)), index=dates, columns=stocks
        )

        result = factor.compute(daily_lcvol=daily_lcvol, T=20)
        assert result.iloc[0].notna().all()
        assert result.notna().all().all()
