import numpy as np
import pandas as pd
import pytest

from factors.active_chip_ratio import ActiveChipRatioFactor


@pytest.fixture
def factor():
    return ActiveChipRatioFactor()


class TestActiveChipRatioMetadata:
    def test_name(self, factor):
        assert factor.name == "ACTIVE_CHIP_RATIO"

    def test_category(self, factor):
        assert factor.category == "行为金融筹码"

    def test_repr(self, factor):
        assert "ACTIVE_CHIP_RATIO" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "ACTIVE_CHIP_RATIO"
        assert meta["category"] == "行为金融筹码"


class TestActiveChipRatioHandCalculated:
    def test_constant_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        data = pd.DataFrame(0.5, index=dates, columns=["A"])
        result = factor.compute(daily_active_chip_ratio=data, T=20)
        np.testing.assert_array_almost_equal(result["A"].values, 0.5)

    def test_ema_manual_T3(self, factor):
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        data = pd.DataFrame([0.1, 0.2, 0.3, 0.4], index=dates, columns=["A"])
        result = factor.compute(daily_active_chip_ratio=data, T=3)
        assert result.iloc[0, 0] == pytest.approx(0.1, rel=1e-6)
        assert result.iloc[1, 0] == pytest.approx(0.5 / 3, rel=1e-6)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame({"A": [0.3] * 10, "B": [0.7] * 10}, index=dates)
        result = factor.compute(daily_active_chip_ratio=data, T=5)
        np.testing.assert_array_almost_equal(result["A"].values, 0.3)
        np.testing.assert_array_almost_equal(result["B"].values, 0.7)


class TestActiveChipRatioEdgeCases:
    def test_single_value(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        data = pd.DataFrame([0.42], index=dates, columns=["A"])
        result = factor.compute(daily_active_chip_ratio=data, T=20)
        assert result.iloc[0, 0] == pytest.approx(0.42, rel=1e-10)

    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        values = np.ones(10) * 0.5
        values[3] = np.nan
        data = pd.DataFrame(values, index=dates, columns=["A"])
        result = factor.compute(daily_active_chip_ratio=data, T=5)
        assert isinstance(result, pd.DataFrame)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(np.nan, index=dates, columns=["A"])
        result = factor.compute(daily_active_chip_ratio=data, T=5)
        assert result.isna().all().all()


class TestActiveChipRatioOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        data = pd.DataFrame(
            np.random.uniform(0.1, 0.9, (30, 3)), index=dates, columns=stocks
        )
        result = factor.compute(daily_active_chip_ratio=data, T=20)
        assert result.shape == data.shape
        assert list(result.columns) == list(data.columns)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame([0.1, 0.2, 0.3, 0.4, 0.5], index=dates, columns=["A"])
        result = factor.compute(daily_active_chip_ratio=data, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_no_leading_nan_min_periods_1(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(
            np.random.uniform(0.1, 0.9, (10, 2)), index=dates, columns=["A", "B"]
        )
        result = factor.compute(daily_active_chip_ratio=data, T=20)
        assert result.iloc[0].notna().all()
