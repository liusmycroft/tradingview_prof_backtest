import numpy as np
import pandas as pd
import pytest

from factors.chip_distribution_shape import (
    ChipDistributionKurtosisFactor,
    ChipDistributionSkewnessFactor,
)


@pytest.fixture
def kurtosis_factor():
    return ChipDistributionKurtosisFactor()


@pytest.fixture
def skewness_factor():
    return ChipDistributionSkewnessFactor()


class TestChipDistributionKurtosisMetadata:
    def test_name(self, kurtosis_factor):
        assert kurtosis_factor.name == "CHIP_DISTRIBUTION_KURTOSIS"

    def test_category(self, kurtosis_factor):
        assert kurtosis_factor.category == "行为金融-筹码分布"

    def test_repr(self, kurtosis_factor):
        assert "CHIP_DISTRIBUTION_KURTOSIS" in repr(kurtosis_factor)

    def test_get_metadata(self, kurtosis_factor):
        meta = kurtosis_factor.get_metadata()
        assert meta["name"] == "CHIP_DISTRIBUTION_KURTOSIS"
        assert meta["category"] == "行为金融-筹码分布"


class TestChipDistributionSkewnessMetadata:
    def test_name(self, skewness_factor):
        assert skewness_factor.name == "CHIP_DISTRIBUTION_SKEWNESS"

    def test_category(self, skewness_factor):
        assert skewness_factor.category == "行为金融-筹码分布"

    def test_repr(self, skewness_factor):
        assert "CHIP_DISTRIBUTION_SKEWNESS" in repr(skewness_factor)

    def test_get_metadata(self, skewness_factor):
        meta = skewness_factor.get_metadata()
        assert meta["name"] == "CHIP_DISTRIBUTION_SKEWNESS"
        assert meta["category"] == "行为金融-筹码分布"


class TestChipDistributionKurtosisHandCalculated:
    def test_constant_input(self, kurtosis_factor):
        """常数输入时，EMA 应等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        data = pd.DataFrame(3.0, index=dates, columns=["A"])

        result = kurtosis_factor.compute(daily_chip_kurtosis=data, T=20)

        np.testing.assert_array_almost_equal(result["A"].values, 3.0)

    def test_ema_manual_T3(self, kurtosis_factor):
        """T=3, 手动验证 EMA 值。"""
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        data = pd.DataFrame([2.0, 3.0, 4.0, 5.0], index=dates, columns=["A"])

        result = kurtosis_factor.compute(daily_chip_kurtosis=data, T=3)

        assert result.iloc[0, 0] == pytest.approx(2.0, rel=1e-6)
        # (0.5*2 + 1.0*3) / 1.5 = 2.6667
        assert result.iloc[1, 0] == pytest.approx(4.0 / 1.5, rel=1e-6)
        # (0.25*2 + 0.5*3 + 1.0*4) / 1.75 = 3.4286
        assert result.iloc[2, 0] == pytest.approx(6.0 / 1.75, rel=1e-6)
        # (0.125*2 + 0.25*3 + 0.5*4 + 1.0*5) / 1.875 = 4.2667
        assert result.iloc[3, 0] == pytest.approx(8.0 / 1.875, rel=1e-6)

    def test_two_stocks_independent(self, kurtosis_factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(
            {"A": [3.0] * 10, "B": [5.0] * 10}, index=dates
        )

        result = kurtosis_factor.compute(daily_chip_kurtosis=data, T=5)

        np.testing.assert_array_almost_equal(result["A"].values, 3.0)
        np.testing.assert_array_almost_equal(result["B"].values, 5.0)


class TestChipDistributionSkewnessHandCalculated:
    def test_constant_input(self, skewness_factor):
        """常数输入时，EMA 应等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        data = pd.DataFrame(-0.5, index=dates, columns=["A"])

        result = skewness_factor.compute(daily_chip_skewness=data, T=20)

        np.testing.assert_array_almost_equal(result["A"].values, -0.5)

    def test_ema_manual_T3(self, skewness_factor):
        """T=3, 手动验证 EMA 值。"""
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        data = pd.DataFrame([-0.2, 0.1, 0.3, -0.1], index=dates, columns=["A"])

        result = skewness_factor.compute(daily_chip_skewness=data, T=3)

        assert result.iloc[0, 0] == pytest.approx(-0.2, rel=1e-6)
        # (0.5*(-0.2) + 1.0*0.1) / 1.5 = 0.0
        assert result.iloc[1, 0] == pytest.approx(0.0, abs=1e-10)
        # (0.25*(-0.2) + 0.5*0.1 + 1.0*0.3) / 1.75 = 0.3/1.75
        assert result.iloc[2, 0] == pytest.approx(0.3 / 1.75, rel=1e-6)
        # (0.125*(-0.2) + 0.25*0.1 + 0.5*0.3 + 1.0*(-0.1)) / 1.875
        expected_3 = (-0.025 + 0.025 + 0.15 - 0.1) / 1.875
        assert result.iloc[3, 0] == pytest.approx(expected_3, rel=1e-6)


class TestChipDistributionEdgeCases:
    def test_single_value_kurtosis(self, kurtosis_factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        data = pd.DataFrame([4.2], index=dates, columns=["A"])

        result = kurtosis_factor.compute(daily_chip_kurtosis=data, T=20)
        assert result.iloc[0, 0] == pytest.approx(4.2, rel=1e-10)

    def test_single_value_skewness(self, skewness_factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        data = pd.DataFrame([-0.3], index=dates, columns=["A"])

        result = skewness_factor.compute(daily_chip_skewness=data, T=20)
        assert result.iloc[0, 0] == pytest.approx(-0.3, rel=1e-10)

    def test_all_nan_kurtosis(self, kurtosis_factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(np.nan, index=dates, columns=["A"])

        result = kurtosis_factor.compute(daily_chip_kurtosis=data, T=5)
        assert result.isna().all().all()

    def test_all_nan_skewness(self, skewness_factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(np.nan, index=dates, columns=["A"])

        result = skewness_factor.compute(daily_chip_skewness=data, T=5)
        assert result.isna().all().all()

    def test_nan_in_input_kurtosis(self, kurtosis_factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        values = np.ones(10) * 3.0
        values[3] = np.nan
        data = pd.DataFrame(values, index=dates, columns=["A"])

        result = kurtosis_factor.compute(daily_chip_kurtosis=data, T=5)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_zero_input_kurtosis(self, kurtosis_factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(0.0, index=dates, columns=["A"])

        result = kurtosis_factor.compute(daily_chip_kurtosis=data, T=5)
        for val in result["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)


class TestChipDistributionOutputShape:
    def test_output_shape_kurtosis(self, kurtosis_factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        data = pd.DataFrame(
            np.random.uniform(2, 6, (30, 3)), index=dates, columns=stocks
        )

        result = kurtosis_factor.compute(daily_chip_kurtosis=data, T=20)

        assert result.shape == data.shape
        assert list(result.columns) == list(data.columns)
        assert isinstance(result, pd.DataFrame)

    def test_output_shape_skewness(self, skewness_factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        data = pd.DataFrame(
            np.random.uniform(-1, 1, (30, 3)), index=dates, columns=stocks
        )

        result = skewness_factor.compute(daily_chip_skewness=data, T=20)

        assert result.shape == data.shape
        assert list(result.columns) == list(data.columns)
        assert isinstance(result, pd.DataFrame)

    def test_no_leading_nan_kurtosis(self, kurtosis_factor):
        """min_periods=1, 第一行就有值。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(
            np.random.uniform(2, 5, (10, 2)), index=dates, columns=["A", "B"]
        )

        result = kurtosis_factor.compute(daily_chip_kurtosis=data, T=20)
        assert result.iloc[0].notna().all()
        assert result.notna().all().all()
