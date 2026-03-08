import numpy as np
import pandas as pd
import pytest

from factors.chip_distribution_cv import ChipDistributionCVFactor


@pytest.fixture
def factor():
    return ChipDistributionCVFactor()


class TestChipDistributionCVMetadata:
    def test_name(self, factor):
        assert factor.name == "CHIP_DISTRIBUTION_CV"

    def test_category(self, factor):
        assert factor.category == "行为金融-筹码分布"

    def test_repr(self, factor):
        assert "CHIP_DISTRIBUTION_CV" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "CHIP_DISTRIBUTION_CV"
        assert meta["category"] == "行为金融-筹码分布"


class TestChipDistributionCVHandCalculated:
    def test_constant_cv(self, factor):
        """std=2, mean=10 => cv=0.2, rolling mean of constant = 0.2."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        chip_std = pd.DataFrame(2.0, index=dates, columns=stocks)
        chip_mean = pd.DataFrame(10.0, index=dates, columns=stocks)

        result = factor.compute(chip_std=chip_std, chip_mean=chip_mean, T=5)
        np.testing.assert_array_almost_equal(result["A"].values, 0.2)

    def test_manual_rolling(self, factor):
        """Hand-calculated CV with T=3, min_periods=1.
        cv = [1/10, 2/10, 3/10, 4/10] = [0.1, 0.2, 0.3, 0.4]
        rolling(3, min_periods=1):
          day0: 0.1
          day1: (0.1+0.2)/2 = 0.15
          day2: (0.1+0.2+0.3)/3 = 0.2
          day3: (0.2+0.3+0.4)/3 = 0.3
        """
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        stocks = ["A"]
        chip_std = pd.DataFrame([1.0, 2.0, 3.0, 4.0], index=dates, columns=stocks)
        chip_mean = pd.DataFrame(10.0, index=dates, columns=stocks)

        result = factor.compute(chip_std=chip_std, chip_mean=chip_mean, T=3)
        expected = [0.1, 0.15, 0.2, 0.3]
        np.testing.assert_array_almost_equal(result["A"].values, expected)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        chip_std = pd.DataFrame({"A": [1.0] * 10, "B": [3.0] * 10}, index=dates)
        chip_mean = pd.DataFrame({"A": [10.0] * 10, "B": [10.0] * 10}, index=dates)

        result = factor.compute(chip_std=chip_std, chip_mean=chip_mean, T=5)
        np.testing.assert_array_almost_equal(result["A"].values, 0.1)
        np.testing.assert_array_almost_equal(result["B"].values, 0.3)


class TestChipDistributionCVEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        std_vals = np.ones(10) * 2.0
        std_vals[3] = np.nan
        chip_std = pd.DataFrame(std_vals, index=dates, columns=stocks)
        chip_mean = pd.DataFrame(10.0, index=dates, columns=stocks)

        result = factor.compute(chip_std=chip_std, chip_mean=chip_mean, T=5)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        chip_std = pd.DataFrame(np.nan, index=dates, columns=stocks)
        chip_mean = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(chip_std=chip_std, chip_mean=chip_mean, T=5)
        assert result.isna().all().all()

    def test_zero_mean(self, factor):
        """mean=0 => cv=inf or NaN."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        chip_std = pd.DataFrame(1.0, index=dates, columns=stocks)
        chip_mean = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(chip_std=chip_std, chip_mean=chip_mean, T=3)
        assert isinstance(result, pd.DataFrame)


class TestChipDistributionCVOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        chip_std = pd.DataFrame(
            np.random.uniform(0.5, 3, (30, 3)), index=dates, columns=stocks
        )
        chip_mean = pd.DataFrame(
            np.random.uniform(5, 20, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(chip_std=chip_std, chip_mean=chip_mean, T=20)
        assert result.shape == chip_std.shape
        assert list(result.columns) == list(chip_std.columns)
        assert list(result.index) == list(chip_std.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        chip_std = pd.DataFrame(2.0, index=dates, columns=stocks)
        chip_mean = pd.DataFrame(10.0, index=dates, columns=stocks)

        result = factor.compute(chip_std=chip_std, chip_mean=chip_mean, T=3)
        assert isinstance(result, pd.DataFrame)
