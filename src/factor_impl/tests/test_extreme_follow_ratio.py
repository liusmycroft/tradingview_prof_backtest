import numpy as np
import pandas as pd
import pytest

from factors.extreme_follow_ratio import ExtremeFollowRatioFactor


@pytest.fixture
def factor():
    return ExtremeFollowRatioFactor()


class TestExtremeFollowRatioMetadata:
    def test_name(self, factor):
        assert factor.name == "EXTREME_FOLLOW_RATIO"

    def test_category(self, factor):
        assert factor.category == "量价因子改进"

    def test_repr(self, factor):
        assert "EXTREME_FOLLOW_RATIO" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "EXTREME_FOLLOW_RATIO"
        assert meta["category"] == "量价因子改进"


class TestExtremeFollowRatioHandCalculated:
    """用手算数据验证 ratio + rolling(window=T, min_periods=1).mean() 的正确性。"""

    def test_constant_equal_inputs(self, factor):
        """follow == reverse => ratio = 1.0, rolling mean = 1.0"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        daily_follow = pd.DataFrame(0.5, index=dates, columns=stocks)
        daily_reverse = pd.DataFrame(0.5, index=dates, columns=stocks)

        result = factor.compute(
            daily_follow_ratio=daily_follow,
            daily_reverse_ratio=daily_reverse,
            T=20,
        )

        np.testing.assert_array_almost_equal(result["A"].values, 1.0)

    def test_ratio_manual(self, factor):
        """Manual ratio calculation.

        follow = [2, 4, 6], reverse = [1, 2, 3]
        ratio = [2, 2, 2]
        rolling(3, min_periods=1).mean():
          mean_0 = 2.0
          mean_1 = 2.0
          mean_2 = 2.0
        """
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]
        daily_follow = pd.DataFrame([2.0, 4.0, 6.0], index=dates, columns=stocks)
        daily_reverse = pd.DataFrame([1.0, 2.0, 3.0], index=dates, columns=stocks)

        result = factor.compute(
            daily_follow_ratio=daily_follow,
            daily_reverse_ratio=daily_reverse,
            T=3,
        )

        np.testing.assert_array_almost_equal(result["A"].values, 2.0)

    def test_varying_ratio(self, factor):
        """
        follow = [3, 6, 9, 12], reverse = [1, 2, 3, 4]
        ratio = [3, 3, 3, 3]
        rolling(2, min_periods=1).mean():
          mean_0 = 3.0
          mean_1 = 3.0
          mean_2 = 3.0
          mean_3 = 3.0
        """
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        stocks = ["A"]
        daily_follow = pd.DataFrame([3.0, 6.0, 9.0, 12.0], index=dates, columns=stocks)
        daily_reverse = pd.DataFrame([1.0, 2.0, 3.0, 4.0], index=dates, columns=stocks)

        result = factor.compute(
            daily_follow_ratio=daily_follow,
            daily_reverse_ratio=daily_reverse,
            T=2,
        )

        np.testing.assert_array_almost_equal(result["A"].values, 3.0)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        daily_follow = pd.DataFrame(
            {"A": [2.0] * 10, "B": [6.0] * 10}, index=dates
        )
        daily_reverse = pd.DataFrame(
            {"A": [1.0] * 10, "B": [2.0] * 10}, index=dates
        )

        result = factor.compute(
            daily_follow_ratio=daily_follow,
            daily_reverse_ratio=daily_reverse,
            T=5,
        )

        np.testing.assert_array_almost_equal(result["A"].values, 2.0)
        np.testing.assert_array_almost_equal(result["B"].values, 3.0)


class TestExtremeFollowRatioEdgeCases:
    def test_zero_reverse_produces_nan(self, factor):
        """reverse = 0 should produce NaN (division by zero handled)."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_follow = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0], index=dates, columns=stocks)
        daily_reverse = pd.DataFrame([0.0, 0.0, 0.0, 0.0, 0.0], index=dates, columns=stocks)

        result = factor.compute(
            daily_follow_ratio=daily_follow,
            daily_reverse_ratio=daily_reverse,
            T=3,
        )
        assert result.isna().all().all()

    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        follow_vals = np.ones(10) * 2.0
        follow_vals[3] = np.nan
        reverse_vals = np.ones(10) * 1.0

        daily_follow = pd.DataFrame(follow_vals, index=dates, columns=stocks)
        daily_reverse = pd.DataFrame(reverse_vals, index=dates, columns=stocks)

        result = factor.compute(
            daily_follow_ratio=daily_follow,
            daily_reverse_ratio=daily_reverse,
            T=5,
        )
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily_follow = pd.DataFrame(np.nan, index=dates, columns=stocks)
        daily_reverse = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(
            daily_follow_ratio=daily_follow,
            daily_reverse_ratio=daily_reverse,
            T=5,
        )
        assert result.isna().all().all()

    def test_single_value(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A"]
        daily_follow = pd.DataFrame([4.0], index=dates, columns=stocks)
        daily_reverse = pd.DataFrame([2.0], index=dates, columns=stocks)

        result = factor.compute(
            daily_follow_ratio=daily_follow,
            daily_reverse_ratio=daily_reverse,
            T=20,
        )
        assert result.iloc[0, 0] == pytest.approx(2.0, rel=1e-10)


class TestExtremeFollowRatioOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily_follow = pd.DataFrame(
            np.random.uniform(1, 10, (30, 3)), index=dates, columns=stocks
        )
        daily_reverse = pd.DataFrame(
            np.random.uniform(1, 10, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(
            daily_follow_ratio=daily_follow,
            daily_reverse_ratio=daily_reverse,
            T=20,
        )

        assert result.shape == daily_follow.shape
        assert list(result.columns) == list(daily_follow.columns)
        assert list(result.index) == list(daily_follow.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_follow = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0], index=dates, columns=stocks)
        daily_reverse = pd.DataFrame([1.0, 1.0, 1.0, 1.0, 1.0], index=dates, columns=stocks)

        result = factor.compute(
            daily_follow_ratio=daily_follow,
            daily_reverse_ratio=daily_reverse,
            T=3,
        )
        assert isinstance(result, pd.DataFrame)
