import numpy as np
import pandas as pd
import pytest

from factors.leading_volume_anomaly_at_min import LeadingVolumeAnomalyAtMinFactor


@pytest.fixture
def factor():
    return LeadingVolumeAnomalyAtMinFactor()


class TestLeadingVolumeAnomalyAtMinMetadata:
    def test_name(self, factor):
        assert factor.name == "AMT_MIN"

    def test_category(self, factor):
        assert factor.category == "高频动量反转"

    def test_repr(self, factor):
        assert "AMT_MIN" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "AMT_MIN"
        assert meta["category"] == "高频动量反转"


class TestLeadingVolumeAnomalyAtMinHandCalculated:
    """手算验证 rolling(T, min_periods=1).mean()"""

    def test_constant_input(self, factor):
        """常数输入时，滚动均值等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        daily_amt_min = pd.DataFrame(2.5, index=dates, columns=stocks)

        result = factor.compute(daily_amt_min=daily_amt_min, T=20)
        np.testing.assert_array_almost_equal(result["A"].values, 2.5)

    def test_rolling_mean_T3(self, factor):
        """T=3, 手动验证滚动均值。

        data = [1, 2, 3, 4, 5]
        min_periods=1:
          row 0: mean(1) = 1.0
          row 1: mean(1, 2) = 1.5
          row 2: mean(1, 2, 3) = 2.0
          row 3: mean(2, 3, 4) = 3.0
          row 4: mean(3, 4, 5) = 4.0
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_amt_min = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0], index=dates, columns=stocks)

        result = factor.compute(daily_amt_min=daily_amt_min, T=3)

        assert result.iloc[0, 0] == pytest.approx(1.0, rel=1e-10)
        assert result.iloc[1, 0] == pytest.approx(1.5, rel=1e-10)
        assert result.iloc[2, 0] == pytest.approx(2.0, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(3.0, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(4.0, rel=1e-10)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        daily_amt_min = pd.DataFrame(
            {"A": [1.5] * 10, "B": [3.0] * 10}, index=dates
        )

        result = factor.compute(daily_amt_min=daily_amt_min, T=5)
        np.testing.assert_array_almost_equal(result["A"].values, 1.5)
        np.testing.assert_array_almost_equal(result["B"].values, 3.0)


class TestLeadingVolumeAnomalyAtMinEdgeCases:
    def test_single_value(self, factor):
        """单个数据点的滚动均值应等于该值 (min_periods=1)。"""
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A"]
        daily_amt_min = pd.DataFrame([3.14], index=dates, columns=stocks)

        result = factor.compute(daily_amt_min=daily_amt_min, T=20)
        assert result.iloc[0, 0] == pytest.approx(3.14, rel=1e-10)

    def test_nan_in_input(self, factor):
        """输入含 NaN 时不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        values = np.ones(10) * 2.0
        values[3] = np.nan
        daily_amt_min = pd.DataFrame(values, index=dates, columns=stocks)

        result = factor.compute(daily_amt_min=daily_amt_min, T=5)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        """全 NaN 输入时结果应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily_amt_min = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_amt_min=daily_amt_min, T=5)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        """全零输入时结果应全为 0。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily_amt_min = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_amt_min=daily_amt_min, T=5)
        for val in result["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)


class TestLeadingVolumeAnomalyAtMinOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily_amt_min = pd.DataFrame(
            np.random.uniform(0.5, 5.0, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_amt_min=daily_amt_min, T=20)
        assert result.shape == daily_amt_min.shape
        assert list(result.columns) == list(daily_amt_min.columns)
        assert list(result.index) == list(daily_amt_min.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_amt_min = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0], index=dates, columns=stocks)

        result = factor.compute(daily_amt_min=daily_amt_min, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_no_leading_nan_min_periods_1(self, factor):
        """min_periods=1，所以第一行就有值。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        daily_amt_min = pd.DataFrame(
            np.random.uniform(0.5, 5.0, (10, 2)), index=dates, columns=stocks
        )

        result = factor.compute(daily_amt_min=daily_amt_min, T=20)
        assert result.iloc[0].notna().all()
        assert result.notna().all().all()
