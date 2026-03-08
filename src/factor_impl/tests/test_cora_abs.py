import numpy as np
import pandas as pd
import pytest

from factors.cora_abs import CoraAbsFactor


@pytest.fixture
def factor():
    return CoraAbsFactor()


class TestCoraAbsMetadata:
    def test_name(self, factor):
        assert factor.name == "CORA_A"

    def test_category(self, factor):
        assert factor.category == "高频量价相关性"

    def test_repr(self, factor):
        assert "CORA_A" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "CORA_A"
        assert meta["category"] == "高频量价相关性"


class TestCoraAbsHandCalculated:
    """手算验证 rolling mean。"""

    def test_constant_input(self, factor):
        """常数 0.3, T=3 => rolling mean = 0.3。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(0.3, index=dates, columns=stocks)

        result = factor.compute(daily_cora_a=daily, T=3)

        for val in result["A"].values:
            assert val == pytest.approx(0.3, rel=1e-10)

    def test_varying_input_T3(self, factor):
        """T=3, data=[0.1, 0.2, 0.3, 0.4, 0.5]

        rolling(3, min_periods=1):
          day0: 0.1
          day1: (0.1+0.2)/2 = 0.15
          day2: (0.1+0.2+0.3)/3 = 0.2
          day3: (0.2+0.3+0.4)/3 = 0.3
          day4: (0.3+0.4+0.5)/3 = 0.4
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(
            [0.1, 0.2, 0.3, 0.4, 0.5], index=dates, columns=stocks
        )

        result = factor.compute(daily_cora_a=daily, T=3)

        assert result.iloc[0, 0] == pytest.approx(0.1, rel=1e-10)
        assert result.iloc[1, 0] == pytest.approx(0.15, rel=1e-10)
        assert result.iloc[2, 0] == pytest.approx(0.2, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(0.3, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(0.4, rel=1e-10)

    def test_negative_corr(self, factor):
        """负相关性 -0.5 => 均值 = -0.5。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(-0.5, index=dates, columns=stocks)

        result = factor.compute(daily_cora_a=daily, T=3)
        for val in result["A"].values:
            assert val == pytest.approx(-0.5, rel=1e-10)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A", "B"]
        daily = pd.DataFrame(
            {"A": [0.2] * 5, "B": [-0.3] * 5}, index=dates
        )

        result = factor.compute(daily_cora_a=daily, T=3)

        np.testing.assert_array_almost_equal(result["A"].values, 0.2)
        np.testing.assert_array_almost_equal(result["B"].values, -0.3)

    def test_mixed_values_T4(self, factor):
        """T=4, data=[0.2, 0.8, -0.2, -0.8] => mean=0.0"""
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame([0.2, 0.8, -0.2, -0.8], index=dates, columns=stocks)

        result = factor.compute(daily_cora_a=daily, T=4)

        # day3: mean(0.2, 0.8, -0.2, -0.8) = 0.0
        assert result.iloc[3, 0] == pytest.approx(0.0, abs=1e-10)


class TestCoraAbsEdgeCases:
    def test_single_value(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame([0.42], index=dates, columns=stocks)

        result = factor.compute(daily_cora_a=daily, T=20)
        assert result.iloc[0, 0] == pytest.approx(0.42, rel=1e-10)

    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        values = [0.3, np.nan, 0.4, 0.5, 0.2]
        daily = pd.DataFrame(values, index=dates, columns=stocks)

        result = factor.compute(daily_cora_a=daily, T=3)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (5, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_cora_a=daily, T=3)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_cora_a=daily, T=3)
        for val in result["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)


class TestCoraAbsOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily = pd.DataFrame(
            np.random.uniform(-1, 1, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_cora_a=daily, T=10)

        assert result.shape == daily.shape
        assert list(result.columns) == list(daily.columns)
        assert list(result.index) == list(daily.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame([0.3] * 5, index=dates, columns=stocks)

        result = factor.compute(daily_cora_a=daily, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_no_leading_nan_min_periods_1(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        daily = pd.DataFrame(
            np.random.uniform(-1, 1, (10, 2)), index=dates, columns=stocks
        )

        result = factor.compute(daily_cora_a=daily, T=20)
        assert result.iloc[0].notna().all()
        assert result.notna().all().all()
