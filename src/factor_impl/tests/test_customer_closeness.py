import numpy as np
import pandas as pd
import pytest

from factors.customer_closeness import CustomerClosenessFactor


@pytest.fixture
def factor():
    return CustomerClosenessFactor()


class TestCustomerClosenessMetadata:
    def test_name(self, factor):
        assert factor.name == "CUSTOMER_CLOSENESS"

    def test_category(self, factor):
        assert factor.category == "图谱网络-网络结构"

    def test_repr(self, factor):
        assert "CUSTOMER_CLOSENESS" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "CUSTOMER_CLOSENESS"
        assert meta["category"] == "图谱网络-网络结构"


class TestCustomerClosenessHandCalculated:
    def test_passthrough(self, factor):
        """因子应直接返回输入值的副本。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A", "B"]
        daily = pd.DataFrame(
            {"A": [0.5, 0.6, 0.7, 0.8, 0.9], "B": [0.3, 0.4, 0.5, 0.6, 0.7]},
            index=dates,
        )

        result = factor.compute(daily_closeness=daily)
        pd.testing.assert_frame_equal(result, daily)

    def test_constant_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(0.42, index=dates, columns=stocks)

        result = factor.compute(daily_closeness=daily)
        np.testing.assert_array_almost_equal(result["A"].values, 0.42)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A", "B"]
        daily = pd.DataFrame(
            {"A": [0.1] * 5, "B": [0.9] * 5}, index=dates
        )

        result = factor.compute(daily_closeness=daily)
        np.testing.assert_array_almost_equal(result["A"].values, 0.1)
        np.testing.assert_array_almost_equal(result["B"].values, 0.9)

    def test_result_is_copy(self, factor):
        """修改结果不应影响输入。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame([1.0, 2.0, 3.0], index=dates, columns=stocks)

        result = factor.compute(daily_closeness=daily)
        result.iloc[0, 0] = 999.0
        assert daily.iloc[0, 0] == 1.0


class TestCustomerClosenessEdgeCases:
    def test_single_value(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame([0.55], index=dates, columns=stocks)

        result = factor.compute(daily_closeness=daily)
        assert result.iloc[0, 0] == pytest.approx(0.55, rel=1e-10)

    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        values = [0.1, 0.2, np.nan, 0.4, 0.5]
        daily = pd.DataFrame(values, index=dates, columns=stocks)

        result = factor.compute(daily_closeness=daily)
        assert isinstance(result, pd.DataFrame)
        assert np.isnan(result.iloc[2, 0])

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_closeness=daily)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_closeness=daily)
        for val in result["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)


class TestCustomerClosenessOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B", "C"]
        daily = pd.DataFrame(
            np.random.uniform(0.1, 1.0, (10, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_closeness=daily)
        assert result.shape == daily.shape
        assert list(result.columns) == list(daily.columns)
        assert list(result.index) == list(daily.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame([0.1, 0.2, 0.3], index=dates, columns=stocks)

        result = factor.compute(daily_closeness=daily)
        assert isinstance(result, pd.DataFrame)
