import numpy as np
import pandas as pd
import pytest

from factors.supply_chain_degree import SupplyChainDegreeFactor


@pytest.fixture
def factor():
    return SupplyChainDegreeFactor()


class TestSupplyChainDegreeMetadata:
    def test_name(self, factor):
        assert factor.name == "SUPPLY_CHAIN_DEGREE"

    def test_category(self, factor):
        assert factor.category == "图谱网络-网络结构"

    def test_repr(self, factor):
        assert "SUPPLY_CHAIN_DEGREE" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "SUPPLY_CHAIN_DEGREE"
        assert meta["category"] == "图谱网络-网络结构"


class TestSupplyChainDegreeHandCalculated:
    """用手算数据验证计算的正确性。"""

    def test_simple_addition(self, factor):
        """d_i = d_in + d_out, 简单加法验证。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_in_degree = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0], index=dates, columns=stocks)
        daily_out_degree = pd.DataFrame([5.0, 4.0, 3.0, 2.0, 1.0], index=dates, columns=stocks)

        result = factor.compute(
            daily_in_degree=daily_in_degree,
            daily_out_degree=daily_out_degree,
        )

        np.testing.assert_array_almost_equal(result["A"].values, [6.0, 6.0, 6.0, 6.0, 6.0])

    def test_zero_degree(self, factor):
        """入度和出度都为 0 时, 节点度为 0。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_in_degree = pd.DataFrame(0.0, index=dates, columns=stocks)
        daily_out_degree = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_in_degree=daily_in_degree,
            daily_out_degree=daily_out_degree,
        )

        np.testing.assert_array_almost_equal(result["A"].values, 0.0)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A", "B"]
        daily_in_degree = pd.DataFrame(
            {"A": [3.0] * 5, "B": [7.0] * 5}, index=dates
        )
        daily_out_degree = pd.DataFrame(
            {"A": [2.0] * 5, "B": [1.0] * 5}, index=dates
        )

        result = factor.compute(
            daily_in_degree=daily_in_degree,
            daily_out_degree=daily_out_degree,
        )

        np.testing.assert_array_almost_equal(result["A"].values, 5.0)
        np.testing.assert_array_almost_equal(result["B"].values, 8.0)

    def test_varying_degrees(self, factor):
        """验证逐行加法。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]
        daily_in_degree = pd.DataFrame([1.0, 3.0, 5.0], index=dates, columns=stocks)
        daily_out_degree = pd.DataFrame([2.0, 4.0, 6.0], index=dates, columns=stocks)

        result = factor.compute(
            daily_in_degree=daily_in_degree,
            daily_out_degree=daily_out_degree,
        )

        np.testing.assert_array_almost_equal(result["A"].values, [3.0, 7.0, 11.0])


class TestSupplyChainDegreeEdgeCases:
    def test_nan_in_input(self, factor):
        """输入含 NaN 时, 对应位置结果为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        in_vals = [1.0, np.nan, 3.0, 4.0, 5.0]
        daily_in_degree = pd.DataFrame(in_vals, index=dates, columns=stocks)
        daily_out_degree = pd.DataFrame(1.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_in_degree=daily_in_degree,
            daily_out_degree=daily_out_degree,
        )
        assert result.shape == (5, 1)
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[0, 0] == pytest.approx(2.0)

    def test_all_nan(self, factor):
        """全 NaN 输入时, 结果应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_in_degree = pd.DataFrame(np.nan, index=dates, columns=stocks)
        daily_out_degree = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(
            daily_in_degree=daily_in_degree,
            daily_out_degree=daily_out_degree,
        )
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        """全零输入时, 结果应全为 0。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_in_degree = pd.DataFrame(0.0, index=dates, columns=stocks)
        daily_out_degree = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_in_degree=daily_in_degree,
            daily_out_degree=daily_out_degree,
        )
        for val in result["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)


class TestSupplyChainDegreeOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily_in_degree = pd.DataFrame(
            np.random.randint(0, 10, (30, 3)).astype(float), index=dates, columns=stocks
        )
        daily_out_degree = pd.DataFrame(
            np.random.randint(0, 10, (30, 3)).astype(float), index=dates, columns=stocks
        )

        result = factor.compute(
            daily_in_degree=daily_in_degree,
            daily_out_degree=daily_out_degree,
        )

        assert result.shape == daily_in_degree.shape
        assert list(result.columns) == list(daily_in_degree.columns)
        assert list(result.index) == list(daily_in_degree.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_in_degree = pd.DataFrame([1.0] * 5, index=dates, columns=stocks)
        daily_out_degree = pd.DataFrame([1.0] * 5, index=dates, columns=stocks)

        result = factor.compute(
            daily_in_degree=daily_in_degree,
            daily_out_degree=daily_out_degree,
        )
        assert isinstance(result, pd.DataFrame)

    def test_no_nan_with_valid_input(self, factor):
        """有效输入时不应产生 NaN。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        daily_in_degree = pd.DataFrame(
            np.random.randint(0, 10, (10, 2)).astype(float), index=dates, columns=stocks
        )
        daily_out_degree = pd.DataFrame(
            np.random.randint(0, 10, (10, 2)).astype(float), index=dates, columns=stocks
        )

        result = factor.compute(
            daily_in_degree=daily_in_degree,
            daily_out_degree=daily_out_degree,
        )
        assert result.notna().all().all()
