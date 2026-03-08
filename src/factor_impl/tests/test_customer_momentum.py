import numpy as np
import pandas as pd
import pytest

from factors.customer_momentum import CustomerMomentumFactor


@pytest.fixture
def factor():
    return CustomerMomentumFactor()


class TestCustomerMomentumMetadata:
    def test_name(self, factor):
        assert factor.name == "CMOM"

    def test_category(self, factor):
        assert factor.category == "图谱网络-动量溢出"

    def test_repr(self, factor):
        assert "CMOM" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "CMOM"
        assert meta["category"] == "图谱网络-动量溢出"


class TestCustomerMomentumHandCalculated:
    def test_simple_mapping(self, factor):
        """简单映射：A 的最大客户是 B，B 的最大客户是 C。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        returns = pd.DataFrame(
            {"A": [0.01, 0.02, 0.03],
             "B": [0.04, 0.05, 0.06],
             "C": [0.07, 0.08, 0.09]},
            index=dates,
        )
        largest_customer = pd.Series({"A": "B", "B": "C", "C": "A"})

        result = factor.compute(returns=returns, largest_customer=largest_customer)

        # A 的因子值 = B 的收益
        np.testing.assert_array_almost_equal(result["A"].values, [0.04, 0.05, 0.06])
        # B 的因子值 = C 的收益
        np.testing.assert_array_almost_equal(result["B"].values, [0.07, 0.08, 0.09])
        # C 的因子值 = A 的收益
        np.testing.assert_array_almost_equal(result["C"].values, [0.01, 0.02, 0.03])

    def test_missing_customer(self, factor):
        """最大客户不在 returns 中时，结果为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=2, freq="D")
        returns = pd.DataFrame(
            {"A": [0.01, 0.02], "B": [0.03, 0.04]}, index=dates
        )
        largest_customer = pd.Series({"A": "B", "B": "X"})  # X 不在 returns 中

        result = factor.compute(returns=returns, largest_customer=largest_customer)

        np.testing.assert_array_almost_equal(result["A"].values, [0.03, 0.04])
        assert result["B"].isna().all()

    def test_no_customer_mapping(self, factor):
        """股票没有最大客户映射时，结果为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=2, freq="D")
        returns = pd.DataFrame(
            {"A": [0.01, 0.02], "B": [0.03, 0.04]}, index=dates
        )
        largest_customer = pd.Series({"A": "B"})  # B 没有映射

        result = factor.compute(returns=returns, largest_customer=largest_customer)

        np.testing.assert_array_almost_equal(result["A"].values, [0.03, 0.04])
        assert result["B"].isna().all()

    def test_nan_customer(self, factor):
        """最大客户为 NaN 时，结果为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=2, freq="D")
        returns = pd.DataFrame(
            {"A": [0.01, 0.02], "B": [0.03, 0.04]}, index=dates
        )
        largest_customer = pd.Series({"A": "B", "B": np.nan})

        result = factor.compute(returns=returns, largest_customer=largest_customer)

        np.testing.assert_array_almost_equal(result["A"].values, [0.03, 0.04])
        assert result["B"].isna().all()


class TestCustomerMomentumEdgeCases:
    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B", "C"]
        returns = pd.DataFrame(
            np.random.randn(10, 3) * 0.05, index=dates, columns=stocks
        )
        largest_customer = pd.Series({"A": "B", "B": "C", "C": "A"})

        result = factor.compute(returns=returns, largest_customer=largest_customer)
        assert result.shape == returns.shape
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == stocks

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        returns = pd.DataFrame({"A": [0.01, 0.02, 0.03]}, index=dates)
        largest_customer = pd.Series({"A": "A"})  # 自身

        result = factor.compute(returns=returns, largest_customer=largest_customer)
        assert isinstance(result, pd.DataFrame)
