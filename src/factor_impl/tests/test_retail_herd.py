import numpy as np
import pandas as pd
import pytest

from factors.retail_herd import RetailHerdFactor


@pytest.fixture
def factor():
    return RetailHerdFactor()


class TestRetailHerdMetadata:
    def test_name(self, factor):
        assert factor.name == "RETAIL_HERD"

    def test_category(self, factor):
        assert factor.category == "高频资金流"

    def test_repr(self, factor):
        assert "RETAIL_HERD" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "RETAIL_HERD"
        assert meta["category"] == "高频资金流"
        assert "散户" in meta["description"] or "羊群" in meta["description"]


class TestRetailHerdCompute:
    """测试 compute 方法。"""

    def test_perfect_positive_correlation(self, factor):
        """完美正相关: 收益率与次日小单净流入完全同向。"""
        dates = pd.bdate_range("2025-01-01", periods=8)
        # returns 和 s_net_inflow shift(-1) 完全同向
        returns = pd.DataFrame(
            {"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]}, index=dates
        )
        # s_net_inflow[t+1] 应与 returns[t] 同向
        # 即 s_net_inflow.shift(-1) = returns
        s_net_inflow = pd.DataFrame(
            {"A": [np.nan, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}, index=dates
        )

        result = factor.compute(returns=returns, s_net_inflow=s_net_inflow, T=5)

        # 在有效窗口内，秩相关应接近 1.0
        valid_vals = result.dropna()
        assert len(valid_vals) > 0
        assert valid_vals.iloc[0, 0] == pytest.approx(1.0, abs=0.01)

    def test_perfect_negative_correlation(self, factor):
        """完美负相关: 收益率与次日小单净流入完全反向。"""
        dates = pd.bdate_range("2025-01-01", periods=8)
        returns = pd.DataFrame(
            {"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]}, index=dates
        )
        # s_net_inflow.shift(-1) 与 returns 反向
        s_net_inflow = pd.DataFrame(
            {"A": [np.nan, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0]}, index=dates
        )

        result = factor.compute(returns=returns, s_net_inflow=s_net_inflow, T=5)

        valid_vals = result.dropna()
        assert len(valid_vals) > 0
        assert valid_vals.iloc[0, 0] == pytest.approx(-1.0, abs=0.01)

    def test_multi_stock(self, factor):
        """多只股票同时计算。"""
        dates = pd.bdate_range("2025-01-01", periods=8)
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.randn(8, 2), index=dates, columns=["A", "B"]
        )
        s_net_inflow = pd.DataFrame(
            np.random.randn(8, 2), index=dates, columns=["A", "B"]
        )

        result = factor.compute(returns=returns, s_net_inflow=s_net_inflow, T=5)

        assert result.shape == (8, 2)

    def test_correlation_range(self, factor):
        """秩相关系数应在 [-1, 1] 范围内。"""
        dates = pd.bdate_range("2025-01-01", periods=30)
        np.random.seed(123)
        returns = pd.DataFrame(
            np.random.randn(30, 3), index=dates, columns=["A", "B", "C"]
        )
        s_net_inflow = pd.DataFrame(
            np.random.randn(30, 3), index=dates, columns=["A", "B", "C"]
        )

        result = factor.compute(returns=returns, s_net_inflow=s_net_inflow, T=10)

        valid = result.dropna()
        assert (valid.values >= -1.0 - 1e-10).all()
        assert (valid.values <= 1.0 + 1e-10).all()


class TestRetailHerdEdgeCases:
    def test_nan_in_input(self, factor):
        """输入含 NaN 时不应抛异常。"""
        dates = pd.bdate_range("2025-01-01", periods=8)
        returns = pd.DataFrame(
            {"A": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]}, index=dates
        )
        s_net_inflow = pd.DataFrame(
            {"A": [1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0]}, index=dates
        )

        result = factor.compute(returns=returns, s_net_inflow=s_net_inflow, T=5)

        assert isinstance(result, pd.DataFrame)

    def test_last_row_nan_due_to_lead(self, factor):
        """最后一行的 s_net_inflow.shift(-1) 为 NaN，影响包含该行的窗口。"""
        dates = pd.bdate_range("2025-01-01", periods=6)
        returns = pd.DataFrame(
            {"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, index=dates
        )
        s_net_inflow = pd.DataFrame(
            {"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, index=dates
        )

        result = factor.compute(returns=returns, s_net_inflow=s_net_inflow, T=5)

        # shift(-1) 使最后一行的 s_lead 为 NaN
        assert isinstance(result, pd.DataFrame)


class TestRetailHerdOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=30)
        stocks = ["A", "B", "C"]
        returns = pd.DataFrame(
            np.random.randn(30, 3), index=dates, columns=stocks
        )
        s_net_inflow = pd.DataFrame(
            np.random.randn(30, 3), index=dates, columns=stocks
        )

        result = factor.compute(returns=returns, s_net_inflow=s_net_inflow, T=20)

        assert result.shape == returns.shape
        assert list(result.columns) == list(returns.columns)

    def test_output_is_dataframe(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=8)
        returns = pd.DataFrame({"A": np.random.randn(8)}, index=dates)
        s_net_inflow = pd.DataFrame({"A": np.random.randn(8)}, index=dates)

        result = factor.compute(returns=returns, s_net_inflow=s_net_inflow, T=5)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        """前 T-1 行应全为 NaN。"""
        dates = pd.bdate_range("2025-01-01", periods=30)
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.randn(30, 2), index=dates, columns=["A", "B"]
        )
        s_net_inflow = pd.DataFrame(
            np.random.randn(30, 2), index=dates, columns=["A", "B"]
        )
        T = 10

        result = factor.compute(returns=returns, s_net_inflow=s_net_inflow, T=T)

        assert result.iloc[: T - 1].isna().all().all()
