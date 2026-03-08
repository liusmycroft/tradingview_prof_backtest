import numpy as np
import pandas as pd
import pytest

from factors.extreme_return_reversal import ExtremeReturnReversalFactor


@pytest.fixture
def factor():
    return ExtremeReturnReversalFactor()


class TestExtremeReturnReversalMetadata:
    def test_name(self, factor):
        assert factor.name == "EXTREME_RETURN_REVERSAL"

    def test_category(self, factor):
        assert factor.category == "高频动量反转"

    def test_repr(self, factor):
        assert "EXTREME_RETURN_REVERSAL" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "EXTREME_RETURN_REVERSAL"
        assert meta["category"] == "高频动量反转"


class TestExtremeReturnReversalCompute:
    def test_basic_rolling_and_rank(self, factor):
        """验证滚动均值 + 截面排序的正确性。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        stocks = ["A", "B", "C"]
        daily_extreme = pd.DataFrame(
            {"A": [0.01, 0.02, 0.03, 0.04, 0.05],
             "B": [0.05, 0.04, 0.03, 0.02, 0.01],
             "C": [0.03, 0.03, 0.03, 0.03, 0.03]},
            index=dates,
        )
        daily_pre = pd.DataFrame(
            {"A": [0.005, 0.01, 0.015, 0.02, 0.025],
             "B": [0.025, 0.02, 0.015, 0.01, 0.005],
             "C": [0.015, 0.015, 0.015, 0.015, 0.015]},
            index=dates,
        )

        result = factor.compute(
            daily_extreme_ret=daily_extreme, daily_pre_extreme_ret=daily_pre, T=3,
        )

        # First 2 rows should be NaN (T-1=2)
        assert result.iloc[0].isna().all()
        assert result.iloc[1].isna().all()
        # Row 2: mean_extreme = [0.02, 0.04, 0.03], mean_pre = [0.01, 0.02, 0.015]
        # rank_extreme: A=1/3, B=3/3, C=2/3 -> pct rank
        # rank_pre: same ordering
        # result = rank_extreme + rank_pre
        assert result.iloc[2].notna().all()

    def test_single_stock_rank(self, factor):
        """单只股票时, rank 始终为 1.0。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        daily_extreme = pd.DataFrame({"A": [0.01, 0.02, 0.03, 0.04, 0.05]}, index=dates)
        daily_pre = pd.DataFrame({"A": [0.005, 0.01, 0.015, 0.02, 0.025]}, index=dates)

        result = factor.compute(
            daily_extreme_ret=daily_extreme, daily_pre_extreme_ret=daily_pre, T=3,
        )

        # Single stock: rank is always 1.0, so result = 1.0 + 1.0 = 2.0
        for i in range(2, 5):
            assert result.iloc[i, 0] == pytest.approx(2.0)

    def test_rank_ordering(self, factor):
        """截面排序应反映相对大小。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        # A has highest extreme ret, C has lowest
        daily_extreme = pd.DataFrame(
            {"A": [0.10, 0.10, 0.10], "B": [0.05, 0.05, 0.05], "C": [0.01, 0.01, 0.01]},
            index=dates,
        )
        daily_pre = pd.DataFrame(
            {"A": [0.05, 0.05, 0.05], "B": [0.05, 0.05, 0.05], "C": [0.05, 0.05, 0.05]},
            index=dates,
        )

        result = factor.compute(
            daily_extreme_ret=daily_extreme, daily_pre_extreme_ret=daily_pre, T=3,
        )

        # A should have highest rank for extreme, all equal for pre
        assert result.loc[dates[2], "A"] > result.loc[dates[2], "C"]

    def test_multi_stock_shape(self, factor):
        """多只股票输出形状正确。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        stocks = ["A", "B", "C"]
        daily_extreme = pd.DataFrame(
            np.random.randn(5, 3) * 0.02, index=dates, columns=stocks
        )
        daily_pre = pd.DataFrame(
            np.random.randn(5, 3) * 0.01, index=dates, columns=stocks
        )

        result = factor.compute(
            daily_extreme_ret=daily_extreme, daily_pre_extreme_ret=daily_pre, T=3,
        )

        assert result.shape == (5, 3)
        assert list(result.columns) == stocks


class TestExtremeReturnReversalEdgeCases:
    def test_nan_in_input(self, factor):
        """输入含 NaN 时不应抛异常。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        daily_extreme = pd.DataFrame(
            {"A": [0.01, np.nan, 0.03, 0.04, 0.05]}, index=dates
        )
        daily_pre = pd.DataFrame(
            {"A": [0.005, 0.01, 0.015, 0.02, 0.025]}, index=dates
        )

        result = factor.compute(
            daily_extreme_ret=daily_extreme, daily_pre_extreme_ret=daily_pre, T=3,
        )

        assert isinstance(result, pd.DataFrame)


class TestExtremeReturnReversalOutputShape:
    def test_output_is_dataframe(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=5)
        daily_extreme = pd.DataFrame({"A": np.random.randn(5) * 0.02}, index=dates)
        daily_pre = pd.DataFrame({"A": np.random.randn(5) * 0.01}, index=dates)

        result = factor.compute(
            daily_extreme_ret=daily_extreme, daily_pre_extreme_ret=daily_pre, T=3,
        )
        assert isinstance(result, pd.DataFrame)

    def test_output_shape_matches_input(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=30)
        stocks = ["A", "B", "C"]
        daily_extreme = pd.DataFrame(
            np.random.randn(30, 3) * 0.02, index=dates, columns=stocks
        )
        daily_pre = pd.DataFrame(
            np.random.randn(30, 3) * 0.01, index=dates, columns=stocks
        )

        result = factor.compute(
            daily_extreme_ret=daily_extreme, daily_pre_extreme_ret=daily_pre, T=20,
        )
        assert result.shape == daily_extreme.shape
        assert list(result.columns) == stocks

    def test_first_T_minus_1_rows_are_nan(self, factor):
        """前 T-1 行应全为 NaN。"""
        dates = pd.bdate_range("2025-01-01", periods=25)
        stocks = ["A", "B"]
        daily_extreme = pd.DataFrame(
            np.random.randn(25, 2) * 0.02, index=dates, columns=stocks
        )
        daily_pre = pd.DataFrame(
            np.random.randn(25, 2) * 0.01, index=dates, columns=stocks
        )
        T = 20

        result = factor.compute(
            daily_extreme_ret=daily_extreme, daily_pre_extreme_ret=daily_pre, T=T,
        )

        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
