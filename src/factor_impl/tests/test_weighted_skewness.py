import numpy as np
import pandas as pd
import pytest

from factors.weighted_skewness import WeightedSkewnessFactor


@pytest.fixture
def factor():
    return WeightedSkewnessFactor()


class TestWeightedSkewnessMetadata:
    def test_name(self, factor):
        assert factor.name == "WEIGHTED_SKEWNESS"

    def test_category(self, factor):
        assert factor.category == "高频收益分布"

    def test_repr(self, factor):
        assert "WEIGHTED_SKEWNESS" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "WEIGHTED_SKEWNESS"
        assert meta["category"] == "高频收益分布"
        assert "偏度" in meta["description"]


class TestWeightedSkewnessCompute:
    """测试 compute 方法。"""

    def test_basic_rolling_mean(self, factor):
        """验证滚动均值计算的正确性。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        data = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=dates)

        result = factor.compute(daily_weighted_skew=data, T=3)

        # 前 2 行应为 NaN (T-1=2)
        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        # 第 3 行: mean(1, 2, 3) = 2.0
        assert result.iloc[2, 0] == pytest.approx(2.0)
        # 第 4 行: mean(2, 3, 4) = 3.0
        assert result.iloc[3, 0] == pytest.approx(3.0)
        # 第 5 行: mean(3, 4, 5) = 4.0
        assert result.iloc[4, 0] == pytest.approx(4.0)

    def test_T20_known_values(self, factor):
        """T=20 时验证已知值。"""
        dates = pd.bdate_range("2025-01-01", periods=25)
        np.random.seed(42)
        vals = np.random.randn(25)
        data = pd.DataFrame({"A": vals}, index=dates)

        result = factor.compute(daily_weighted_skew=data, T=20)

        # 前 19 行应为 NaN
        assert result.iloc[:19].isna().all().all()
        # 第 20 行: mean of first 20 values
        expected = vals[:20].mean()
        assert result.iloc[19, 0] == pytest.approx(expected)

    def test_multi_stock(self, factor):
        """多只股票同时计算。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        data = pd.DataFrame(
            {"A": [1.0, 2.0, 3.0, 4.0, 5.0], "B": [5.0, 4.0, 3.0, 2.0, 1.0]},
            index=dates,
        )

        result = factor.compute(daily_weighted_skew=data, T=3)

        assert result.iloc[2, 0] == pytest.approx(2.0)  # A: mean(1,2,3)
        assert result.iloc[2, 1] == pytest.approx(4.0)  # B: mean(5,4,3)

    def test_negative_values(self, factor):
        """负偏度值的处理。"""
        dates = pd.bdate_range("2025-01-01", periods=4)
        data = pd.DataFrame({"A": [-1.0, -2.0, -3.0, -4.0]}, index=dates)

        result = factor.compute(daily_weighted_skew=data, T=3)

        assert result.iloc[2, 0] == pytest.approx(-2.0)
        assert result.iloc[3, 0] == pytest.approx(-3.0)


class TestWeightedSkewnessEdgeCases:
    def test_nan_in_input(self, factor):
        """输入含 NaN 时不应抛异常。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        data = pd.DataFrame({"A": [1.0, np.nan, 3.0, 4.0, 5.0]}, index=dates)

        result = factor.compute(daily_weighted_skew=data, T=3)

        assert isinstance(result, pd.DataFrame)
        # 含 NaN 的窗口结果也应为 NaN
        assert np.isnan(result.iloc[2, 0])

    def test_all_same_values(self, factor):
        """所有值相同时，均值等于该值。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        data = pd.DataFrame({"A": [2.5] * 5}, index=dates)

        result = factor.compute(daily_weighted_skew=data, T=3)

        assert result.iloc[2, 0] == pytest.approx(2.5)
        assert result.iloc[4, 0] == pytest.approx(2.5)


class TestWeightedSkewnessOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=30)
        stocks = ["A", "B", "C"]
        data = pd.DataFrame(
            np.random.randn(30, 3), index=dates, columns=stocks
        )

        result = factor.compute(daily_weighted_skew=data, T=20)

        assert result.shape == data.shape
        assert list(result.columns) == list(data.columns)
        assert list(result.index) == list(data.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=5)
        data = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=dates)

        result = factor.compute(daily_weighted_skew=data, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        """前 T-1 行应全为 NaN。"""
        dates = pd.bdate_range("2025-01-01", periods=25)
        data = pd.DataFrame(
            np.random.randn(25, 2), index=dates, columns=["A", "B"]
        )
        T = 20

        result = factor.compute(daily_weighted_skew=data, T=T)

        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
