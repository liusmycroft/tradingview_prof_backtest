import numpy as np
import pandas as pd
import pytest

from factors.morning_mist import MorningMistFactor


@pytest.fixture
def factor():
    return MorningMistFactor()


class TestMorningMistMetadata:
    def test_name(self, factor):
        assert factor.name == "MORNING_MIST"

    def test_category(self, factor):
        assert factor.category == "高频量价"

    def test_repr(self, factor):
        assert "MORNING_MIST" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "MORNING_MIST"
        assert meta["category"] == "高频量价"
        assert "t值" in meta["description"] or "标准差" in meta["description"]


class TestMorningMistCompute:
    """测试 compute 方法。"""

    def test_basic_rolling_mean(self, factor):
        """验证滚动均值计算的正确性。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        data = pd.DataFrame({"A": [0.5, 1.0, 1.5, 2.0, 2.5]}, index=dates)

        result = factor.compute(daily_morning_mist=data, T=3)

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(1.0)  # mean(0.5, 1.0, 1.5)
        assert result.iloc[3, 0] == pytest.approx(1.5)  # mean(1.0, 1.5, 2.0)
        assert result.iloc[4, 0] == pytest.approx(2.0)  # mean(1.5, 2.0, 2.5)

    def test_T20_known_values(self, factor):
        """T=20 时验证已知值。"""
        dates = pd.bdate_range("2025-01-01", periods=25)
        np.random.seed(123)
        vals = np.random.uniform(0.5, 2.0, 25)
        data = pd.DataFrame({"A": vals}, index=dates)

        result = factor.compute(daily_morning_mist=data, T=20)

        assert result.iloc[:19].isna().all().all()
        expected = vals[:20].mean()
        assert result.iloc[19, 0] == pytest.approx(expected)

    def test_multi_stock(self, factor):
        """多只股票同时计算。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        data = pd.DataFrame(
            {"A": [1.0, 2.0, 3.0, 4.0, 5.0], "B": [10.0, 20.0, 30.0, 40.0, 50.0]},
            index=dates,
        )

        result = factor.compute(daily_morning_mist=data, T=3)

        assert result.iloc[2, 0] == pytest.approx(2.0)
        assert result.iloc[2, 1] == pytest.approx(20.0)


class TestMorningMistEdgeCases:
    def test_nan_in_input(self, factor):
        """输入含 NaN 时不应抛异常。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        data = pd.DataFrame({"A": [1.0, np.nan, 3.0, 4.0, 5.0]}, index=dates)

        result = factor.compute(daily_morning_mist=data, T=3)

        assert isinstance(result, pd.DataFrame)
        assert np.isnan(result.iloc[2, 0])

    def test_constant_values(self, factor):
        """常数输入时均值等于该常数。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        data = pd.DataFrame({"A": [1.5] * 5}, index=dates)

        result = factor.compute(daily_morning_mist=data, T=3)

        assert result.iloc[2, 0] == pytest.approx(1.5)


class TestMorningMistOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=30)
        stocks = ["A", "B", "C"]
        data = pd.DataFrame(
            np.random.uniform(0.5, 2.0, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_morning_mist=data, T=20)

        assert result.shape == data.shape
        assert list(result.columns) == list(data.columns)

    def test_output_is_dataframe(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=5)
        data = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=dates)

        result = factor.compute(daily_morning_mist=data, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        """前 T-1 行应全为 NaN。"""
        dates = pd.bdate_range("2025-01-01", periods=25)
        data = pd.DataFrame(
            np.random.uniform(0.5, 2.0, (25, 2)), index=dates, columns=["A", "B"]
        )
        T = 20

        result = factor.compute(daily_morning_mist=data, T=T)

        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
