import numpy as np
import pandas as pd
import pytest

from factors.supply_centrality_change import SupplyCentralityChangeFactor


@pytest.fixture
def factor():
    return SupplyCentralityChangeFactor()


class TestSupplyCentralityChangeMetadata:
    def test_name(self, factor):
        assert factor.name == "SUPPLY_CENTRALITY_CHANGE"

    def test_category(self, factor):
        assert factor.category == "图谱网络-网络结构"

    def test_repr(self, factor):
        assert "SUPPLY_CENTRALITY_CHANGE" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "SUPPLY_CENTRALITY_CHANGE"
        assert meta["category"] == "图谱网络-网络结构"


class TestSupplyCentralityChangeCompute:
    def test_known_values(self, factor):
        """用已知数据验证差值计算。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        stocks = ["A", "B"]
        current = pd.DataFrame(
            [[0.8, 0.6], [0.5, 0.3], [0.9, 0.7]],
            index=dates, columns=stocks,
        )
        previous = pd.DataFrame(
            [[0.5, 0.4], [0.5, 0.5], [0.3, 0.9]],
            index=dates, columns=stocks,
        )

        result = factor.compute(current_centrality=current, previous_centrality=previous)

        expected = pd.DataFrame(
            [[0.3, 0.2], [0.0, -0.2], [0.6, -0.2]],
            index=dates, columns=stocks,
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_equal_inputs(self, factor):
        """当期与上期相同时，变化应为零。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        df = pd.DataFrame({"A": [0.1, 0.2, 0.3, 0.4, 0.5]}, index=dates)

        result = factor.compute(current_centrality=df, previous_centrality=df)
        expected = pd.DataFrame({"A": [0.0] * 5}, index=dates)
        pd.testing.assert_frame_equal(result, expected)

    def test_zero_previous(self, factor):
        """上期全零时，变化等于当期值。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        current = pd.DataFrame({"A": [0.5, 0.6, 0.7]}, index=dates)
        previous = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=dates)

        result = factor.compute(current_centrality=current, previous_centrality=previous)
        pd.testing.assert_frame_equal(result, current)

    def test_negative_change(self, factor):
        """中心性下降时，因子值应为负。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        current = pd.DataFrame({"A": [0.2, 0.1, 0.0]}, index=dates)
        previous = pd.DataFrame({"A": [0.5, 0.5, 0.5]}, index=dates)

        result = factor.compute(current_centrality=current, previous_centrality=previous)
        assert (result["A"] < 0).all()

    def test_nan_propagation(self, factor):
        """输入含 NaN 时，输出对应位置也应为 NaN。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        current = pd.DataFrame({"A": [0.5, np.nan, 0.7]}, index=dates)
        previous = pd.DataFrame({"A": [0.3, 0.4, np.nan]}, index=dates)

        result = factor.compute(current_centrality=current, previous_centrality=previous)
        assert pytest.approx(result.iloc[0, 0]) == 0.2
        assert pd.isna(result.iloc[1, 0])
        assert pd.isna(result.iloc[2, 0])

    def test_output_shape(self, factor):
        """输出形状应与输入一致。"""
        np.random.seed(42)
        dates = pd.bdate_range("2025-01-01", periods=30)
        stocks = ["A", "B", "C"]
        current = pd.DataFrame(np.random.rand(30, 3), index=dates, columns=stocks)
        previous = pd.DataFrame(np.random.rand(30, 3), index=dates, columns=stocks)

        result = factor.compute(current_centrality=current, previous_centrality=previous)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (30, 3)
        assert list(result.columns) == stocks
