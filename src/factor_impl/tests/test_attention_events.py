import numpy as np
import pandas as pd
import pytest

from factors.attention_events import AttentionEventsFactor


@pytest.fixture
def factor():
    return AttentionEventsFactor()


class TestAttentionEventsMetadata:
    def test_name(self, factor):
        assert factor.name == "ATTENTION_EVENTS"

    def test_category(self, factor):
        assert factor.category == "行为金融"

    def test_repr(self, factor):
        assert "ATTENTION_EVENTS" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "ATTENTION_EVENTS"
        assert meta["category"] == "行为金融"


class TestAttentionEventsHandCalculated:
    """用手算数据验证 limit_up + limit_down 的正确性。"""

    def test_basic_addition(self, factor):
        """已知数据验证: 涨停 + 跌停 = 总关注事件。

        limit_up   = [[2, 1], [0, 3], [1, 0]]
        limit_down = [[1, 0], [1, 1], [0, 2]]
        expected   = [[3, 1], [1, 4], [1, 2]]
        """
        dates = pd.date_range("2024-01-01", periods=3, freq="ME")
        stocks = ["A", "B"]
        limit_up = pd.DataFrame(
            [[2, 1], [0, 3], [1, 0]], index=dates, columns=stocks
        )
        limit_down = pd.DataFrame(
            [[1, 0], [1, 1], [0, 2]], index=dates, columns=stocks
        )

        result = factor.compute(limit_up_count=limit_up, limit_down_count=limit_down)

        expected = pd.DataFrame(
            [[3, 1], [1, 4], [1, 2]], index=dates, columns=stocks
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_single_stock(self, factor):
        """单只股票计算。"""
        dates = pd.date_range("2024-01-01", periods=4, freq="ME")
        limit_up = pd.DataFrame({"X": [3, 0, 1, 2]}, index=dates)
        limit_down = pd.DataFrame({"X": [0, 2, 1, 0]}, index=dates)

        result = factor.compute(limit_up_count=limit_up, limit_down_count=limit_down)
        np.testing.assert_array_equal(result["X"].values, [3, 2, 2, 2])

    def test_zero_counts(self, factor):
        """全零输入应返回全零。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="ME")
        stocks = ["A"]
        zeros = pd.DataFrame(0, index=dates, columns=stocks)

        result = factor.compute(limit_up_count=zeros, limit_down_count=zeros)
        assert (result.values == 0).all()


class TestAttentionEventsEdgeCases:
    def test_nan_fill_value(self, factor):
        """一侧含 NaN 时, fill_value=0 应正确处理。

        limit_up   = [1, NaN, 2]
        limit_down = [1,  1, NaN]
        result     = [2,  1,  2]  (NaN 被视为 0)
        """
        dates = pd.date_range("2024-01-01", periods=3, freq="ME")
        limit_up = pd.DataFrame({"A": [1, np.nan, 2]}, index=dates)
        limit_down = pd.DataFrame({"A": [1, 1, np.nan]}, index=dates)

        result = factor.compute(limit_up_count=limit_up, limit_down_count=limit_down)
        assert result.iloc[0, 0] == 2.0
        assert result.iloc[1, 0] == 1.0
        assert result.iloc[2, 0] == 2.0

    def test_both_nan(self, factor):
        """两侧同一位置都为 NaN 时, add(fill_value=0) 仍为 NaN（两侧均缺失）。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="ME")
        limit_up = pd.DataFrame({"A": [1, np.nan, 2]}, index=dates)
        limit_down = pd.DataFrame({"A": [1, np.nan, 3]}, index=dates)

        result = factor.compute(limit_up_count=limit_up, limit_down_count=limit_down)
        assert pd.isna(result.iloc[1, 0])

    def test_large_values(self, factor):
        """大数值不应溢出。"""
        dates = pd.date_range("2024-01-01", periods=2, freq="ME")
        limit_up = pd.DataFrame({"A": [1000000, 2000000]}, index=dates)
        limit_down = pd.DataFrame({"A": [3000000, 4000000]}, index=dates)

        result = factor.compute(limit_up_count=limit_up, limit_down_count=limit_down)
        assert result.iloc[0, 0] == 4000000
        assert result.iloc[1, 0] == 6000000


class TestAttentionEventsOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=12, freq="ME")
        stocks = ["A", "B", "C"]
        limit_up = pd.DataFrame(
            np.ones((12, 3), dtype=int), index=dates, columns=stocks
        )
        limit_down = pd.DataFrame(
            np.ones((12, 3), dtype=int), index=dates, columns=stocks
        )

        result = factor.compute(limit_up_count=limit_up, limit_down_count=limit_down)

        assert result.shape == (12, 3)
        assert list(result.columns) == stocks
        assert list(result.index) == list(dates)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=3, freq="ME")
        stocks = ["A"]
        limit_up = pd.DataFrame([1, 2, 3], index=dates, columns=stocks)
        limit_down = pd.DataFrame([4, 5, 6], index=dates, columns=stocks)

        result = factor.compute(limit_up_count=limit_up, limit_down_count=limit_down)
        assert isinstance(result, pd.DataFrame)

    def test_no_rolling_no_leading_nan(self, factor):
        """无滚动窗口, 所有行都应有值。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="ME")
        stocks = ["A", "B"]
        limit_up = pd.DataFrame(
            np.random.randint(0, 5, (10, 2)), index=dates, columns=stocks
        )
        limit_down = pd.DataFrame(
            np.random.randint(0, 5, (10, 2)), index=dates, columns=stocks
        )

        result = factor.compute(limit_up_count=limit_up, limit_down_count=limit_down)
        assert result.notna().all().all()
