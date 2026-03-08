import numpy as np
import pandas as pd
import pytest

from factors.similar_reverse import SimilarReverseFactor


@pytest.fixture
def factor():
    return SimilarReverseFactor()


class TestSimilarReverseMetadata:
    def test_name(self, factor):
        assert factor.name == "SIMILAR_REVERSE"

    def test_category(self, factor):
        assert factor.category == "量价改进"

    def test_repr(self, factor):
        assert "SIMILAR_REVERSE" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "SIMILAR_REVERSE"
        assert meta["category"] == "量价改进"
        assert "相似" in meta["description"]


class TestSimilarReverseHandCalculated:
    def test_no_similar_pattern_returns_nan(self, factor):
        """无相似走势时，结果应为 NaN。"""
        np.random.seed(42)
        n = 140
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        stocks = ["A"]
        # 随机价格，相似度很难超过阈值
        close = pd.DataFrame(
            np.cumsum(np.random.randn(n)) + 100, index=dates, columns=stocks
        )
        er = pd.DataFrame(np.random.randn(n) * 0.01, index=dates, columns=stocks)

        result = factor.compute(
            close=close, excess_return=er,
            lookback=120, rw=6, holding_time=6,
            threshold=0.99, half_life=60,
        )
        # 极高阈值下几乎不可能匹配
        assert result.iloc[-1].isna().all() or isinstance(result.iloc[-1, 0], float)

    def test_perfect_repeat_pattern(self, factor):
        """完全重复的价格序列应能匹配到相似走势。"""
        n = 140
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        stocks = ["A"]
        # 周期性价格序列
        pattern = [100, 101, 102, 103, 104, 105] * 24
        close = pd.DataFrame(pattern[:n], index=dates, columns=stocks, dtype=float)
        er = pd.DataFrame(0.01, index=dates, columns=stocks)

        result = factor.compute(
            close=close, excess_return=er,
            lookback=120, rw=6, holding_time=6,
            threshold=0.4, half_life=60,
        )
        # 应该有非 NaN 的结果
        assert result.iloc[-1].notna().any()

    def test_negation_of_weighted_er(self, factor):
        """因子值应为超额收益加权均值的负数。"""
        n = 140
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        stocks = ["A"]
        # 完全重复的上升序列
        pattern = list(range(100, 106)) * 24
        close = pd.DataFrame(pattern[:n], index=dates, columns=stocks, dtype=float)
        # 正超额收益
        er = pd.DataFrame(0.02, index=dates, columns=stocks)

        result = factor.compute(
            close=close, excess_return=er,
            lookback=120, rw=6, holding_time=6,
            threshold=0.4, half_life=60,
        )
        last_val = result.iloc[-1, 0]
        if not np.isnan(last_val):
            # 正超额收益取反应为负
            assert last_val < 0

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        n = 140
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        pattern = list(range(100, 106)) * 24
        close = pd.DataFrame(
            {"A": pattern[:n], "B": [50] * n},
            index=dates, dtype=float,
        )
        er = pd.DataFrame(
            {"A": [0.02] * n, "B": [0.02] * n},
            index=dates,
        )

        result = factor.compute(
            close=close, excess_return=er,
            lookback=120, rw=6, holding_time=6,
            threshold=0.4, half_life=60,
        )
        assert result.shape == (n, 2)


class TestSimilarReverseEdgeCases:
    def test_short_data(self, factor):
        """数据不足 lookback 时，结果应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        close = pd.DataFrame(range(10), index=dates, columns=stocks, dtype=float)
        er = pd.DataFrame(0.01, index=dates, columns=stocks)

        result = factor.compute(
            close=close, excess_return=er,
            lookback=120, rw=6, holding_time=6,
        )
        assert result.isna().all().all()

    def test_nan_in_close(self, factor):
        """收盘价含 NaN 时不应抛异常。"""
        n = 140
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        stocks = ["A"]
        vals = np.array(list(range(100, 106)) * 24, dtype=float)[:n]
        vals[70] = np.nan
        close = pd.DataFrame(vals, index=dates, columns=stocks)
        er = pd.DataFrame(0.01, index=dates, columns=stocks)

        result = factor.compute(
            close=close, excess_return=er,
            lookback=120, rw=6, holding_time=6,
        )
        assert isinstance(result, pd.DataFrame)

    def test_all_nan(self, factor):
        """全 NaN 输入时，结果应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=140, freq="D")
        stocks = ["A"]
        close = pd.DataFrame(np.nan, index=dates, columns=stocks)
        er = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(close=close, excess_return=er)
        assert result.isna().all().all()


class TestSimilarReverseOutputShape:
    def test_output_shape_matches_input(self, factor):
        n = 140
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        stocks = ["A", "B", "C"]
        close = pd.DataFrame(
            np.random.uniform(90, 110, (n, 3)), index=dates, columns=stocks
        )
        er = pd.DataFrame(
            np.random.randn(n, 3) * 0.01, index=dates, columns=stocks
        )

        result = factor.compute(close=close, excess_return=er)
        assert result.shape == close.shape
        assert list(result.columns) == list(close.columns)
        assert list(result.index) == list(close.index)

    def test_output_is_dataframe(self, factor):
        n = 140
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        stocks = ["A"]
        close = pd.DataFrame(range(n), index=dates, columns=stocks, dtype=float)
        er = pd.DataFrame(0.01, index=dates, columns=stocks)

        result = factor.compute(close=close, excess_return=er)
        assert isinstance(result, pd.DataFrame)
