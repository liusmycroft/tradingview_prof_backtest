import numpy as np
import pandas as pd
import pytest

from factors.k_line_pattern import KLinePatternFactor


@pytest.fixture
def factor():
    return KLinePatternFactor()


class TestKLinePatternMetadata:
    def test_name(self, factor):
        assert factor.name == "K_LINE_PATTERN"

    def test_category(self, factor):
        assert factor.category == "量价因子改进"

    def test_repr(self, factor):
        assert "K_LINE_PATTERN" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "K_LINE_PATTERN"


class TestKLinePatternCompute:
    def test_constant_win_rate(self, factor):
        """常数胜率时，因子 = 胜率 * sum(weights)。"""
        dates = pd.date_range("2024-01-01", periods=40, freq="D")
        stocks = ["A"]
        wr = pd.DataFrame(0.5, index=dates, columns=stocks)

        result = factor.compute(pattern_win_rate=wr, lookback=40, half_life=20.0)
        # sum of weights = sum(0.5^((39-t)/20)) for t=0..39
        distances = np.arange(39, -1, -1, dtype=float)
        weights = np.power(0.5, distances / 20.0)
        expected = 0.5 * np.sum(weights)
        assert result.iloc[-1, 0] == pytest.approx(expected, rel=1e-6)

    def test_leading_nan(self, factor):
        """前 lookback-1 行应为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        stocks = ["A"]
        wr = pd.DataFrame(np.random.rand(50), index=dates, columns=stocks)

        result = factor.compute(pattern_win_rate=wr, lookback=40)
        assert result.iloc[:39]["A"].isna().all()
        assert result.iloc[39:]["A"].notna().all()

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        stocks = ["A", "B"]
        wr = pd.DataFrame(np.random.rand(50, 2), index=dates, columns=stocks)

        result = factor.compute(pattern_win_rate=wr, lookback=40)
        assert result.shape == wr.shape
        assert isinstance(result, pd.DataFrame)

    def test_recent_weight_higher(self, factor):
        """近期高胜率应使因子更大。"""
        dates = pd.date_range("2024-01-01", periods=40, freq="D")
        stocks = ["A", "B"]
        wr_a = [0.3] * 20 + [0.7] * 20  # 近期高
        wr_b = [0.7] * 20 + [0.3] * 20  # 近期低
        wr = pd.DataFrame({"A": wr_a, "B": wr_b}, index=dates)

        result = factor.compute(pattern_win_rate=wr, lookback=40, half_life=20.0)
        assert result.iloc[-1]["A"] > result.iloc[-1]["B"]
