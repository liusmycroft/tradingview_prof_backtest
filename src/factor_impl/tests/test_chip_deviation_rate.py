import numpy as np
import pandas as pd
import pytest

from factors.chip_deviation_rate import ChipDeviationRateFactor


@pytest.fixture
def factor():
    return ChipDeviationRateFactor()


class TestChipDeviationRateMetadata:
    def test_name(self, factor):
        assert factor.name == "CHIP_DEVIATION_RATE"

    def test_category(self, factor):
        assert factor.category == "行为金融-筹码分布"

    def test_repr(self, factor):
        assert "CHIP_DEVIATION_RATE" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "CHIP_DEVIATION_RATE"
        assert meta["category"] == "行为金融-筹码分布"


class TestChipDeviationRateHandCalculated:
    """手算验证 BIAS_t = winner_t * turnover_t + BIAS_{t-1} * (1 - turnover_t)"""

    def test_simple_3days(self, factor):
        """3 天手动计算。

        winner   = [0.6, 0.7, 0.8]
        turnover = [0.1, 0.2, 0.3]

        BIAS_0 = 0.6  (初始值)
        BIAS_1 = 0.7 * 0.2 + 0.6 * (1 - 0.2) = 0.14 + 0.48 = 0.62
        BIAS_2 = 0.8 * 0.3 + 0.62 * (1 - 0.3) = 0.24 + 0.434 = 0.674
        """
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        winner = pd.DataFrame([0.6, 0.7, 0.8], index=dates, columns=stocks)
        turnover = pd.DataFrame([0.1, 0.2, 0.3], index=dates, columns=stocks)

        result = factor.compute(winner=winner, turnover=turnover)

        assert result.iloc[0, 0] == pytest.approx(0.6, rel=1e-10)
        assert result.iloc[1, 0] == pytest.approx(0.62, rel=1e-10)
        assert result.iloc[2, 0] == pytest.approx(0.674, rel=1e-10)

    def test_full_turnover(self, factor):
        """换手率为 1 时，BIAS 应等于当日 winner。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        winner = pd.DataFrame([0.3, 0.5, 0.7, 0.4, 0.6], index=dates, columns=stocks)
        turnover = pd.DataFrame([1.0] * 5, index=dates, columns=stocks)

        result = factor.compute(winner=winner, turnover=turnover)

        for i in range(5):
            assert result.iloc[i, 0] == pytest.approx(winner.iloc[i, 0], rel=1e-10)

    def test_zero_turnover(self, factor):
        """换手率为 0 时，BIAS 应保持初始值不变。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        winner = pd.DataFrame([0.5, 0.6, 0.7, 0.8, 0.9], index=dates, columns=stocks)
        turnover = pd.DataFrame([0.0] * 5, index=dates, columns=stocks)

        result = factor.compute(winner=winner, turnover=turnover)

        for i in range(5):
            assert result.iloc[i, 0] == pytest.approx(0.5, rel=1e-10)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A", "B"]

        winner = pd.DataFrame({"A": [0.5, 0.6, 0.7], "B": [0.8, 0.9, 1.0]}, index=dates)
        turnover = pd.DataFrame({"A": [0.1, 0.1, 0.1], "B": [0.5, 0.5, 0.5]}, index=dates)

        result = factor.compute(winner=winner, turnover=turnover)

        # Stock A: BIAS_0=0.5, BIAS_1=0.6*0.1+0.5*0.9=0.51, BIAS_2=0.7*0.1+0.51*0.9=0.529
        assert result.iloc[0, 0] == pytest.approx(0.5, rel=1e-10)
        assert result.iloc[1, 0] == pytest.approx(0.51, rel=1e-10)
        assert result.iloc[2, 0] == pytest.approx(0.529, rel=1e-10)

        # Stock B: BIAS_0=0.8, BIAS_1=0.9*0.5+0.8*0.5=0.85, BIAS_2=1.0*0.5+0.85*0.5=0.925
        assert result.iloc[0, 1] == pytest.approx(0.8, rel=1e-10)
        assert result.iloc[1, 1] == pytest.approx(0.85, rel=1e-10)
        assert result.iloc[2, 1] == pytest.approx(0.925, rel=1e-10)

    def test_constant_winner_converges(self, factor):
        """winner 恒定时，BIAS 应逐渐收敛到 winner 值。"""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        stocks = ["A"]

        winner = pd.DataFrame(0.7, index=dates, columns=stocks)
        turnover = pd.DataFrame(0.05, index=dates, columns=stocks)

        result = factor.compute(winner=winner, turnover=turnover)
        assert result.iloc[-1, 0] == pytest.approx(0.7, rel=1e-4)


class TestChipDeviationRateEdgeCases:
    def test_nan_in_winner(self, factor):
        """winner 含 NaN 时不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        winner = pd.DataFrame([0.5, np.nan, 0.7, 0.8, 0.9], index=dates, columns=stocks)
        turnover = pd.DataFrame([0.1] * 5, index=dates, columns=stocks)

        result = factor.compute(winner=winner, turnover=turnover)
        assert isinstance(result, pd.DataFrame)

    def test_single_day(self, factor):
        """单日数据应返回 winner 值。"""
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A"]
        winner = pd.DataFrame([0.65], index=dates, columns=stocks)
        turnover = pd.DataFrame([0.1], index=dates, columns=stocks)

        result = factor.compute(winner=winner, turnover=turnover)
        assert result.iloc[0, 0] == pytest.approx(0.65, rel=1e-10)


class TestChipDeviationRateOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        winner = pd.DataFrame(np.random.uniform(0.3, 0.9, (30, 3)), index=dates, columns=stocks)
        turnover = pd.DataFrame(np.random.uniform(0.01, 0.1, (30, 3)), index=dates, columns=stocks)

        result = factor.compute(winner=winner, turnover=turnover)
        assert result.shape == winner.shape
        assert list(result.columns) == list(winner.columns)
        assert list(result.index) == list(winner.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        winner = pd.DataFrame([0.5] * 5, index=dates, columns=stocks)
        turnover = pd.DataFrame([0.1] * 5, index=dates, columns=stocks)

        result = factor.compute(winner=winner, turnover=turnover)
        assert isinstance(result, pd.DataFrame)

    def test_no_nan_in_output(self, factor):
        """正常输入时输出不应有 NaN。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        winner = pd.DataFrame(np.random.uniform(0.3, 0.9, (10, 2)), index=dates, columns=stocks)
        turnover = pd.DataFrame(np.random.uniform(0.01, 0.1, (10, 2)), index=dates, columns=stocks)

        result = factor.compute(winner=winner, turnover=turnover)
        assert result.notna().all().all()
