import numpy as np
import pandas as pd
import pytest

from factors.weighted_profit_freq import WeightedProfitFreqFactor


@pytest.fixture
def factor():
    return WeightedProfitFreqFactor()


class TestWeightedProfitFreqMetadata:
    def test_name(self, factor):
        assert factor.name == "WEIGHTED_PROFIT_FREQ"

    def test_category(self, factor):
        assert factor.category == "量价因子改进"

    def test_repr(self, factor):
        assert "WEIGHTED_PROFIT_FREQ" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "WEIGHTED_PROFIT_FREQ"
        assert meta["category"] == "量价因子改进"


class TestWeightedProfitFreqHandCalculated:
    """手算验证加权盈利频率。"""

    def test_all_above_threshold(self, factor):
        """所有收益都超过阈值时，权重之和 / M。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        # 所有值都 > 0.02
        excess_return = pd.DataFrame(0.05, index=dates, columns=stocks)

        result = factor.compute(excess_return=excess_return, M=5, u=0.02, lam=10.0)

        # 最后一天: 所有5天都超过阈值
        # weights: 0.5^(4/10), 0.5^(3/10), 0.5^(2/10), 0.5^(1/10), 0.5^(0/10)
        # = 0.5^0.4, 0.5^0.3, 0.5^0.2, 0.5^0.1, 1.0
        w = [0.5 ** (4 / 10), 0.5 ** (3 / 10), 0.5 ** (2 / 10), 0.5 ** (1 / 10), 1.0]
        expected = sum(w) / 5
        assert result.iloc[-1, 0] == pytest.approx(expected, rel=1e-6)

    def test_all_below_threshold(self, factor):
        """所有收益都低于阈值时，因子值为 0。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        excess_return = pd.DataFrame(0.01, index=dates, columns=stocks)

        result = factor.compute(excess_return=excess_return, M=5, u=0.02, lam=10.0)

        for val in result["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)

    def test_single_day_above(self, factor):
        """只有最后一天超过阈值，权重为 1.0 (距离=0)。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]
        excess_return = pd.DataFrame(
            [0.01, 0.01, 0.05], index=dates, columns=stocks
        )

        result = factor.compute(excess_return=excess_return, M=3, u=0.02, lam=10.0)

        # 最后一天: 只有 day2 超过阈值, w = 0.5^(0/10) = 1.0
        expected = 1.0 / 3
        assert result.iloc[-1, 0] == pytest.approx(expected, rel=1e-6)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A", "B"]
        excess_return = pd.DataFrame(
            {"A": [0.05] * 5, "B": [0.01] * 5}, index=dates
        )

        result = factor.compute(excess_return=excess_return, M=5, u=0.02, lam=10.0)

        assert result.iloc[-1]["A"] > 0
        assert result.iloc[-1]["B"] == pytest.approx(0.0, abs=1e-15)

    def test_decay_weight_recent_higher(self, factor):
        """近期超过阈值的贡献应大于远期。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        # A: 只有最后一天超过阈值 (近期)
        # B: 只有第一天超过阈值 (远期)
        vals_a = [0.01] * 9 + [0.05]
        vals_b = [0.05] + [0.01] * 9
        excess_return = pd.DataFrame({"A": vals_a, "B": vals_b}, index=dates)

        result = factor.compute(excess_return=excess_return, M=10, u=0.02, lam=10.0)

        # A 的因子值应大于 B（近期权重更高）
        assert result.iloc[-1]["A"] > result.iloc[-1]["B"]


class TestWeightedProfitFreqEdgeCases:
    def test_single_value(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A"]
        excess_return = pd.DataFrame([0.05], index=dates, columns=stocks)

        result = factor.compute(excess_return=excess_return, M=40, u=0.02, lam=10.0)
        # 1 day above threshold, weight = 1.0, result = 1.0 / 40
        assert result.iloc[0, 0] == pytest.approx(1.0 / 40, rel=1e-6)

    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        values = [0.05, np.nan, 0.03, 0.01, 0.04]
        excess_return = pd.DataFrame(values, index=dates, columns=stocks)

        result = factor.compute(excess_return=excess_return, M=5, u=0.02, lam=10.0)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (5, 1)

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        excess_return = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(excess_return=excess_return, M=5, u=0.02, lam=10.0)
        for val in result["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)


class TestWeightedProfitFreqOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        excess_return = pd.DataFrame(
            np.random.uniform(-0.05, 0.05, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(excess_return=excess_return, M=40, u=0.02, lam=10.0)

        assert result.shape == excess_return.shape
        assert list(result.columns) == list(excess_return.columns)
        assert list(result.index) == list(excess_return.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        excess_return = pd.DataFrame([0.01, 0.02, 0.03, 0.04, 0.05], index=dates, columns=stocks)

        result = factor.compute(excess_return=excess_return, M=5, u=0.02, lam=10.0)
        assert isinstance(result, pd.DataFrame)

    def test_non_negative(self, factor):
        """因子值应非负。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A", "B"]
        np.random.seed(42)
        excess_return = pd.DataFrame(
            np.random.uniform(-0.05, 0.05, (20, 2)), index=dates, columns=stocks
        )

        result = factor.compute(excess_return=excess_return, M=10, u=0.02, lam=10.0)
        assert (result.values[~np.isnan(result.values)] >= 0).all()
