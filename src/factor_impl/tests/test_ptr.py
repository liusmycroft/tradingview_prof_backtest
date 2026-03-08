import numpy as np
import pandas as pd
import pytest

from factors.ptr import PTRFactor


@pytest.fixture
def factor():
    return PTRFactor()


class TestMetadata:
    def test_name(self, factor):
        assert factor.name == "PTR"

    def test_category(self, factor):
        assert factor.category == "行为金融-筹码分布"

    def test_repr(self, factor):
        assert "PTR" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "PTR"
        assert meta["category"] == "行为金融-筹码分布"


class TestHandCalculated:
    def test_constant_winner_zero_ptr(self, factor):
        """winner 不变时, diff=0, PTR=0"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        winner = pd.DataFrame(0.5, index=dates, columns=stocks)
        turnover = pd.DataFrame(0.03, index=dates, columns=stocks)

        result = factor.compute(winner=winner, turnover=turnover, T=20)
        # 第一行 diff=NaN, 所以需要 T+1 行才有有效值
        # 从 row 1 开始 diff=0, 需要 T=20 行 => row 20 开始有值
        assert result.iloc[-1, 0] == pytest.approx(0.0, abs=1e-15)

    def test_linearly_increasing_winner(self, factor):
        """winner 线性增长, diff 恒定。

        winner = [0.1, 0.2, 0.3, 0.4, 0.5]
        diff   = [NaN, 0.1, 0.1, 0.1, 0.1]
        turnover = 0.05
        ptr    = [NaN, 2.0, 2.0, 2.0, 2.0]
        T=3: day3=mean(2,2,2)=2.0, day4=mean(2,2,2)=2.0
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        winner = pd.DataFrame([0.1, 0.2, 0.3, 0.4, 0.5], index=dates, columns=stocks)
        turnover = pd.DataFrame(0.05, index=dates, columns=stocks)

        result = factor.compute(winner=winner, turnover=turnover, T=3)
        # row 0: diff=NaN => ptr=NaN
        # row 1: diff=0.1, ptr=2.0
        # row 2: diff=0.1, ptr=2.0
        # row 3: diff=0.1, ptr=2.0 => rolling(3) from row1..3 = mean(2,2,2)=2.0
        assert result.iloc[3, 0] == pytest.approx(2.0, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(2.0, rel=1e-10)

    def test_varying_ptr_T3(self, factor):
        """T=3, 手动验证。

        winner = [0.1, 0.3, 0.4, 0.8, 1.0]
        diff   = [NaN, 0.2, 0.1, 0.4, 0.2]
        turnover = 0.1
        ptr    = [NaN, 2.0, 1.0, 4.0, 2.0]
        T=3: row3=mean(2,1,4)=7/3, row4=mean(1,4,2)=7/3
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        winner = pd.DataFrame([0.1, 0.3, 0.4, 0.8, 1.0], index=dates, columns=stocks)
        turnover = pd.DataFrame(0.1, index=dates, columns=stocks)

        result = factor.compute(winner=winner, turnover=turnover, T=3)
        assert result.iloc[3, 0] == pytest.approx(7.0 / 3, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(7.0 / 3, rel=1e-10)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        # A: winner constant => ptr=0; B: winner +0.01/day => ptr=0.01/0.05=0.2
        winner_a = [0.5] * 25
        winner_b = [0.3 + 0.01 * i for i in range(25)]
        winner = pd.DataFrame({"A": winner_a, "B": winner_b}, index=dates)
        turnover = pd.DataFrame({"A": [0.05] * 25, "B": [0.05] * 25}, index=dates)

        result = factor.compute(winner=winner, turnover=turnover, T=20)
        assert result.iloc[-1, 0] == pytest.approx(0.0, abs=1e-10)
        assert result.iloc[-1, 1] == pytest.approx(0.2, rel=1e-6)


class TestEdgeCases:
    def test_nan_in_input(self, factor):
        """输入含 NaN 时, 不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        winner = pd.DataFrame([0.1, np.nan, 0.3, 0.4, 0.5], index=dates, columns=stocks)
        turnover = pd.DataFrame(0.05, index=dates, columns=stocks)

        result = factor.compute(winner=winner, turnover=turnover, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_zero_turnover(self, factor):
        """turnover 为零时, 结果应为 inf/NaN, 不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        winner = pd.DataFrame([0.1, 0.2, 0.3, 0.4, 0.5], index=dates, columns=stocks)
        turnover = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(winner=winner, turnover=turnover, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_insufficient_data(self, factor):
        """数据不足 T 天时, 全部为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        winner = pd.DataFrame(0.5, index=dates, columns=stocks)
        turnover = pd.DataFrame(0.05, index=dates, columns=stocks)

        result = factor.compute(winner=winner, turnover=turnover, T=20)
        assert result.isna().all().all()

    def test_all_nan(self, factor):
        """全 NaN 输入时, 结果应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        winner = pd.DataFrame(np.nan, index=dates, columns=stocks)
        turnover = pd.DataFrame(0.05, index=dates, columns=stocks)

        result = factor.compute(winner=winner, turnover=turnover, T=5)
        assert result.isna().all().all()


class TestOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        winner = pd.DataFrame(np.random.rand(30, 3) * 0.5 + 0.3, index=dates, columns=stocks)
        turnover = pd.DataFrame(np.random.rand(30, 3) * 0.05 + 0.01, index=dates, columns=stocks)

        result = factor.compute(winner=winner, turnover=turnover, T=20)
        assert result.shape == winner.shape
        assert list(result.columns) == list(winner.columns)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        winner = pd.DataFrame(0.5, index=dates, columns=stocks)
        turnover = pd.DataFrame(0.05, index=dates, columns=stocks)

        result = factor.compute(winner=winner, turnover=turnover, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_first_row_always_nan(self, factor):
        """第一行 winner_diff 为 NaN, 所以 PTR 第一行始终为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        winner = pd.DataFrame(np.random.rand(10, 2), index=dates, columns=stocks)
        turnover = pd.DataFrame(np.random.rand(10, 2) * 0.05 + 0.01, index=dates, columns=stocks)

        result = factor.compute(winner=winner, turnover=turnover, T=3)
        assert result.iloc[0].isna().all()
