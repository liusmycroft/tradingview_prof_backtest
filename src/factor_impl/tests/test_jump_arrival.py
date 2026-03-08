import numpy as np
import pandas as pd
import pytest

from factors.jump_arrival import JumpArrivalFactor


@pytest.fixture
def factor():
    return JumpArrivalFactor()


class TestJumpArrivalMetadata:
    def test_name(self, factor):
        assert factor.name == "JUMP_ARRIVAL"

    def test_category(self, factor):
        assert factor.category == "高频波动跳跃"

    def test_repr(self, factor):
        assert "JUMP_ARRIVAL" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "JUMP_ARRIVAL"


class TestJumpArrivalCompute:
    def test_all_jumps(self, factor):
        """所有天都有跳跃时，JArr=1.0。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        daily_jump = pd.DataFrame({"A": np.ones(20)}, index=dates)

        result = factor.compute(daily_jump_indicator=daily_jump, D=20)
        assert result.iloc[-1, 0] == pytest.approx(1.0)

    def test_no_jumps(self, factor):
        """没有跳跃时，JArr=0.0。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        daily_jump = pd.DataFrame({"A": np.zeros(20)}, index=dates)

        result = factor.compute(daily_jump_indicator=daily_jump, D=20)
        assert result.iloc[-1, 0] == pytest.approx(0.0)

    def test_half_jumps(self, factor):
        """一半天有跳跃时，JArr=0.5。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        vals = [1.0] * 10 + [0.0] * 10
        daily_jump = pd.DataFrame({"A": vals}, index=dates)

        result = factor.compute(daily_jump_indicator=daily_jump, D=20)
        assert result.iloc[-1, 0] == pytest.approx(0.5)

    def test_rolling_window(self, factor):
        """验证滚动窗口行为。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        daily_jump = pd.DataFrame({"A": [1.0, 0.0, 1.0, 0.0, 1.0]}, index=dates)

        result = factor.compute(daily_jump_indicator=daily_jump, D=3)
        # D=3, min_periods=1
        # row 0: mean([1]) = 1.0
        # row 1: mean([1,0]) = 0.5
        # row 2: mean([1,0,1]) = 2/3
        # row 3: mean([0,1,0]) = 1/3
        # row 4: mean([1,0,1]) = 2/3
        assert result.iloc[0, 0] == pytest.approx(1.0)
        assert result.iloc[1, 0] == pytest.approx(0.5)
        assert result.iloc[2, 0] == pytest.approx(2.0 / 3.0, rel=1e-6)

    def test_multi_stock(self, factor):
        """多只股票独立计算。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        daily_jump = pd.DataFrame({
            "A": np.ones(10),
            "B": np.zeros(10),
        }, index=dates)

        result = factor.compute(daily_jump_indicator=daily_jump, D=10)
        assert result.iloc[-1, 0] == pytest.approx(1.0)
        assert result.iloc[-1, 1] == pytest.approx(0.0)


class TestJumpArrivalEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        daily_jump = pd.DataFrame({"A": [1.0, 0.0, np.nan, 1.0, 0.0,
                                          1.0, 0.0, 1.0, 0.0, 1.0]}, index=dates)

        result = factor.compute(daily_jump_indicator=daily_jump, D=5)
        assert isinstance(result, pd.DataFrame)

    def test_single_value(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        daily_jump = pd.DataFrame({"A": [1.0]}, index=dates)

        result = factor.compute(daily_jump_indicator=daily_jump, D=20)
        assert result.iloc[0, 0] == pytest.approx(1.0)


class TestJumpArrivalOutputShape:
    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily_jump = pd.DataFrame(
            np.random.choice([0.0, 1.0], (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_jump_indicator=daily_jump, D=20)
        assert result.shape == (30, 3)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        daily_jump = pd.DataFrame({"A": [1.0, 0.0, 1.0, 0.0, 1.0]}, index=dates)

        result = factor.compute(daily_jump_indicator=daily_jump, D=5)
        assert isinstance(result, pd.DataFrame)
