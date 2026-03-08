import numpy as np
import pandas as pd
import pytest

from factors.chip_turnover import ChipTurnoverFactor


@pytest.fixture
def factor():
    return ChipTurnoverFactor()


class TestChipTurnoverMetadata:
    def test_name(self, factor):
        assert factor.name == "CHIP_TURNOVER"

    def test_category(self, factor):
        assert factor.category == "行为金融-筹码分布"

    def test_repr(self, factor):
        assert "CHIP_TURNOVER" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "CHIP_TURNOVER"
        assert meta["category"] == "行为金融-筹码分布"


class TestChipTurnoverHandCalculated:
    """手算验证 rolling(T).mean()。"""

    def test_constant_input(self, factor):
        """常数输入时, 均值等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(0.05, index=dates, columns=stocks)

        result = factor.compute(daily_chip_turnover=daily, T=20)
        assert result.iloc[-1, 0] == pytest.approx(0.05, rel=1e-10)

    def test_manual_T3(self, factor):
        """T=3, 手动验证。

        data = [0.01, 0.02, 0.03, 0.04, 0.05]
        T=3:
          row 2: mean(0.01, 0.02, 0.03) = 0.02
          row 3: mean(0.02, 0.03, 0.04) = 0.03
          row 4: mean(0.03, 0.04, 0.05) = 0.04
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(
            [0.01, 0.02, 0.03, 0.04, 0.05], index=dates, columns=stocks
        )

        result = factor.compute(daily_chip_turnover=daily, T=3)

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(0.02, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(0.03, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(0.04, rel=1e-10)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        daily = pd.DataFrame(
            {"A": [0.03] * 25, "B": [0.08] * 25}, index=dates
        )

        result = factor.compute(daily_chip_turnover=daily, T=20)
        assert result.iloc[-1, 0] == pytest.approx(0.03, rel=1e-10)
        assert result.iloc[-1, 1] == pytest.approx(0.08, rel=1e-10)

    def test_step_change(self, factor):
        """前10天0.02, 后10天0.06, T=20均值=0.04。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        vals = [0.02] * 10 + [0.06] * 10
        daily = pd.DataFrame(vals, index=dates, columns=stocks)

        result = factor.compute(daily_chip_turnover=daily, T=20)
        assert result.iloc[-1, 0] == pytest.approx(0.04, rel=1e-6)


class TestChipTurnoverEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(np.ones(10) * 0.05, index=dates, columns=stocks)
        daily.iloc[3, 0] = np.nan

        result = factor.compute(daily_chip_turnover=daily, T=5)
        assert isinstance(result, pd.DataFrame)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_chip_turnover=daily, T=5)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_chip_turnover=daily, T=20)
        for val in result.iloc[19:]["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)

    def test_insufficient_window(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(0.05, index=dates, columns=stocks)

        result = factor.compute(daily_chip_turnover=daily, T=20)
        assert result.isna().all().all()


class TestChipTurnoverOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily = pd.DataFrame(
            np.random.uniform(0, 0.1, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_chip_turnover=daily, T=20)

        assert result.shape == daily.shape
        assert list(result.columns) == list(daily.columns)
        assert list(result.index) == list(daily.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(0.05, index=dates, columns=stocks)

        result = factor.compute(daily_chip_turnover=daily, T=20)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        T = 20
        daily = pd.DataFrame(0.05, index=dates, columns=stocks)

        result = factor.compute(daily_chip_turnover=daily, T=T)

        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
