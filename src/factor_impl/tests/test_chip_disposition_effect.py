import numpy as np
import pandas as pd
import pytest

from factors.chip_disposition_effect import ChipDispositionEffectFactor


@pytest.fixture
def factor():
    return ChipDispositionEffectFactor()


class TestChipDispositionEffectMetadata:
    def test_name(self, factor):
        assert factor.name == "CHIP_DISPOSITION_EFFECT"

    def test_category(self, factor):
        assert factor.category == "行为金融-处置效应"

    def test_repr(self, factor):
        assert "CHIP_DISPOSITION_EFFECT" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "CHIP_DISPOSITION_EFFECT"
        assert meta["category"] == "行为金融-处置效应"


class TestChipDispositionEffectCompute:
    def test_constant_input(self, factor):
        """常数输入验证。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        profit = pd.DataFrame(0.6, index=dates, columns=stocks)
        loss = pd.DataFrame(0.4, index=dates, columns=stocks)
        turnover = pd.DataFrame(0.05, index=dates, columns=stocks)

        result = factor.compute(
            profit_chip_ratio=profit, loss_chip_ratio=loss, turnover=turnover, T=5
        )
        # PGR = 0.6 * 0.05 = 0.03, PLR = 0.4 * 0.05 = 0.02, CDE = 0.01
        np.testing.assert_array_almost_equal(result["A"].values, 0.01)

    def test_no_disposition_effect(self, factor):
        """盈利和亏损筹码占比相等时，CDE = 0。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        profit = pd.DataFrame(0.5, index=dates, columns=stocks)
        loss = pd.DataFrame(0.5, index=dates, columns=stocks)
        turnover = pd.DataFrame(0.1, index=dates, columns=stocks)

        result = factor.compute(
            profit_chip_ratio=profit, loss_chip_ratio=loss, turnover=turnover, T=5
        )
        np.testing.assert_array_almost_equal(result["A"].values, 0.0)

    def test_negative_cde(self, factor):
        """亏损筹码占比大于盈利时，CDE 为负。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        profit = pd.DataFrame(0.2, index=dates, columns=stocks)
        loss = pd.DataFrame(0.8, index=dates, columns=stocks)
        turnover = pd.DataFrame(0.1, index=dates, columns=stocks)

        result = factor.compute(
            profit_chip_ratio=profit, loss_chip_ratio=loss, turnover=turnover, T=3
        )
        # PGR = 0.02, PLR = 0.08, CDE = -0.06
        np.testing.assert_array_almost_equal(result["A"].values, -0.06)

    def test_manual_rolling_T3(self, factor):
        """T=3 手动验证滚动均值。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        profit = pd.DataFrame([0.6, 0.7, 0.5, 0.8, 0.4], index=dates, columns=stocks)
        loss = pd.DataFrame([0.4, 0.3, 0.5, 0.2, 0.6], index=dates, columns=stocks)
        turnover = pd.DataFrame([0.1, 0.1, 0.1, 0.1, 0.1], index=dates, columns=stocks)

        result = factor.compute(
            profit_chip_ratio=profit, loss_chip_ratio=loss, turnover=turnover, T=3
        )

        # daily_cde = (profit - loss) * turnover = [0.02, 0.04, 0.0, 0.06, -0.02]
        # rolling mean T=3, min_periods=1:
        #   [0.02, 0.03, 0.02, 0.03333, 0.01333]
        assert result.iloc[0, 0] == pytest.approx(0.02, rel=1e-6)
        assert result.iloc[1, 0] == pytest.approx(0.03, rel=1e-6)
        assert result.iloc[2, 0] == pytest.approx(0.02, rel=1e-6)
        assert result.iloc[3, 0] == pytest.approx(0.1 / 3, rel=1e-6)
        assert result.iloc[4, 0] == pytest.approx(0.04 / 3, rel=1e-6)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        profit = pd.DataFrame({"A": [0.7] * 10, "B": [0.3] * 10}, index=dates)
        loss = pd.DataFrame({"A": [0.3] * 10, "B": [0.7] * 10}, index=dates)
        turnover = pd.DataFrame({"A": [0.1] * 10, "B": [0.1] * 10}, index=dates)

        result = factor.compute(
            profit_chip_ratio=profit, loss_chip_ratio=loss, turnover=turnover, T=5
        )
        np.testing.assert_array_almost_equal(result["A"].values, 0.04)
        np.testing.assert_array_almost_equal(result["B"].values, -0.04)


class TestChipDispositionEffectEdgeCases:
    def test_zero_turnover(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        profit = pd.DataFrame(0.6, index=dates, columns=stocks)
        loss = pd.DataFrame(0.4, index=dates, columns=stocks)
        turnover = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(
            profit_chip_ratio=profit, loss_chip_ratio=loss, turnover=turnover, T=3
        )
        np.testing.assert_array_almost_equal(result["A"].values, 0.0)

    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        profit = pd.DataFrame([0.6, np.nan, 0.5, 0.7, 0.4], index=dates, columns=stocks)
        loss = pd.DataFrame(0.4, index=dates, columns=stocks)
        turnover = pd.DataFrame(0.1, index=dates, columns=stocks)

        result = factor.compute(
            profit_chip_ratio=profit, loss_chip_ratio=loss, turnover=turnover, T=3
        )
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (5, 1)


class TestChipDispositionEffectOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        profit = pd.DataFrame(np.random.uniform(0, 1, (30, 3)), index=dates, columns=stocks)
        loss = pd.DataFrame(np.random.uniform(0, 1, (30, 3)), index=dates, columns=stocks)
        turnover = pd.DataFrame(np.random.uniform(0.01, 0.1, (30, 3)), index=dates, columns=stocks)

        result = factor.compute(
            profit_chip_ratio=profit, loss_chip_ratio=loss, turnover=turnover
        )
        assert result.shape == profit.shape
        assert list(result.columns) == list(profit.columns)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        profit = pd.DataFrame(0.6, index=dates, columns=stocks)
        loss = pd.DataFrame(0.4, index=dates, columns=stocks)
        turnover = pd.DataFrame(0.1, index=dates, columns=stocks)

        result = factor.compute(
            profit_chip_ratio=profit, loss_chip_ratio=loss, turnover=turnover, T=3
        )
        assert isinstance(result, pd.DataFrame)
