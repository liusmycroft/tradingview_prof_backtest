import numpy as np
import pandas as pd
import pytest

from factors.locked_chips_ratio import LockedChipsRatioFactor


@pytest.fixture
def factor():
    return LockedChipsRatioFactor()


class TestLockedChipsRatioMetadata:
    def test_name(self, factor):
        assert factor.name == "LOCKED_CHIPS_RATIO"

    def test_category(self, factor):
        assert factor.category == "行为金融-筹码分布"

    def test_repr(self, factor):
        assert "LOCKED_CHIPS_RATIO" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "LOCKED_CHIPS_RATIO"
        assert meta["category"] == "行为金融-筹码分布"


class TestLockedChipsRatioHandCalculated:
    """用手算数据验证计算的正确性。"""

    def test_constant_input(self, factor):
        """常数输入时, combined = above - below, EMA 应等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily_locked_above = pd.DataFrame(0.3, index=dates, columns=stocks)
        daily_locked_below = pd.DataFrame(0.1, index=dates, columns=stocks)
        # combined = 0.3 - 0.1 = 0.2

        result = factor.compute(
            daily_locked_above=daily_locked_above,
            daily_locked_below=daily_locked_below,
            T=20,
        )

        valid = result.dropna()
        np.testing.assert_array_almost_equal(valid["A"].values, 0.2)

    def test_equal_above_below(self, factor):
        """上方和下方占比相等时, 因子为 0。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily_locked_above = pd.DataFrame(0.2, index=dates, columns=stocks)
        daily_locked_below = pd.DataFrame(0.2, index=dates, columns=stocks)

        result = factor.compute(
            daily_locked_above=daily_locked_above,
            daily_locked_below=daily_locked_below,
            T=20,
        )

        valid = result.dropna()
        np.testing.assert_array_almost_equal(valid["A"].values, 0.0)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A", "B"]
        daily_locked_above = pd.DataFrame(
            {"A": [0.3] * 25, "B": [0.1] * 25}, index=dates
        )
        daily_locked_below = pd.DataFrame(
            {"A": [0.1] * 25, "B": [0.3] * 25}, index=dates
        )
        # A: 0.3-0.1=0.2, B: 0.1-0.3=-0.2

        result = factor.compute(
            daily_locked_above=daily_locked_above,
            daily_locked_below=daily_locked_below,
            T=20,
        )

        valid = result.dropna()
        np.testing.assert_array_almost_equal(valid["A"].values, 0.2)
        np.testing.assert_array_almost_equal(valid["B"].values, -0.2)

    def test_ema_weights_recent_more(self, factor):
        """EMA 应对近期数据赋予更高权重。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        above_vals = [0.1] * 20 + [0.5] * 5
        daily_locked_above = pd.DataFrame(above_vals, index=dates, columns=stocks)
        daily_locked_below = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_locked_above=daily_locked_above,
            daily_locked_below=daily_locked_below,
            T=20,
        )

        last_val = result.iloc[-1, 0]
        assert 0.1 < last_val < 0.5


class TestLockedChipsRatioEdgeCases:
    def test_nan_in_input(self, factor):
        """输入含 NaN 时不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        above = np.ones(25) * 0.3
        above[10] = np.nan
        daily_locked_above = pd.DataFrame(above, index=dates, columns=stocks)
        daily_locked_below = pd.DataFrame(0.1, index=dates, columns=stocks)

        result = factor.compute(
            daily_locked_above=daily_locked_above,
            daily_locked_below=daily_locked_below,
            T=20,
        )
        assert result.shape == (25, 1)

    def test_all_nan(self, factor):
        """全 NaN 输入时, 结果应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily_locked_above = pd.DataFrame(np.nan, index=dates, columns=stocks)
        daily_locked_below = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(
            daily_locked_above=daily_locked_above,
            daily_locked_below=daily_locked_below,
            T=20,
        )
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        """全零输入时, 结果应全为 0。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily_locked_above = pd.DataFrame(0.0, index=dates, columns=stocks)
        daily_locked_below = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_locked_above=daily_locked_above,
            daily_locked_below=daily_locked_below,
            T=20,
        )
        valid = result.dropna()
        for val in valid["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)


class TestLockedChipsRatioOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily_locked_above = pd.DataFrame(
            np.random.uniform(0, 0.5, (30, 3)), index=dates, columns=stocks
        )
        daily_locked_below = pd.DataFrame(
            np.random.uniform(0, 0.5, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(
            daily_locked_above=daily_locked_above,
            daily_locked_below=daily_locked_below,
            T=20,
        )

        assert result.shape == daily_locked_above.shape
        assert list(result.columns) == list(daily_locked_above.columns)
        assert list(result.index) == list(daily_locked_above.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily_locked_above = pd.DataFrame([0.3] * 25, index=dates, columns=stocks)
        daily_locked_below = pd.DataFrame([0.1] * 25, index=dates, columns=stocks)

        result = factor.compute(
            daily_locked_above=daily_locked_above,
            daily_locked_below=daily_locked_below,
            T=20,
        )
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        """min_periods=T, 前 T-1 行应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        T = 20
        daily_locked_above = pd.DataFrame(
            np.random.uniform(0, 0.5, (30, 2)), index=dates, columns=stocks
        )
        daily_locked_below = pd.DataFrame(
            np.random.uniform(0, 0.5, (30, 2)), index=dates, columns=stocks
        )

        result = factor.compute(
            daily_locked_above=daily_locked_above,
            daily_locked_below=daily_locked_below,
            T=T,
        )

        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
