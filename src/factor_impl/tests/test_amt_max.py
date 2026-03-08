import numpy as np
import pandas as pd
import pytest

from factors.amt_max import AmtMaxFactor


@pytest.fixture
def factor():
    return AmtMaxFactor()


class TestAmtMaxMetadata:
    def test_name(self, factor):
        assert factor.name == "AMT_MAX"

    def test_category(self, factor):
        assert factor.category == "高频动量反转"

    def test_repr(self, factor):
        assert "AMT_MAX" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "AMT_MAX"
        assert meta["category"] == "高频动量反转"


class TestAmtMaxHandCalculated:
    """手算验证 rolling(T).mean()。"""

    def test_constant_input(self, factor):
        """常数输入时, 均值等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(2.5, index=dates, columns=stocks)

        result = factor.compute(daily_amt_max=daily, T=20)
        assert result.iloc[-1, 0] == pytest.approx(2.5, rel=1e-10)

    def test_manual_T3(self, factor):
        """T=3, 手动验证。

        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        T=3:
          row 2: mean(1,2,3) = 2.0
          row 3: mean(2,3,4) = 3.0
          row 4: mean(3,4,5) = 4.0
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0], index=dates, columns=stocks)

        result = factor.compute(daily_amt_max=daily, T=3)

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(2.0, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(3.0, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(4.0, rel=1e-10)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        daily = pd.DataFrame(
            {"A": [1.5] * 25, "B": [3.0] * 25}, index=dates
        )

        result = factor.compute(daily_amt_max=daily, T=20)
        assert result.iloc[-1, 0] == pytest.approx(1.5, rel=1e-10)
        assert result.iloc[-1, 1] == pytest.approx(3.0, rel=1e-10)


class TestAmtMaxEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(np.ones(10) * 2.0, index=dates, columns=stocks)
        daily.iloc[3, 0] = np.nan

        result = factor.compute(daily_amt_max=daily, T=5)
        assert isinstance(result, pd.DataFrame)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_amt_max=daily, T=5)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_amt_max=daily, T=20)
        for val in result.iloc[19:]["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)

    def test_insufficient_window(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(2.0, index=dates, columns=stocks)

        result = factor.compute(daily_amt_max=daily, T=20)
        assert result.isna().all().all()


class TestAmtMaxOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily = pd.DataFrame(
            np.random.uniform(0.5, 5.0, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_amt_max=daily, T=20)

        assert result.shape == daily.shape
        assert list(result.columns) == list(daily.columns)
        assert list(result.index) == list(daily.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(2.0, index=dates, columns=stocks)

        result = factor.compute(daily_amt_max=daily, T=20)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        T = 20
        daily = pd.DataFrame(2.0, index=dates, columns=stocks)

        result = factor.compute(daily_amt_max=daily, T=T)

        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
