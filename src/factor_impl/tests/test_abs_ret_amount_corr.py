import numpy as np
import pandas as pd
import pytest

from factors.abs_ret_amount_corr import AbsRetAmountCorrFactor


@pytest.fixture
def factor():
    return AbsRetAmountCorrFactor()


class TestAbsRetAmountCorrMetadata:
    def test_name(self, factor):
        assert factor.name == "CORA"

    def test_category(self, factor):
        assert factor.category == "高频量价相关性"

    def test_repr(self, factor):
        assert "CORA" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "CORA"
        assert meta["category"] == "高频量价相关性"


class TestAbsRetAmountCorrHandCalculated:
    def test_constant_input(self, factor):
        """常数输入时, 滚动均值应等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_cora = pd.DataFrame(0.3, index=dates, columns=stocks)

        result = factor.compute(daily_cora=daily_cora, T=3)

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(0.3, rel=1e-10)

    def test_varying_T3(self, factor):
        """T=3, 变化的相关性。

        cora = [0.2, 0.4, 0.6]
        rolling(3): day2: mean(0.2, 0.4, 0.6) = 0.4
        """
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]
        daily_cora = pd.DataFrame([0.2, 0.4, 0.6], index=dates, columns=stocks)

        result = factor.compute(daily_cora=daily_cora, T=3)
        assert result.iloc[2, 0] == pytest.approx(0.4, rel=1e-10)

    def test_negative_corr(self, factor):
        """负相关性 -0.5, T=3 => 均值 = -0.5。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_cora = pd.DataFrame(-0.5, index=dates, columns=stocks)

        result = factor.compute(daily_cora=daily_cora, T=3)
        assert result.iloc[2, 0] == pytest.approx(-0.5, rel=1e-10)

    def test_rolling_window_slides(self, factor):
        """验证滚动窗口正确滑动 (T=3)。

        cora = [1, 2, 3, 4, 5]
        rolling(3):
          day2: mean(1,2,3) = 2
          day3: mean(2,3,4) = 3
          day4: mean(3,4,5) = 4
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_cora = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0], index=dates, columns=stocks)

        result = factor.compute(daily_cora=daily_cora, T=3)
        assert result.iloc[2, 0] == pytest.approx(2.0, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(3.0, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(4.0, rel=1e-10)

    def test_two_stocks(self, factor):
        """两只股票并行计算。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A", "B"]
        daily_cora = pd.DataFrame(
            {"A": [0.1, 0.2, 0.3], "B": [-0.1, -0.2, -0.3]}, index=dates
        )

        result = factor.compute(daily_cora=daily_cora, T=3)
        assert result.loc[dates[2], "A"] == pytest.approx(0.2, rel=1e-10)
        assert result.loc[dates[2], "B"] == pytest.approx(-0.2, rel=1e-10)


class TestAbsRetAmountCorrEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]
        daily_cora = pd.DataFrame([0.3, np.nan, 0.5], index=dates, columns=stocks)

        result = factor.compute(daily_cora=daily_cora, T=3)
        assert np.isnan(result.iloc[2, 0])

    def test_all_zero(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_cora = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_cora=daily_cora, T=3)
        assert result.iloc[2, 0] == pytest.approx(0.0, abs=1e-15)

    def test_insufficient_data(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_cora = pd.DataFrame(0.3, index=dates, columns=stocks)

        result = factor.compute(daily_cora=daily_cora, T=20)
        assert result.isna().all().all()


class TestAbsRetAmountCorrOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        stocks = ["A", "B", "C"]
        daily_cora = pd.DataFrame(
            np.random.uniform(-1, 1, (50, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_cora=daily_cora, T=20)
        assert result.shape == daily_cora.shape
        assert list(result.columns) == list(daily_cora.columns)
        assert list(result.index) == list(daily_cora.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_cora = pd.DataFrame(0.3, index=dates, columns=stocks)

        result = factor.compute(daily_cora=daily_cora, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        T = 5
        daily_cora = pd.DataFrame(
            np.random.uniform(-1, 1, (10, 2)), index=dates, columns=stocks
        )

        result = factor.compute(daily_cora=daily_cora, T=T)
        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
