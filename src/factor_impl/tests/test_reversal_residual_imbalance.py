import numpy as np
import pandas as pd
import pytest

from factors.reversal_residual_imbalance import ReversalResidualImbalanceFactor


@pytest.fixture
def factor():
    return ReversalResidualImbalanceFactor()


class TestReversalResidualImbalanceMetadata:
    def test_name(self, factor):
        assert factor.name == "REVERSAL_RESIDUAL_IMBALANCE"

    def test_category(self, factor):
        assert factor.category == "高频成交分布"

    def test_repr(self, factor):
        assert "REVERSAL_RESIDUAL_IMBALANCE" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "REVERSAL_RESIDUAL_IMBALANCE"
        assert meta["category"] == "高频成交分布"


class TestReversalResidualImbalanceHandCalculated:
    """手算验证 (reversal_residual * non_isolated_imbalance).rolling(T).mean()。"""

    def test_constant_product(self, factor):
        """常数输入时，乘积恒定，均值等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        residual = pd.DataFrame(2.0, index=dates, columns=stocks)
        imbalance = pd.DataFrame(3.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_reversal_residual=residual,
            daily_non_isolated_imbalance=imbalance,
            T=20,
        )
        assert result.iloc[-1, 0] == pytest.approx(6.0, rel=1e-10)

    def test_manual_T3(self, factor):
        """T=3, 手动验证。

        residual  = [1, 2, 3, 4, 5]
        imbalance = [2, 2, 2, 2, 2]
        product   = [2, 4, 6, 8, 10]
        T=3:
          row 2: mean(2,4,6) = 4.0
          row 3: mean(4,6,8) = 6.0
          row 4: mean(6,8,10) = 8.0
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        residual = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0], index=dates, columns=stocks)
        imbalance = pd.DataFrame(2.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_reversal_residual=residual,
            daily_non_isolated_imbalance=imbalance,
            T=3,
        )

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(4.0, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(6.0, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(8.0, rel=1e-10)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        residual = pd.DataFrame({"A": [1.0] * 25, "B": [3.0] * 25}, index=dates)
        imbalance = pd.DataFrame({"A": [2.0] * 25, "B": [4.0] * 25}, index=dates)

        result = factor.compute(
            daily_reversal_residual=residual,
            daily_non_isolated_imbalance=imbalance,
            T=20,
        )
        assert result.iloc[-1, 0] == pytest.approx(2.0, rel=1e-10)
        assert result.iloc[-1, 1] == pytest.approx(12.0, rel=1e-10)

    def test_negative_values(self, factor):
        """负值输入应正常处理。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        residual = pd.DataFrame(-1.0, index=dates, columns=stocks)
        imbalance = pd.DataFrame(2.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_reversal_residual=residual,
            daily_non_isolated_imbalance=imbalance,
            T=20,
        )
        assert result.iloc[-1, 0] == pytest.approx(-2.0, rel=1e-10)


class TestReversalResidualImbalanceEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        residual = pd.DataFrame(np.ones(10), index=dates, columns=stocks)
        residual.iloc[3, 0] = np.nan
        imbalance = pd.DataFrame(2.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_reversal_residual=residual,
            daily_non_isolated_imbalance=imbalance,
            T=5,
        )
        assert isinstance(result, pd.DataFrame)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        residual = pd.DataFrame(np.nan, index=dates, columns=stocks)
        imbalance = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(
            daily_reversal_residual=residual,
            daily_non_isolated_imbalance=imbalance,
            T=5,
        )
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        residual = pd.DataFrame(0.0, index=dates, columns=stocks)
        imbalance = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_reversal_residual=residual,
            daily_non_isolated_imbalance=imbalance,
            T=20,
        )
        for val in result.iloc[19:]["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)

    def test_insufficient_window(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        residual = pd.DataFrame(1.0, index=dates, columns=stocks)
        imbalance = pd.DataFrame(2.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_reversal_residual=residual,
            daily_non_isolated_imbalance=imbalance,
            T=20,
        )
        assert result.isna().all().all()


class TestReversalResidualImbalanceOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        residual = pd.DataFrame(
            np.random.randn(30, 3), index=dates, columns=stocks
        )
        imbalance = pd.DataFrame(
            np.random.randn(30, 3), index=dates, columns=stocks
        )

        result = factor.compute(
            daily_reversal_residual=residual,
            daily_non_isolated_imbalance=imbalance,
            T=20,
        )

        assert result.shape == residual.shape
        assert list(result.columns) == list(residual.columns)
        assert list(result.index) == list(residual.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        residual = pd.DataFrame(1.0, index=dates, columns=stocks)
        imbalance = pd.DataFrame(2.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_reversal_residual=residual,
            daily_non_isolated_imbalance=imbalance,
            T=20,
        )
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        T = 20
        residual = pd.DataFrame(1.0, index=dates, columns=stocks)
        imbalance = pd.DataFrame(2.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_reversal_residual=residual,
            daily_non_isolated_imbalance=imbalance,
            T=T,
        )

        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
