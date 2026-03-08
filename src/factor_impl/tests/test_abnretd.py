import numpy as np
import pandas as pd
import pytest

from factors.abnretd import ABNRETDFactor


@pytest.fixture
def factor():
    return ABNRETDFactor()


class TestABNRETDMetadata:
    def test_name(self, factor):
        assert factor.name == "ABNRETD"

    def test_category(self, factor):
        assert factor.category == "行为金融-投资者注意力"

    def test_repr(self, factor):
        assert "ABNRETD" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "ABNRETD"
        assert meta["category"] == "行为金融-投资者注意力"


class TestABNRETDCompute:
    def test_known_values(self, factor):
        """手算验证: max(|r_i - r_mkt|) over rolling window."""
        dates = pd.bdate_range("2025-01-01", periods=5)
        stock_returns = pd.DataFrame({"A": [0.02, -0.01, 0.05, 0.00, -0.03]}, index=dates)
        market_returns = pd.Series([0.01, 0.00, 0.02, -0.01, 0.01], index=dates)

        result = factor.compute(
            stock_returns=stock_returns, market_returns=market_returns, T=3,
        )

        # abnormal: |0.01|, |−0.01|, |0.03|, |0.01|, |−0.04|
        #         = [0.01, 0.01, 0.03, 0.01, 0.04]
        # T=3 rolling max:
        #   row 0,1: NaN
        #   row 2: max(0.01, 0.01, 0.03) = 0.03
        #   row 3: max(0.01, 0.03, 0.01) = 0.03
        #   row 4: max(0.03, 0.01, 0.04) = 0.04
        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(0.03)
        assert result.iloc[3, 0] == pytest.approx(0.03)
        assert result.iloc[4, 0] == pytest.approx(0.04)

    def test_zero_market_return(self, factor):
        """市场收益为0时, ABNRETD = max(|r_i|)。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        stock_returns = pd.DataFrame({"A": [0.01, -0.02, 0.03, -0.04, 0.05]}, index=dates)
        market_returns = pd.Series([0.0] * 5, index=dates)

        result = factor.compute(
            stock_returns=stock_returns, market_returns=market_returns, T=3,
        )

        assert result.iloc[2, 0] == pytest.approx(0.03)
        assert result.iloc[3, 0] == pytest.approx(0.04)
        assert result.iloc[4, 0] == pytest.approx(0.05)

    def test_stock_equals_market(self, factor):
        """个股收益 == 市场收益时, ABNRETD = 0。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02], index=dates)
        stock_returns = pd.DataFrame({"A": returns}, index=dates)

        result = factor.compute(
            stock_returns=stock_returns, market_returns=returns, T=3,
        )

        for i in range(2, 5):
            assert result.iloc[i, 0] == pytest.approx(0.0, abs=1e-15)

    def test_multi_stock(self, factor):
        """多只股票独立计算。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        stock_returns = pd.DataFrame(
            {"A": [0.02, -0.01, 0.05, 0.00, -0.03],
             "B": [0.01, 0.01, 0.01, 0.01, 0.01]},
            index=dates,
        )
        market_returns = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01], index=dates)

        result = factor.compute(
            stock_returns=stock_returns, market_returns=market_returns, T=3,
        )

        assert result.shape == (5, 2)
        # B always equals market, so ABNRETD_B = 0
        for i in range(2, 5):
            assert result.loc[dates[i], "B"] == pytest.approx(0.0, abs=1e-15)

    def test_result_non_negative(self, factor):
        """ABNRETD 应始终非负（绝对值的最大值）。"""
        np.random.seed(42)
        dates = pd.bdate_range("2025-01-01", periods=30)
        stock_returns = pd.DataFrame(
            np.random.randn(30, 2) * 0.02, index=dates, columns=["A", "B"]
        )
        market_returns = pd.Series(np.random.randn(30) * 0.01, index=dates)

        result = factor.compute(
            stock_returns=stock_returns, market_returns=market_returns, T=20,
        )

        valid = result.dropna()
        assert (valid >= 0).all().all()


class TestABNRETDEdgeCases:
    def test_nan_in_stock_returns(self, factor):
        """个股收益含 NaN 时不应抛异常。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        stock_returns = pd.DataFrame({"A": [0.01, np.nan, 0.03, 0.02, 0.01]}, index=dates)
        market_returns = pd.Series([0.01] * 5, index=dates)

        result = factor.compute(
            stock_returns=stock_returns, market_returns=market_returns, T=3,
        )

        assert isinstance(result, pd.DataFrame)


class TestABNRETDOutputShape:
    def test_output_is_dataframe(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=5)
        stock_returns = pd.DataFrame({"A": np.random.randn(5) * 0.02}, index=dates)
        market_returns = pd.Series(np.random.randn(5) * 0.01, index=dates)

        result = factor.compute(
            stock_returns=stock_returns, market_returns=market_returns, T=3,
        )
        assert isinstance(result, pd.DataFrame)

    def test_output_shape_matches_input(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=30)
        stocks = ["A", "B", "C"]
        stock_returns = pd.DataFrame(
            np.random.randn(30, 3) * 0.02, index=dates, columns=stocks
        )
        market_returns = pd.Series(np.random.randn(30) * 0.01, index=dates)

        result = factor.compute(
            stock_returns=stock_returns, market_returns=market_returns, T=20,
        )
        assert result.shape == stock_returns.shape
        assert list(result.columns) == stocks

    def test_first_T_minus_1_rows_are_nan(self, factor):
        """前 T-1 行应全为 NaN。"""
        dates = pd.bdate_range("2025-01-01", periods=25)
        stock_returns = pd.DataFrame(
            np.random.randn(25, 2) * 0.02, index=dates, columns=["A", "B"]
        )
        market_returns = pd.Series(np.random.randn(25) * 0.01, index=dates)
        T = 20

        result = factor.compute(
            stock_returns=stock_returns, market_returns=market_returns, T=T,
        )

        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
