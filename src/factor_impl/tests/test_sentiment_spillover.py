import numpy as np
import pandas as pd
import pytest

from factors.sentiment_spillover import SentimentSpilloverFactor


@pytest.fixture
def factor():
    return SentimentSpilloverFactor()


class TestSentimentSpilloverMetadata:
    def test_name(self, factor):
        assert factor.name == "RNBR_TOV"

    def test_category(self, factor):
        assert factor.category == "高频成交分布"

    def test_repr(self, factor):
        assert "RNBR_TOV" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "RNBR_TOV"
        assert meta["category"] == "高频成交分布"


class TestSentimentSpilloverHandCalculated:
    def test_perfect_linear_relationship(self, factor):
        """turnover = 2 * nbr_turnover 时, 残差应为 0。"""
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A", "B", "C", "D"]
        nbr = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], index=dates, columns=stocks)
        tov = pd.DataFrame([[2.0, 4.0, 6.0, 8.0]], index=dates, columns=stocks)

        result = factor.compute(turnover=tov, nbr_turnover=nbr)
        # Perfect linear fit => residuals = 0
        for s in stocks:
            assert result.iloc[0][s] == pytest.approx(0.0, abs=1e-8)

    def test_residual_nonzero_for_outlier(self, factor):
        """一只股票偏离线性关系时, 其残差应非零。"""
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A", "B", "C", "D"]
        nbr = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], index=dates, columns=stocks)
        # A is an outlier: tov=10 instead of ~2
        tov = pd.DataFrame([[10.0, 4.0, 6.0, 8.0]], index=dates, columns=stocks)

        result = factor.compute(turnover=tov, nbr_turnover=nbr)
        # A should have positive residual (actual > fitted)
        assert result.iloc[0, 0] > 0

    def test_residuals_sum_to_zero(self, factor):
        """OLS 残差之和应为 0 (有截距项)。"""
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A", "B", "C", "D", "E"]
        np.random.seed(42)
        nbr = pd.DataFrame(
            np.random.rand(1, 5) * 0.1, index=dates, columns=stocks
        )
        tov = pd.DataFrame(
            np.random.rand(1, 5) * 0.1, index=dates, columns=stocks
        )

        result = factor.compute(turnover=tov, nbr_turnover=nbr)
        residual_sum = result.iloc[0].sum()
        assert residual_sum == pytest.approx(0.0, abs=1e-8)


class TestSentimentSpilloverEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A", "B", "C", "D"]
        nbr = pd.DataFrame([[1.0, np.nan, 3.0, 4.0]], index=dates, columns=stocks)
        tov = pd.DataFrame([[2.0, 4.0, 6.0, 8.0]], index=dates, columns=stocks)

        result = factor.compute(turnover=tov, nbr_turnover=nbr)
        assert isinstance(result, pd.DataFrame)
        # B should be NaN (excluded from regression)
        assert pd.isna(result.iloc[0, 1])

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A", "B", "C"]
        tov = pd.DataFrame(np.nan, index=dates, columns=stocks)
        nbr = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(turnover=tov, nbr_turnover=nbr)
        assert result.isna().all().all()

    def test_too_few_stocks(self, factor):
        """少于 3 只有效股票时, 该行应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A", "B"]
        nbr = pd.DataFrame([[1.0, 2.0]], index=dates, columns=stocks)
        tov = pd.DataFrame([[2.0, 4.0]], index=dates, columns=stocks)

        result = factor.compute(turnover=tov, nbr_turnover=nbr)
        assert result.isna().all().all()


class TestSentimentSpilloverOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A", "B", "C", "D"]
        np.random.seed(42)
        tov = pd.DataFrame(
            np.random.rand(5, 4) * 0.1, index=dates, columns=stocks
        )
        nbr = pd.DataFrame(
            np.random.rand(5, 4) * 0.1, index=dates, columns=stocks
        )

        result = factor.compute(turnover=tov, nbr_turnover=nbr)
        assert result.shape == tov.shape
        assert list(result.columns) == list(tov.columns)
        assert list(result.index) == list(tov.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A", "B", "C", "D"]
        np.random.seed(42)
        tov = pd.DataFrame(
            np.random.rand(3, 4) * 0.1, index=dates, columns=stocks
        )
        nbr = pd.DataFrame(
            np.random.rand(3, 4) * 0.1, index=dates, columns=stocks
        )

        result = factor.compute(turnover=tov, nbr_turnover=nbr)
        assert isinstance(result, pd.DataFrame)
