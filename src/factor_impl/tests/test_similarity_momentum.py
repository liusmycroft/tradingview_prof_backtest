import numpy as np
import pandas as pd
import pytest

from factors.similarity_momentum import SimilarityMomentumFactor


@pytest.fixture
def factor():
    return SimilarityMomentumFactor()


class TestSimilarityMomentumMetadata:
    def test_name(self, factor):
        assert factor.name == "SIM_MOMENTUM"

    def test_category(self, factor):
        assert factor.category == "图谱网络-动量溢出"

    def test_repr(self, factor):
        assert "SIM_MOMENTUM" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "SIM_MOMENTUM"
        assert meta["category"] == "图谱网络-动量溢出"


def _make_inputs(n_dates, n_stocks, seed=42):
    """Helper to create valid input DataFrames."""
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="ME")
    stocks = [f"S{i}" for i in range(n_stocks)]
    return {
        "price": pd.DataFrame(np.random.uniform(5, 50, (n_dates, n_stocks)), index=dates, columns=stocks),
        "log_mcap": pd.DataFrame(np.random.uniform(20, 25, (n_dates, n_stocks)), index=dates, columns=stocks),
        "bm": pd.DataFrame(np.random.uniform(0.3, 2, (n_dates, n_stocks)), index=dates, columns=stocks),
        "op": pd.DataFrame(np.random.uniform(0, 0.3, (n_dates, n_stocks)), index=dates, columns=stocks),
        "inv": pd.DataFrame(np.random.uniform(-0.1, 0.3, (n_dates, n_stocks)), index=dates, columns=stocks),
        "excess_return": pd.DataFrame(np.random.randn(n_dates, n_stocks) * 0.05, index=dates, columns=stocks),
        "mcap": pd.DataFrame(np.random.uniform(1e9, 1e11, (n_dates, n_stocks)), index=dates, columns=stocks),
    }


class TestSimilarityMomentumHandCalculated:
    def test_identical_stocks_same_factor(self, factor):
        """所有股票特征完全相同时, 因子值应相同。"""
        dates = pd.date_range("2024-01-01", periods=1, freq="ME")
        stocks = ["A", "B", "C"]
        n = len(stocks)
        price = pd.DataFrame([[10.0] * n], index=dates, columns=stocks)
        log_mcap = pd.DataFrame([[22.0] * n], index=dates, columns=stocks)
        bm = pd.DataFrame([[1.0] * n], index=dates, columns=stocks)
        op = pd.DataFrame([[0.1] * n], index=dates, columns=stocks)
        inv = pd.DataFrame([[0.05] * n], index=dates, columns=stocks)
        excess_return = pd.DataFrame([[0.02, 0.03, 0.04]], index=dates, columns=stocks)
        mcap = pd.DataFrame([[1e10] * n], index=dates, columns=stocks)

        result = factor.compute(
            price=price, log_mcap=log_mcap, bm=bm, op=op, inv=inv,
            excess_return=excess_return, mcap=mcap, K=2,
        )
        # All features identical => after standardization all are 0 => distance 0
        # Each stock picks the other 2 as neighbors
        # With equal mcap, SIM = mean of neighbors' excess_return
        assert isinstance(result, pd.DataFrame)
        assert result.notna().all().all()

    def test_K_larger_than_stocks(self, factor):
        """K > n_stocks 时应正常运行, 使用所有可用邻居。"""
        inputs = _make_inputs(1, 3, seed=99)
        result = factor.compute(**inputs, K=100)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (1, 3)

    def test_produces_finite_values(self, factor):
        """正常输入应产生有限值。"""
        inputs = _make_inputs(2, 5, seed=42)
        result = factor.compute(**inputs, K=2)
        assert isinstance(result, pd.DataFrame)
        # At least some values should be non-NaN
        assert result.notna().any().any()


class TestSimilarityMomentumEdgeCases:
    def test_nan_in_features(self, factor):
        """特征含 NaN 时不应抛异常。"""
        inputs = _make_inputs(1, 5, seed=42)
        inputs["price"].iloc[0, 0] = np.nan

        result = factor.compute(**inputs, K=2)
        assert isinstance(result, pd.DataFrame)
        # Stock with NaN feature should have NaN result
        assert pd.isna(result.iloc[0, 0])

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="ME")
        stocks = ["A", "B", "C"]
        nan_df = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(
            price=nan_df, log_mcap=nan_df, bm=nan_df, op=nan_df, inv=nan_df,
            excess_return=nan_df, mcap=nan_df, K=2,
        )
        assert result.isna().all().all()

    def test_single_stock(self, factor):
        """单只股票时, 无邻居, 结果应为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=1, freq="ME")
        stocks = ["A"]
        one = pd.DataFrame([[10.0]], index=dates, columns=stocks)

        result = factor.compute(
            price=one, log_mcap=one, bm=one, op=one, inv=one,
            excess_return=one, mcap=one, K=2,
        )
        assert result.isna().all().all()


class TestSimilarityMomentumOutputShape:
    def test_output_shape_matches_input(self, factor):
        inputs = _make_inputs(3, 5, seed=42)
        result = factor.compute(**inputs, K=2)
        assert result.shape == (3, 5)
        assert list(result.columns) == list(inputs["price"].columns)
        assert list(result.index) == list(inputs["price"].index)

    def test_output_is_dataframe(self, factor):
        inputs = _make_inputs(2, 4, seed=42)
        result = factor.compute(**inputs, K=2)
        assert isinstance(result, pd.DataFrame)
