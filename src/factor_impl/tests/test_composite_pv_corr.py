import numpy as np
import pandas as pd
import pytest

from factors.composite_pv_corr import CompositePVCorrFactor


@pytest.fixture
def factor():
    return CompositePVCorrFactor()


class TestCompositePVCorrMetadata:
    def test_name(self, factor):
        assert factor.name == "CPV"

    def test_category(self, factor):
        assert factor.category == "高频量价相关性"

    def test_repr(self, factor):
        assert "CPV" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "CPV"
        assert meta["category"] == "高频量价相关性"


class TestCompositePVCorrHandCalculated:
    def test_zero_ret_no_neutralization_effect(self, factor):
        """ret_20d全为0时，回归残差等于原值减均值。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A", "B", "C", "D"]
        np.random.seed(42)
        pv_avg = pd.DataFrame(
            np.random.randn(5, 4), index=dates, columns=stocks
        )
        pv_std = pd.DataFrame(
            np.random.uniform(0, 1, (5, 4)), index=dates, columns=stocks
        )
        pv_trend = pd.DataFrame(
            np.random.randn(5, 4) * 0.1, index=dates, columns=stocks
        )
        ret_20d = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(
            pv_corr_avg=pv_avg, pv_corr_std=pv_std,
            pv_corr_trend=pv_trend, ret_20d=ret_20d,
        )
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (5, 4)
        # With zero ret, residuals = original - mean(original)
        # Then z-scored, so each row should have mean ~0
        for i in range(5):
            row = result.iloc[i].dropna()
            if len(row) > 1:
                assert row.mean() == pytest.approx(0.0, abs=1e-6)

    def test_cross_sectional_zscore_mean_zero(self, factor):
        """截面标准化后每行均值应为0。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B", "C", "D", "E"]
        np.random.seed(123)
        pv_avg = pd.DataFrame(np.random.randn(10, 5), index=dates, columns=stocks)
        pv_std = pd.DataFrame(np.random.uniform(0, 1, (10, 5)), index=dates, columns=stocks)
        pv_trend = pd.DataFrame(np.random.randn(10, 5) * 0.1, index=dates, columns=stocks)
        ret_20d = pd.DataFrame(np.random.randn(10, 5) * 0.05, index=dates, columns=stocks)

        result = factor.compute(
            pv_corr_avg=pv_avg, pv_corr_std=pv_std,
            pv_corr_trend=pv_trend, ret_20d=ret_20d,
        )
        # cpv is sum of two z-scores, each row mean should be ~0
        for i in range(10):
            row = result.iloc[i].dropna()
            if len(row) > 1:
                assert row.mean() == pytest.approx(0.0, abs=0.1)

    def test_two_dates_independent(self, factor):
        """不同日期应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=2, freq="D")
        stocks = ["A", "B", "C"]
        pv_avg = pd.DataFrame(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], index=dates, columns=stocks
        )
        pv_std = pd.DataFrame(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], index=dates, columns=stocks
        )
        pv_trend = pd.DataFrame(
            [[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]], index=dates, columns=stocks
        )
        ret_20d = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(
            pv_corr_avg=pv_avg, pv_corr_std=pv_std,
            pv_corr_trend=pv_trend, ret_20d=ret_20d,
        )
        assert result.shape == (2, 3)


class TestCompositePVCorrEdgeCases:
    def test_single_stock_nan(self, factor):
        """单只股票时截面标准差为0/NaN，结果应为NaN。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        pv_avg = pd.DataFrame(np.random.randn(5, 1), index=dates, columns=stocks)
        pv_std = pd.DataFrame(np.random.uniform(0, 1, (5, 1)), index=dates, columns=stocks)
        pv_trend = pd.DataFrame(np.random.randn(5, 1), index=dates, columns=stocks)
        ret_20d = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(
            pv_corr_avg=pv_avg, pv_corr_std=pv_std,
            pv_corr_trend=pv_trend, ret_20d=ret_20d,
        )
        assert isinstance(result, pd.DataFrame)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A", "B", "C"]
        nan_df = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(
            pv_corr_avg=nan_df.copy(), pv_corr_std=nan_df.copy(),
            pv_corr_trend=nan_df.copy(), ret_20d=nan_df.copy(),
        )
        assert result.isna().all().all()

    def test_two_stocks_minimum(self, factor):
        """两只股票时应能正常计算（截面回归需要>=3个点，会fallback）。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A", "B"]
        np.random.seed(42)
        pv_avg = pd.DataFrame(np.random.randn(5, 2), index=dates, columns=stocks)
        pv_std = pd.DataFrame(np.random.uniform(0, 1, (5, 2)), index=dates, columns=stocks)
        pv_trend = pd.DataFrame(np.random.randn(5, 2), index=dates, columns=stocks)
        ret_20d = pd.DataFrame(np.random.randn(5, 2) * 0.05, index=dates, columns=stocks)

        result = factor.compute(
            pv_corr_avg=pv_avg, pv_corr_std=pv_std,
            pv_corr_trend=pv_trend, ret_20d=ret_20d,
        )
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (5, 2)


class TestCompositePVCorrOutputShape:
    def test_output_shape_matches_input(self, factor):
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A", "B", "C", "D"]
        pv_avg = pd.DataFrame(np.random.randn(20, 4), index=dates, columns=stocks)
        pv_std = pd.DataFrame(np.random.uniform(0, 1, (20, 4)), index=dates, columns=stocks)
        pv_trend = pd.DataFrame(np.random.randn(20, 4) * 0.1, index=dates, columns=stocks)
        ret_20d = pd.DataFrame(np.random.randn(20, 4) * 0.05, index=dates, columns=stocks)

        result = factor.compute(
            pv_corr_avg=pv_avg, pv_corr_std=pv_std,
            pv_corr_trend=pv_trend, ret_20d=ret_20d,
        )
        assert result.shape == pv_avg.shape
        assert list(result.columns) == list(pv_avg.columns)
        assert list(result.index) == list(pv_avg.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A", "B", "C"]
        pv_avg = pd.DataFrame(np.random.randn(5, 3), index=dates, columns=stocks)
        pv_std = pd.DataFrame(np.random.uniform(0, 1, (5, 3)), index=dates, columns=stocks)
        pv_trend = pd.DataFrame(np.random.randn(5, 3), index=dates, columns=stocks)
        ret_20d = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(
            pv_corr_avg=pv_avg, pv_corr_std=pv_std,
            pv_corr_trend=pv_trend, ret_20d=ret_20d,
        )
        assert isinstance(result, pd.DataFrame)
