import numpy as np
import pandas as pd
import pytest

from factors.peak_price_quantile import PeakPriceQuantileFactor


@pytest.fixture
def factor():
    return PeakPriceQuantileFactor()


class TestPeakPriceQuantileMetadata:
    def test_name(self, factor):
        assert factor.name == "PEAK_PRICE_QUANTILE"

    def test_category(self, factor):
        assert factor.category == "高频量价"

    def test_repr(self, factor):
        assert "PEAK_PRICE_QUANTILE" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "PEAK_PRICE_QUANTILE"
        assert meta["category"] == "高频量价"
        assert "量峰" in meta["description"]


class TestPeakPriceQuantileHandCalculated:
    def test_neutralization_removes_reversal(self, factor):
        """中性化后，因子与反转因子的截面相关性应接近 0。"""
        np.random.seed(42)
        n = 25
        ncols = 20
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        stocks = [f"S{i}" for i in range(ncols)]

        daily_pq = pd.DataFrame(
            np.random.rand(n, ncols), index=dates, columns=stocks
        )
        daily_ret = pd.DataFrame(
            np.random.randn(n, ncols) * 0.05, index=dates, columns=stocks
        )

        result = factor.compute(
            daily_peak_quantile=daily_pq, daily_ret_20=daily_ret, T=20,
        )

        # 最后一行应有值
        last_row = result.iloc[-1]
        valid = last_row.notna()
        if valid.sum() > 3:
            corr = np.corrcoef(last_row[valid].values, daily_ret.iloc[-1][valid].values)[0, 1]
            assert abs(corr) < 0.3  # 中性化后相关性应较低

    def test_constant_quantile(self, factor):
        """量峰分位点为常数时，中性化后残差应接近 0。"""
        n = 25
        ncols = 10
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        stocks = [f"S{i}" for i in range(ncols)]

        daily_pq = pd.DataFrame(0.5, index=dates, columns=stocks)
        daily_ret = pd.DataFrame(
            np.random.randn(n, ncols) * 0.05, index=dates, columns=stocks
        )

        result = factor.compute(
            daily_peak_quantile=daily_pq, daily_ret_20=daily_ret, T=20,
        )

        last_row = result.iloc[-1]
        valid = last_row.notna()
        if valid.sum() > 0:
            np.testing.assert_array_almost_equal(
                last_row[valid].values, 0.0, decimal=10
            )


class TestPeakPriceQuantileEdgeCases:
    def test_short_data(self, factor):
        """数据不足 T 时，结果应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A", "B", "C"]
        daily_pq = pd.DataFrame(np.random.rand(5, 3), index=dates, columns=stocks)
        daily_ret = pd.DataFrame(np.random.randn(5, 3) * 0.05, index=dates, columns=stocks)

        result = factor.compute(
            daily_peak_quantile=daily_pq, daily_ret_20=daily_ret, T=20,
        )
        assert result.isna().all().all()

    def test_too_few_stocks(self, factor):
        """股票数不足 3 时，回归无法进行，结果为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily_pq = pd.DataFrame(np.random.rand(25), index=dates, columns=stocks)
        daily_ret = pd.DataFrame(np.random.randn(25) * 0.05, index=dates, columns=stocks)

        result = factor.compute(
            daily_peak_quantile=daily_pq, daily_ret_20=daily_ret, T=20,
        )
        # 单只股票无法做截面回归
        assert result.isna().all().all()

    def test_nan_in_input(self, factor):
        """输入含 NaN 时不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A", "B", "C", "D"]
        daily_pq = pd.DataFrame(np.random.rand(25, 4), index=dates, columns=stocks)
        daily_pq.iloc[15, 2] = np.nan
        daily_ret = pd.DataFrame(np.random.randn(25, 4) * 0.05, index=dates, columns=stocks)

        result = factor.compute(
            daily_peak_quantile=daily_pq, daily_ret_20=daily_ret, T=20,
        )
        assert isinstance(result, pd.DataFrame)


class TestPeakPriceQuantileOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C", "D", "E"]
        daily_pq = pd.DataFrame(
            np.random.rand(30, 5), index=dates, columns=stocks
        )
        daily_ret = pd.DataFrame(
            np.random.randn(30, 5) * 0.05, index=dates, columns=stocks
        )

        result = factor.compute(
            daily_peak_quantile=daily_pq, daily_ret_20=daily_ret, T=20,
        )
        assert result.shape == daily_pq.shape
        assert list(result.columns) == list(daily_pq.columns)
        assert list(result.index) == list(daily_pq.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A", "B", "C"]
        daily_pq = pd.DataFrame(np.random.rand(25, 3), index=dates, columns=stocks)
        daily_ret = pd.DataFrame(np.random.randn(25, 3) * 0.05, index=dates, columns=stocks)

        result = factor.compute(
            daily_peak_quantile=daily_pq, daily_ret_20=daily_ret, T=20,
        )
        assert isinstance(result, pd.DataFrame)
