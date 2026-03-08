import numpy as np
import pandas as pd
import pytest

from factors.drop_time_centroid import DropTimeCentroidFactor


@pytest.fixture
def factor():
    return DropTimeCentroidFactor()


class TestDropTimeCentroidMetadata:
    def test_name(self, factor):
        assert factor.name == "DROP_TIME_CENTROID"

    def test_category(self, factor):
        assert factor.category == "高频收益分布"

    def test_repr(self, factor):
        assert "DROP_TIME_CENTROID" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "DROP_TIME_CENTROID"
        assert meta["category"] == "高频收益分布"
        assert "时间重心" in meta["description"]


class TestDropTimeCentroidHandCalculated:
    def test_perfect_linear_relation(self, factor):
        """G_d = alpha + beta * G_u 完美线性关系时，残差应为 0。"""
        np.random.seed(42)
        n = 25
        ncols = 10
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        stocks = [f"S{i}" for i in range(ncols)]

        g_up = pd.DataFrame(
            np.random.rand(n, ncols) * 100 + 50, index=dates, columns=stocks
        )
        # G_d = 2 * G_u + 10 (完美线性)
        g_down = 2 * g_up + 10

        result = factor.compute(daily_g_up=g_up, daily_g_down=g_down, T=20)

        # 残差应接近 0，T 日均值也应接近 0
        last_row = result.iloc[-1]
        valid = last_row.notna()
        if valid.sum() > 0:
            np.testing.assert_array_almost_equal(
                last_row[valid].values, 0.0, decimal=8
            )

    def test_residual_nonzero(self, factor):
        """非线性关系时，残差应非零。"""
        np.random.seed(42)
        n = 25
        ncols = 10
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        stocks = [f"S{i}" for i in range(ncols)]

        g_up = pd.DataFrame(
            np.random.rand(n, ncols) * 100 + 50, index=dates, columns=stocks
        )
        g_down = pd.DataFrame(
            np.random.rand(n, ncols) * 100 + 50, index=dates, columns=stocks
        )

        result = factor.compute(daily_g_up=g_up, daily_g_down=g_down, T=20)

        last_row = result.iloc[-1]
        valid = last_row.notna()
        # 至少有一些非零残差
        if valid.sum() > 0:
            assert not np.allclose(last_row[valid].values, 0.0, atol=1e-10)

    def test_residuals_sum_to_zero(self, factor):
        """OLS 残差的截面均值应接近 0。"""
        np.random.seed(42)
        n = 25
        ncols = 20
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        stocks = [f"S{i}" for i in range(ncols)]

        g_up = pd.DataFrame(
            np.random.rand(n, ncols) * 100, index=dates, columns=stocks
        )
        g_down = pd.DataFrame(
            np.random.rand(n, ncols) * 100, index=dates, columns=stocks
        )

        result = factor.compute(daily_g_up=g_up, daily_g_down=g_down, T=20)

        # 每日残差截面均值应接近 0（OLS 性质）
        # 但因子是 T 日均值，所以检查最后一行的截面均值
        last_row = result.iloc[-1]
        valid = last_row.notna()
        if valid.sum() > 3:
            assert abs(last_row[valid].mean()) < 0.5


class TestDropTimeCentroidEdgeCases:
    def test_short_data(self, factor):
        """数据不足 T 时，结果应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A", "B", "C"]
        g_up = pd.DataFrame(np.random.rand(5, 3), index=dates, columns=stocks)
        g_down = pd.DataFrame(np.random.rand(5, 3), index=dates, columns=stocks)

        result = factor.compute(daily_g_up=g_up, daily_g_down=g_down, T=20)
        assert result.isna().all().all()

    def test_too_few_stocks(self, factor):
        """股票数不足 3 时，回归无法进行，结果为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        g_up = pd.DataFrame(np.random.rand(25), index=dates, columns=stocks)
        g_down = pd.DataFrame(np.random.rand(25), index=dates, columns=stocks)

        result = factor.compute(daily_g_up=g_up, daily_g_down=g_down, T=20)
        assert result.isna().all().all()

    def test_nan_in_input(self, factor):
        """输入含 NaN 时不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A", "B", "C", "D"]
        g_up = pd.DataFrame(np.random.rand(25, 4), index=dates, columns=stocks)
        g_up.iloc[15, 2] = np.nan
        g_down = pd.DataFrame(np.random.rand(25, 4), index=dates, columns=stocks)

        result = factor.compute(daily_g_up=g_up, daily_g_down=g_down, T=20)
        assert isinstance(result, pd.DataFrame)

    def test_all_nan(self, factor):
        """全 NaN 输入时，结果应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A", "B", "C"]
        g_up = pd.DataFrame(np.nan, index=dates, columns=stocks)
        g_down = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_g_up=g_up, daily_g_down=g_down, T=20)
        assert result.isna().all().all()


class TestDropTimeCentroidOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C", "D", "E"]
        g_up = pd.DataFrame(
            np.random.rand(30, 5) * 100, index=dates, columns=stocks
        )
        g_down = pd.DataFrame(
            np.random.rand(30, 5) * 100, index=dates, columns=stocks
        )

        result = factor.compute(daily_g_up=g_up, daily_g_down=g_down, T=20)
        assert result.shape == g_up.shape
        assert list(result.columns) == list(g_up.columns)
        assert list(result.index) == list(g_up.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A", "B", "C"]
        g_up = pd.DataFrame(np.random.rand(25, 3), index=dates, columns=stocks)
        g_down = pd.DataFrame(np.random.rand(25, 3), index=dates, columns=stocks)

        result = factor.compute(daily_g_up=g_up, daily_g_down=g_down, T=20)
        assert isinstance(result, pd.DataFrame)
