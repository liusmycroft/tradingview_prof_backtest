import numpy as np
import pandas as pd
import pytest

from factors.night_frost import NightFrostFactor


@pytest.fixture
def factor():
    return NightFrostFactor()


class TestNightFrostMetadata:
    def test_name(self, factor):
        assert factor.name == "NIGHT_FROST"

    def test_category(self, factor):
        assert factor.category == "高频量价相关性"

    def test_repr(self, factor):
        assert "NIGHT_FROST" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "NIGHT_FROST"
        assert meta["category"] == "高频量价相关性"


class TestNightFrostCompute:
    def test_identical_series(self, factor):
        """所有股票t-intercept序列相同时，相关系数绝对值均为1。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A", "B", "C"]
        vals = np.arange(20, dtype=float)
        daily_t = pd.DataFrame(
            {s: vals for s in stocks}, index=dates
        )

        result = factor.compute(daily_t_intercept=daily_t, T=20)
        # 最后一行应为1.0
        assert result.iloc[-1, 0] == pytest.approx(1.0, abs=1e-6)

    def test_output_range(self, factor):
        """相关系数绝对值均值应在 [0, 1]。"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C", "D"]
        daily_t = pd.DataFrame(
            np.random.randn(30, 4), index=dates, columns=stocks
        )

        result = factor.compute(daily_t_intercept=daily_t, T=20)
        valid = result.values[~np.isnan(result.values)]
        assert (valid >= -1e-10).all()
        assert (valid <= 1.0 + 1e-10).all()

    def test_two_stocks(self, factor):
        """两只股票时，因子值 = 两者相关系数绝对值。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        a = np.arange(20, dtype=float)
        b = -a  # 完全负相关
        daily_t = pd.DataFrame({"A": a, "B": b}, index=dates)

        result = factor.compute(daily_t_intercept=daily_t, T=20)
        # |corr(a, -a)| = 1.0
        assert result.iloc[-1, 0] == pytest.approx(1.0, abs=1e-6)
        assert result.iloc[-1, 1] == pytest.approx(1.0, abs=1e-6)


class TestNightFrostEdgeCases:
    def test_single_stock(self, factor):
        """单只股票时结果应为 NaN（无法计算截面相关性）。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        daily_t = pd.DataFrame({"A": np.arange(20, dtype=float)}, index=dates)

        result = factor.compute(daily_t_intercept=daily_t, T=20)
        assert result.isna().all().all()

    def test_nan_in_input(self, factor):
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        daily_t = pd.DataFrame(
            np.random.randn(30, 3), index=dates, columns=["A", "B", "C"]
        )
        daily_t.iloc[5, 0] = np.nan

        result = factor.compute(daily_t_intercept=daily_t, T=20)
        assert isinstance(result, pd.DataFrame)


class TestNightFrostOutputShape:
    def test_output_shape(self, factor):
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily_t = pd.DataFrame(
            np.random.randn(30, 3), index=dates, columns=stocks
        )

        result = factor.compute(daily_t_intercept=daily_t, T=20)
        assert result.shape == daily_t.shape

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        daily_t = pd.DataFrame({"A": np.arange(20.0), "B": np.arange(20.0)}, index=dates)

        result = factor.compute(daily_t_intercept=daily_t, T=20)
        assert isinstance(result, pd.DataFrame)
