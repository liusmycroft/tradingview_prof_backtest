import numpy as np
import pandas as pd
import pytest

from factors.overnight_ret_iv import OvernightRetIVFactor


@pytest.fixture
def factor():
    return OvernightRetIVFactor()


class TestOvernightRetIVMetadata:
    def test_name(self, factor):
        assert factor.name == "OvernightRetIV"

    def test_category(self, factor):
        assert factor.category == "高频波动跳跃"

    def test_repr(self, factor):
        assert "OvernightRetIV" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "OvernightRetIV"
        assert meta["category"] == "高频波动跳跃"


class TestOvernightRetIVHandCalculated:
    def test_pure_noise_has_positive_iv(self, factor):
        """纯噪声（无共同因子）时，特质波动率应为正。"""
        np.random.seed(42)
        T = 20
        n_stocks = 10
        dates = pd.date_range("2024-01-01", periods=T, freq="D")
        stocks = [f"S{i}" for i in range(n_stocks)]

        overnight_ret = pd.DataFrame(
            np.random.randn(T, n_stocks) * 0.01,
            index=dates, columns=stocks,
        )

        result = factor.compute(overnight_ret=overnight_ret, T=T)
        assert result.iloc[-1].notna().any()
        assert (result.iloc[-1].dropna() > 0).all()

    def test_perfect_factor_has_zero_iv(self, factor):
        """所有股票收益完全由单一因子解释时，残差应接近零。"""
        T = 20
        n_stocks = 5
        dates = pd.date_range("2024-01-01", periods=T, freq="D")
        stocks = [f"S{i}" for i in range(n_stocks)]

        np.random.seed(123)
        F = np.random.randn(T) * 0.01
        betas = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        data = np.outer(F, betas)

        overnight_ret = pd.DataFrame(data, index=dates, columns=stocks)

        result = factor.compute(overnight_ret=overnight_ret, T=T)
        valid = result.iloc[-1].dropna()
        assert len(valid) > 0
        assert (valid < 1e-10).all()

    def test_multi_stock_independent_windows(self, factor):
        """不同窗口应独立计算。"""
        np.random.seed(7)
        T = 5
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = [f"S{i}" for i in range(5)]

        overnight_ret = pd.DataFrame(
            np.random.randn(10, 5) * 0.01,
            index=dates, columns=stocks,
        )

        result = factor.compute(overnight_ret=overnight_ret, T=T)
        assert result.iloc[T - 1].notna().any()
        assert result.iloc[: T - 1].isna().all().all()


class TestOvernightRetIVEdgeCases:
    def test_too_few_stocks(self, factor):
        """股票数不足3时，应返回 NaN。"""
        T = 5
        dates = pd.date_range("2024-01-01", periods=T, freq="D")
        stocks = ["A", "B"]

        overnight_ret = pd.DataFrame(
            np.random.randn(T, 2) * 0.01, index=dates, columns=stocks,
        )

        result = factor.compute(overnight_ret=overnight_ret, T=T)
        assert result.isna().all().all()

    def test_nan_in_input(self, factor):
        """输入含 NaN 时, 不应抛异常。"""
        np.random.seed(42)
        T = 5
        dates = pd.date_range("2024-01-01", periods=T, freq="D")
        stocks = [f"S{i}" for i in range(5)]

        overnight_ret = pd.DataFrame(
            np.random.randn(T, 5) * 0.01, index=dates, columns=stocks,
        )
        overnight_ret.iloc[2, 1] = np.nan

        result = factor.compute(overnight_ret=overnight_ret, T=T)
        assert isinstance(result, pd.DataFrame)

    def test_all_nan(self, factor):
        """全 NaN 输入时, 结果应全为 NaN。"""
        T = 5
        dates = pd.date_range("2024-01-01", periods=T, freq="D")
        stocks = ["A", "B", "C"]

        overnight_ret = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(overnight_ret=overnight_ret, T=T)
        assert result.isna().all().all()

    def test_constant_returns(self, factor):
        """所有收益率为常数时，去均值后为零矩阵，SVD 方差为零，应返回 NaN。"""
        T = 5
        dates = pd.date_range("2024-01-01", periods=T, freq="D")
        stocks = [f"S{i}" for i in range(5)]

        overnight_ret = pd.DataFrame(0.01, index=dates, columns=stocks)

        result = factor.compute(overnight_ret=overnight_ret, T=T)
        assert result.isna().all().all()

    def test_insufficient_data(self, factor):
        """数据不足 T 天时, 全部为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = [f"S{i}" for i in range(5)]

        overnight_ret = pd.DataFrame(
            np.random.randn(5, 5) * 0.01, index=dates, columns=stocks,
        )

        result = factor.compute(overnight_ret=overnight_ret, T=20)
        assert result.isna().all().all()


class TestOvernightRetIVOutputShape:
    def test_output_shape_matches_input(self, factor):
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A", "B", "C", "D", "E"]

        overnight_ret = pd.DataFrame(
            np.random.randn(25, 5) * 0.01, index=dates, columns=stocks,
        )

        result = factor.compute(overnight_ret=overnight_ret, T=20)
        assert result.shape == overnight_ret.shape
        assert list(result.columns) == list(overnight_ret.columns)
        assert list(result.index) == list(overnight_ret.index)

    def test_output_is_dataframe(self, factor):
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = [f"S{i}" for i in range(5)]

        overnight_ret = pd.DataFrame(
            np.random.randn(10, 5) * 0.01, index=dates, columns=stocks,
        )

        result = factor.compute(overnight_ret=overnight_ret, T=5)
        assert isinstance(result, pd.DataFrame)
