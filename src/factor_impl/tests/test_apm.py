import numpy as np
import pandas as pd
import pytest

from factors.apm import APMFactor


@pytest.fixture
def factor():
    return APMFactor()


class TestAPMMetadata:
    def test_name(self, factor):
        assert factor.name == "APM"

    def test_category(self, factor):
        assert factor.category == "高频因子-动量反转类"

    def test_repr(self, factor):
        assert "APM" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "APM"


class TestAPMCompute:
    def test_basic(self, factor):
        """基本计算不报错，有输出。"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A", "B", "C", "D", "E"]
        stock_am = pd.DataFrame(np.random.randn(25, 5) * 0.01, index=dates, columns=stocks)
        stock_pm = pd.DataFrame(np.random.randn(25, 5) * 0.01, index=dates, columns=stocks)
        idx_am = pd.DataFrame(np.random.randn(25) * 0.005, index=dates, columns=["index"])
        idx_pm = pd.DataFrame(np.random.randn(25) * 0.005, index=dates, columns=["index"])
        ret20 = pd.DataFrame(np.random.randn(25, 5) * 0.05, index=dates, columns=stocks)

        result = factor.compute(
            stock_am_ret=stock_am, stock_pm_ret=stock_pm,
            index_am_ret=idx_am, index_pm_ret=idx_pm,
            ret20=ret20, N=20,
        )
        assert result.shape == stock_am.shape
        assert isinstance(result, pd.DataFrame)
        # 至少最后几行有值
        assert result.iloc[-1].notna().any()

    def test_leading_nan(self, factor):
        """前 N-1 行应为 NaN。"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A", "B", "C", "D", "E"]
        stock_am = pd.DataFrame(np.random.randn(25, 5) * 0.01, index=dates, columns=stocks)
        stock_pm = pd.DataFrame(np.random.randn(25, 5) * 0.01, index=dates, columns=stocks)
        idx_am = pd.DataFrame(np.random.randn(25) * 0.005, index=dates, columns=["index"])
        idx_pm = pd.DataFrame(np.random.randn(25) * 0.005, index=dates, columns=["index"])
        ret20 = pd.DataFrame(np.random.randn(25, 5) * 0.05, index=dates, columns=stocks)

        result = factor.compute(
            stock_am_ret=stock_am, stock_pm_ret=stock_pm,
            index_am_ret=idx_am, index_pm_ret=idx_pm,
            ret20=ret20, N=20,
        )
        assert result.iloc[:19].isna().all().all()

    def test_residual_mean_near_zero(self, factor):
        """横截面回归残差均值应接近 0。"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A", "B", "C", "D", "E"]
        stock_am = pd.DataFrame(np.random.randn(25, 5) * 0.01, index=dates, columns=stocks)
        stock_pm = pd.DataFrame(np.random.randn(25, 5) * 0.01, index=dates, columns=stocks)
        idx_am = pd.DataFrame(np.random.randn(25) * 0.005, index=dates, columns=["index"])
        idx_pm = pd.DataFrame(np.random.randn(25) * 0.005, index=dates, columns=["index"])
        ret20 = pd.DataFrame(np.random.randn(25, 5) * 0.05, index=dates, columns=stocks)

        result = factor.compute(
            stock_am_ret=stock_am, stock_pm_ret=stock_pm,
            index_am_ret=idx_am, index_pm_ret=idx_pm,
            ret20=ret20, N=20,
        )
        last_row = result.iloc[-1].dropna()
        if len(last_row) > 0:
            assert abs(last_row.mean()) < 1.0
