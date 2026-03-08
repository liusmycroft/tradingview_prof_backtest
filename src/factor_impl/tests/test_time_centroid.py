"""时间重心偏离因子测试"""

import numpy as np
import pandas as pd
import pytest
from factors.time_centroid import TimeCentroidFactor


@pytest.fixture
def factor():
    return TimeCentroidFactor()


@pytest.fixture
def sample_data():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=25)
    stocks = ["000001", "000002", "000003", "000004", "000005"]
    g_up = pd.DataFrame(
        np.random.uniform(0.3, 0.7, (25, 5)), index=dates, columns=stocks
    )
    g_down = pd.DataFrame(
        np.random.uniform(0.3, 0.7, (25, 5)), index=dates, columns=stocks
    )
    return g_up, g_down


class TestMetadata:
    def test_name(self, factor):
        assert factor.name == "TIME_CENTROID"

    def test_category(self, factor):
        assert factor.category == "高频收益分布"

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "TIME_CENTROID"

    def test_repr(self, factor):
        assert "TimeCentroidFactor" in repr(factor)


class TestCompute:
    def test_output_shape(self, factor, sample_data):
        g_up, g_down = sample_data
        result = factor.compute(g_up=g_up, g_down=g_down)
        assert result.shape == g_up.shape

    def test_perfect_linear_zero_residuals(self, factor):
        """当g_down = 2*g_up时，残差应为0"""
        dates = pd.date_range("2024-01-01", periods=25)
        stocks = ["A", "B", "C", "D", "E"]
        np.random.seed(42)
        g_up = pd.DataFrame(
            np.random.uniform(0.2, 0.8, (25, 5)), index=dates, columns=stocks
        )
        g_down = 2 * g_up  # 完美线性关系
        result = factor.compute(g_up=g_up, g_down=g_down, T=20)
        # 残差应接近0
        assert result.abs().max().max() < 1e-10

    def test_residuals_sum_to_zero_cross_section(self, factor):
        """每日截面回归残差之和应接近0"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=5)
        stocks = [f"S{i}" for i in range(10)]
        g_up = pd.DataFrame(
            np.random.uniform(0.2, 0.8, (5, 10)), index=dates, columns=stocks
        )
        g_down = pd.DataFrame(
            np.random.uniform(0.2, 0.8, (5, 10)), index=dates, columns=stocks
        )
        result = factor.compute(g_up=g_up, g_down=g_down, T=1)
        # T=1时，result就是残差本身，每行截面之和应接近0
        for date in dates:
            row_sum = result.loc[date].sum()
            assert abs(row_sum) < 1e-10

    def test_custom_window(self, factor, sample_data):
        g_up, g_down = sample_data
        result_5 = factor.compute(g_up=g_up, g_down=g_down, T=5)
        result_20 = factor.compute(g_up=g_up, g_down=g_down, T=20)
        # 不同窗口应产生不同结果
        assert not result_5.equals(result_20)

    def test_few_stocks_nan(self, factor):
        """股票数量不足3只时应为NaN"""
        dates = pd.date_range("2024-01-01", periods=5)
        stocks = ["A", "B"]
        g_up = pd.DataFrame(np.random.uniform(0.3, 0.7, (5, 2)), index=dates, columns=stocks)
        g_down = pd.DataFrame(np.random.uniform(0.3, 0.7, (5, 2)), index=dates, columns=stocks)
        result = factor.compute(g_up=g_up, g_down=g_down)
        # 只有2只股票，回归仍可进行（2个点可以拟合直线），残差为0
        # 但如果不足3只，我们的实现会跳过
        assert result.isna().all().all()
