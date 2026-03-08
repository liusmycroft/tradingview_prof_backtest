import numpy as np
import pandas as pd
import pytest

from factors.els_correlation import ELSCorrelationFactor


@pytest.fixture
def factor():
    return ELSCorrelationFactor()


class TestELSCorrelationMetadata:
    def test_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "ELSCorrelation"
        assert meta["category"] == "资金流"
        assert meta["description"] != ""


class TestELSCorrelationCompute:
    def test_perfect_positive_correlation(self, factor):
        """EL 和 S 完全单调同向时，Spearman 相关系数应为 1。"""
        dates = pd.bdate_range("2025-01-01", periods=20)
        el = pd.DataFrame({"A": np.arange(1, 21, dtype=float)}, index=dates)
        s = pd.DataFrame({"A": np.arange(1, 21, dtype=float)}, index=dates)

        result = factor.compute(el, s, T=20)
        assert pytest.approx(result.iloc[-1, 0], abs=1e-10) == 1.0

    def test_perfect_negative_correlation(self, factor):
        """EL 和 S 完全单调反向时，Spearman 相关系数应为 -1。"""
        dates = pd.bdate_range("2025-01-01", periods=20)
        el = pd.DataFrame({"A": np.arange(1, 21, dtype=float)}, index=dates)
        s = pd.DataFrame({"A": np.arange(20, 0, -1, dtype=float)}, index=dates)

        result = factor.compute(el, s, T=20)
        assert pytest.approx(result.iloc[-1, 0], abs=1e-10) == -1.0

    def test_known_rank_correlation(self, factor):
        """用可手算的小窗口验证 Spearman 秩相关。"""
        # 5 个数据点: EL=[10,20,30,40,50], S=[50,40,10,20,30]
        # EL ranks: [1,2,3,4,5], S ranks: [5,4,1,2,3]
        # d = [-4,-2,2,2,2], d^2 = [16,4,4,4,4] = 32
        # rho = 1 - 6*32/(5*(25-1)) = 1 - 192/120 = 1 - 1.6 = -0.6
        dates = pd.bdate_range("2025-01-01", periods=5)
        el = pd.DataFrame({"A": [10.0, 20.0, 30.0, 40.0, 50.0]}, index=dates)
        s = pd.DataFrame({"A": [50.0, 40.0, 10.0, 20.0, 30.0]}, index=dates)

        result = factor.compute(el, s, T=5)
        assert pytest.approx(result.iloc[-1, 0], abs=1e-10) == -0.6

    def test_constant_el_returns_nan(self, factor):
        """EL 序列为常数时，秩相关无法定义，应返回 NaN。"""
        dates = pd.bdate_range("2025-01-01", periods=20)
        el = pd.DataFrame({"A": [5.0] * 20}, index=dates)
        s = pd.DataFrame({"A": np.arange(1, 21, dtype=float)}, index=dates)

        result = factor.compute(el, s, T=20)
        assert pd.isna(result.iloc[-1, 0])

    def test_constant_s_returns_nan(self, factor):
        """S 序列为常数时，秩相关无法定义，应返回 NaN。"""
        dates = pd.bdate_range("2025-01-01", periods=20)
        el = pd.DataFrame({"A": np.arange(1, 21, dtype=float)}, index=dates)
        s = pd.DataFrame({"A": [3.0] * 20}, index=dates)

        result = factor.compute(el, s, T=20)
        assert pd.isna(result.iloc[-1, 0])

    def test_nan_in_window_returns_nan(self, factor):
        """窗口内含 NaN 时应返回 NaN。"""
        dates = pd.bdate_range("2025-01-01", periods=20)
        el_vals = np.arange(1, 21, dtype=float)
        el_vals[10] = np.nan
        el = pd.DataFrame({"A": el_vals}, index=dates)
        s = pd.DataFrame({"A": np.arange(1, 21, dtype=float)}, index=dates)

        result = factor.compute(el, s, T=20)
        assert pd.isna(result.iloc[-1, 0])

    def test_insufficient_window_returns_nan(self, factor):
        """数据不足 T 天时应返回 NaN。"""
        dates = pd.bdate_range("2025-01-01", periods=10)
        el = pd.DataFrame({"A": np.arange(10, dtype=float)}, index=dates)
        s = pd.DataFrame({"A": np.arange(10, dtype=float)}, index=dates)

        result = factor.compute(el, s, T=20)
        assert result.isna().all().all()

    def test_output_shape_and_type(self, factor):
        """输出形状和类型应与输入一致。"""
        dates = pd.bdate_range("2025-01-01", periods=30)
        el = pd.DataFrame(
            {"A": np.random.rand(30), "B": np.random.rand(30)}, index=dates
        )
        s = pd.DataFrame(
            {"A": np.random.rand(30), "B": np.random.rand(30)}, index=dates
        )

        result = factor.compute(el, s, T=20)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == el.shape
        assert list(result.columns) == list(el.columns)
        assert (result.index == el.index).all()

    def test_multi_stock_independent(self, factor):
        """多只股票应独立计算。"""
        dates = pd.bdate_range("2025-01-01", periods=20)
        # S1: 完全正相关 -> 1.0, S2: 完全负相关 -> -1.0
        el = pd.DataFrame(
            {
                "S1": np.arange(1, 21, dtype=float),
                "S2": np.arange(1, 21, dtype=float),
            },
            index=dates,
        )
        s = pd.DataFrame(
            {
                "S1": np.arange(1, 21, dtype=float),
                "S2": np.arange(20, 0, -1, dtype=float),
            },
            index=dates,
        )

        result = factor.compute(el, s, T=20)
        assert pytest.approx(result.iloc[-1, 0], abs=1e-10) == 1.0
        assert pytest.approx(result.iloc[-1, 1], abs=1e-10) == -1.0

    def test_rolling_window(self, factor):
        """验证滚动窗口：不同日期应使用各自的窗口数据。"""
        dates = pd.bdate_range("2025-01-01", periods=6)
        # T=5, 第5行和第6行各自使用不同的5天窗口
        el = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]}, index=dates)
        s = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]}, index=dates)

        result = factor.compute(el, s, T=5)
        # 第5行 (index=4): 窗口 [1,2,3,4,5] vs [1,2,3,4,5] -> 1.0
        assert pytest.approx(result.iloc[4, 0], abs=1e-10) == 1.0
        # 第6行 (index=5): 窗口 [2,3,4,5,100] vs [2,3,4,5,100] -> 仍然 1.0
        assert pytest.approx(result.iloc[5, 0], abs=1e-10) == 1.0
