import numpy as np
import pandas as pd
import pytest

from factors.morning_inflow_stability import MorningInflowStabilityFactor


@pytest.fixture
def factor():
    return MorningInflowStabilityFactor()


class TestMorningInflowStabilityMetadata:
    def test_name(self, factor):
        assert factor.name == "MORNING_INFLOW_STABILITY"

    def test_category(self, factor):
        assert factor.category == "量价改进"

    def test_repr(self, factor):
        assert "MORNING_INFLOW_STABILITY" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "MORNING_INFLOW_STABILITY"
        assert meta["category"] == "量价改进"
        assert "稳定" in meta["description"]


class TestMorningInflowStabilityCompute:
    """测试 compute 方法。"""

    def test_basic_known_values(self, factor):
        """用已知数据验证 mean/std 比值。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        # 使用简单数据: [1, 2, 3, 4, 5]
        data = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=dates)

        result = factor.compute(morning_net_inflow_rate=data, T=5)

        # T=5, 只有最后一行有值
        assert result.iloc[:4].isna().all().all()
        expected_mean = np.mean([1.0, 2.0, 3.0, 4.0, 5.0])
        expected_std = np.std([1.0, 2.0, 3.0, 4.0, 5.0], ddof=1)
        expected = expected_mean / expected_std
        assert result.iloc[4, 0] == pytest.approx(expected)

    def test_T3_rolling(self, factor):
        """T=3 滚动窗口验证。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        data = pd.DataFrame({"A": [1.0, 1.0, 1.0, 2.0, 3.0]}, index=dates)

        result = factor.compute(morning_net_inflow_rate=data, T=3)

        # 第 3 行 (idx=2): window=[1,1,1], mean=1, std=0 -> inf or nan
        # pandas std of constant = 0, so mean/0 = inf
        assert np.isinf(result.iloc[2, 0]) or np.isnan(result.iloc[2, 0])

        # 第 4 行 (idx=3): window=[1,1,2], mean=4/3, std=std([1,1,2], ddof=1)
        w = [1.0, 1.0, 2.0]
        expected = np.mean(w) / np.std(w, ddof=1)
        assert result.iloc[3, 0] == pytest.approx(expected)

    def test_negative_values(self, factor):
        """负值输入（净流出）的处理。"""
        dates = pd.bdate_range("2025-01-01", periods=4)
        data = pd.DataFrame({"A": [-0.01, -0.02, -0.01, -0.03]}, index=dates)

        result = factor.compute(morning_net_inflow_rate=data, T=3)

        w = [-0.01, -0.02, -0.01]
        expected = np.mean(w) / np.std(w, ddof=1)
        assert result.iloc[2, 0] == pytest.approx(expected)
        # 均值为负，结果应为负
        assert result.iloc[2, 0] < 0

    def test_multi_stock(self, factor):
        """多只股票同时计算。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        data = pd.DataFrame(
            {"A": [0.01, 0.02, 0.03, 0.04, 0.05],
             "B": [0.05, 0.04, 0.03, 0.02, 0.01]},
            index=dates,
        )

        result = factor.compute(morning_net_inflow_rate=data, T=3)

        assert result.shape == (5, 2)
        # A 递增, B 递减, 但 mean/std 的绝对值应相同（对称）
        # A window [0.01, 0.02, 0.03]: mean=0.02, std=0.01
        # B window [0.05, 0.04, 0.03]: mean=0.04, std=0.01
        assert result.iloc[2, 0] == pytest.approx(0.02 / 0.01)
        assert result.iloc[2, 1] == pytest.approx(0.04 / 0.01)


class TestMorningInflowStabilityEdgeCases:
    def test_nan_in_input(self, factor):
        """输入含 NaN 时不应抛异常。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        data = pd.DataFrame({"A": [0.01, np.nan, 0.03, 0.04, 0.05]}, index=dates)

        result = factor.compute(morning_net_inflow_rate=data, T=3)

        assert isinstance(result, pd.DataFrame)

    def test_zero_std(self, factor):
        """标准差为 0 时结果应为 inf 或 NaN。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        data = pd.DataFrame({"A": [0.01, 0.01, 0.01]}, index=dates)

        result = factor.compute(morning_net_inflow_rate=data, T=3)

        assert np.isinf(result.iloc[2, 0]) or np.isnan(result.iloc[2, 0])


class TestMorningInflowStabilityOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=30)
        stocks = ["A", "B", "C"]
        data = pd.DataFrame(
            np.random.randn(30, 3) * 0.02, index=dates, columns=stocks
        )

        result = factor.compute(morning_net_inflow_rate=data, T=20)

        assert result.shape == data.shape
        assert list(result.columns) == list(data.columns)

    def test_output_is_dataframe(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=5)
        data = pd.DataFrame({"A": [0.01, 0.02, 0.03, 0.04, 0.05]}, index=dates)

        result = factor.compute(morning_net_inflow_rate=data, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        """前 T-1 行应全为 NaN。"""
        dates = pd.bdate_range("2025-01-01", periods=25)
        data = pd.DataFrame(
            np.random.randn(25, 2) * 0.02, index=dates, columns=["A", "B"]
        )
        T = 20

        result = factor.compute(morning_net_inflow_rate=data, T=T)

        assert result.iloc[: T - 1].isna().all().all()
