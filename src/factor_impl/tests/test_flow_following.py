import numpy as np
import pandas as pd
import pytest

from factors.flow_following import FlowFollowingFactor


@pytest.fixture
def factor():
    return FlowFollowingFactor()


class TestFlowFollowingMetadata:
    def test_name(self, factor):
        assert factor.name == "FLOW_FOLLOWING"

    def test_category(self, factor):
        assert factor.category == "高频量价"

    def test_repr(self, factor):
        assert "FLOW_FOLLOWING" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "FLOW_FOLLOWING"
        assert meta["category"] == "高频量价"


class TestFlowFollowingHandCalculated:
    def test_two_stocks_T3(self, factor):
        """两只股票 T=3，手算 |Spearman corr|。"""
        from scipy import stats

        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A", "B"]
        data = np.array([[0.01, 0.02],
                         [0.03, 0.04],
                         [0.02, 0.01]])
        hld = pd.DataFrame(data, index=dates, columns=stocks)

        result = factor.compute(high_low_diff=hld, T=3)

        corr_ab, _ = stats.spearmanr(data[:, 0], data[:, 1])
        expected = abs(corr_ab)

        assert result.iloc[2, 0] == pytest.approx(expected, rel=1e-6)
        assert result.iloc[2, 1] == pytest.approx(expected, rel=1e-6)

    def test_three_stocks_T3(self, factor):
        """三只股票 T=3，验证平均 |Spearman corr|。"""
        from scipy import stats

        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A", "B", "C"]
        data = np.array([[0.01, 0.03, 0.02],
                         [0.02, 0.01, 0.04],
                         [0.03, 0.02, 0.01]])
        hld = pd.DataFrame(data, index=dates, columns=stocks)

        result = factor.compute(high_low_diff=hld, T=3)

        # A 的因子值 = mean(|corr(A,B)|, |corr(A,C)|)
        corr_ab, _ = stats.spearmanr(data[:, 0], data[:, 1])
        corr_ac, _ = stats.spearmanr(data[:, 0], data[:, 2])
        expected_a = (abs(corr_ab) + abs(corr_ac)) / 2.0

        assert result.iloc[2, 0] == pytest.approx(expected_a, rel=1e-6)


class TestFlowFollowingEdgeCases:
    def test_nan_in_data(self, factor):
        """含 NaN 的股票应被跳过。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A", "B"]
        data = np.array([[0.01, np.nan],
                         [0.02, 0.01],
                         [0.03, 0.02]])
        hld = pd.DataFrame(data, index=dates, columns=stocks)

        result = factor.compute(high_low_diff=hld, T=3)
        # A 无法与 B 计算 (B 有 NaN)，所以 A 也为 NaN
        assert np.isnan(result.iloc[2, 0])
        assert np.isnan(result.iloc[2, 1])

    def test_single_stock(self, factor):
        """只有一只股票时，无法计算相关系数，结果为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        hld = pd.DataFrame(
            np.random.normal(0, 0.01, (5, 1)), index=dates, columns=stocks
        )
        result = factor.compute(high_low_diff=hld, T=5)
        assert np.isnan(result.iloc[4, 0])

    def test_constant_series(self, factor):
        """常数序列的 std=0，应跳过。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A", "B"]
        data = np.array([[0.01, 0.01],
                         [0.01, 0.02],
                         [0.01, 0.03]])
        hld = pd.DataFrame(data, index=dates, columns=stocks)

        result = factor.compute(high_low_diff=hld, T=3)
        # A 是常数序列，std=0，应被跳过
        assert np.isnan(result.iloc[2, 0])

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        hld = pd.DataFrame(
            np.random.normal(0, 0.01, (30, 3)), index=dates, columns=stocks
        )
        result = factor.compute(high_low_diff=hld, T=20)
        assert result.shape == hld.shape
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_nan(self, factor):
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B", "C"]
        hld = pd.DataFrame(
            np.random.normal(0, 0.01, (10, 3)), index=dates, columns=stocks
        )
        T = 5
        result = factor.compute(high_low_diff=hld, T=T)
        assert result.iloc[: T - 1].isna().all().all()
