import numpy as np
import pandas as pd
import pytest

from factors.net_commission_buy import NetCommissionBuyFactor


@pytest.fixture
def factor():
    return NetCommissionBuyFactor()


class TestNetCommissionBuyMetadata:
    def test_name(self, factor):
        assert factor.name == "NET_COMMISSION_BUY"

    def test_category(self, factor):
        assert factor.category == "高频资金流"

    def test_repr(self, factor):
        assert "NET_COMMISSION_BUY" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "NET_COMMISSION_BUY"
        assert meta["category"] == "高频资金流"


class TestNetCommissionBuyHandCalculated:
    """用手算数据验证净委买变化率因子。"""

    def test_constant_inputs(self, factor):
        """常数输入: net_change=1000, float_shares=1e8 => rate=1e-5, 均值=1e-5。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        net_change = pd.DataFrame(1000.0, index=dates, columns=stocks)
        float_shares = pd.DataFrame(1e8, index=dates, columns=stocks)

        result = factor.compute(
            net_commission_change=net_change, float_shares=float_shares, T=3
        )

        for i in range(5):
            assert result.iloc[i, 0] == pytest.approx(1e-5, rel=1e-10)

    def test_varying_values_T3(self, factor):
        """T=3, 变化的净委买量。

        daily_rate = [0.1, 0.2, 0.3, 0.4]
        rolling(3, min_periods=1):
          day0: mean([0.1]) = 0.1
          day1: mean([0.1, 0.2]) = 0.15
          day2: mean([0.1, 0.2, 0.3]) = 0.2
          day3: mean([0.2, 0.3, 0.4]) = 0.3
        """
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        stocks = ["A"]

        net_change = pd.DataFrame(
            [100.0, 200.0, 300.0, 400.0], index=dates, columns=stocks
        )
        float_shares = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(
            net_commission_change=net_change, float_shares=float_shares, T=3
        )

        assert result.iloc[0, 0] == pytest.approx(0.1, rel=1e-10)
        assert result.iloc[1, 0] == pytest.approx(0.15, rel=1e-10)
        assert result.iloc[2, 0] == pytest.approx(0.2, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(0.3, rel=1e-10)

    def test_two_stocks(self, factor):
        """两只股票并行计算。

        Stock A: net_change=[100, 200], float_shares=1000
          daily_rate = [0.1, 0.2], rolling(2, min_periods=1): [0.1, 0.15]

        Stock B: net_change=[10, 20], float_shares=100
          daily_rate = [0.1, 0.2], rolling(2, min_periods=1): [0.1, 0.15]
        """
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A", "B"]

        net_change = pd.DataFrame(
            {"A": [100.0, 200.0, 300.0], "B": [10.0, 20.0, 30.0]}, index=dates
        )
        float_shares = pd.DataFrame(
            {"A": [1000.0] * 3, "B": [100.0] * 3}, index=dates
        )

        result = factor.compute(
            net_commission_change=net_change, float_shares=float_shares, T=2
        )

        # A: [0.1, 0.2, 0.3] -> rolling(2): [0.1, 0.15, 0.25]
        # B: [0.1, 0.2, 0.3] -> rolling(2): [0.1, 0.15, 0.25]
        np.testing.assert_allclose(result["A"].values, [0.1, 0.15, 0.25], atol=1e-12)
        np.testing.assert_allclose(result["B"].values, [0.1, 0.15, 0.25], atol=1e-12)

    def test_negative_net_change(self, factor):
        """净委买为负（卖方压力大于买方）。

        daily_rate = [-0.1, -0.2, -0.3]
        rolling(2, min_periods=1): [-0.1, -0.15, -0.25]
        """
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        net_change = pd.DataFrame([-100.0, -200.0, -300.0], index=dates, columns=stocks)
        float_shares = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(
            net_commission_change=net_change, float_shares=float_shares, T=2
        )

        np.testing.assert_allclose(
            result["A"].values, [-0.1, -0.15, -0.25], atol=1e-12
        )


class TestNetCommissionBuyEdgeCases:
    def test_zero_float_shares(self, factor):
        """流通股本为零时, 除法产生 inf, 不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        net_change = pd.DataFrame(1000.0, index=dates, columns=stocks)
        float_shares = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(
            net_commission_change=net_change, float_shares=float_shares, T=2
        )
        assert result.shape == (3, 1)
        assert np.all(np.isinf(result.values) | np.isnan(result.values))

    def test_nan_propagation(self, factor):
        """输入含 NaN 时, rolling mean(min_periods=1) 会跳过 NaN。"""
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        stocks = ["A"]

        net_change = pd.DataFrame(
            [100.0, np.nan, 100.0, 100.0], index=dates, columns=stocks
        )
        float_shares = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(
            net_commission_change=net_change, float_shares=float_shares, T=2
        )

        # daily_rate: [0.1, NaN, 0.1, 0.1]
        # rolling(2, min_periods=1): [0.1, 0.1, 0.1, 0.1]
        np.testing.assert_allclose(result["A"].values, [0.1, 0.1, 0.1, 0.1], atol=1e-12)

    def test_t_equals_one(self, factor):
        """T=1 时, 结果等于日度变化率本身。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        net_change = pd.DataFrame([100.0, 200.0, 300.0], index=dates, columns=stocks)
        float_shares = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(
            net_commission_change=net_change, float_shares=float_shares, T=1
        )

        np.testing.assert_allclose(result["A"].values, [0.1, 0.2, 0.3], atol=1e-12)

    def test_zero_net_change(self, factor):
        """净委买为 0 时, 因子值应为 0。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        net_change = pd.DataFrame(0.0, index=dates, columns=stocks)
        float_shares = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(
            net_commission_change=net_change, float_shares=float_shares, T=3
        )
        assert result.iloc[2, 0] == pytest.approx(0.0, abs=1e-15)


class TestNetCommissionBuyOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        stocks = ["A", "B", "C"]

        net_change = pd.DataFrame(
            np.random.randn(50, 3) * 1000, index=dates, columns=stocks
        )
        float_shares = pd.DataFrame(
            np.random.uniform(1e6, 1e8, (50, 3)), index=dates, columns=stocks
        )

        result = factor.compute(
            net_commission_change=net_change, float_shares=float_shares, T=20
        )

        assert result.shape == net_change.shape
        assert list(result.columns) == list(net_change.columns)
        assert list(result.index) == list(net_change.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        net_change = pd.DataFrame(100.0, index=dates, columns=stocks)
        float_shares = pd.DataFrame(1e8, index=dates, columns=stocks)

        result = factor.compute(
            net_commission_change=net_change, float_shares=float_shares, T=3
        )
        assert isinstance(result, pd.DataFrame)

    def test_min_periods_1_all_rows_have_values(self, factor):
        """min_periods=1, 所以从第一行起就有值。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]

        net_change = pd.DataFrame(
            np.random.randn(10, 2) * 1000, index=dates, columns=stocks
        )
        float_shares = pd.DataFrame(1e8, index=dates, columns=stocks)

        result = factor.compute(
            net_commission_change=net_change, float_shares=float_shares, T=5
        )

        assert result.notna().all().all()
