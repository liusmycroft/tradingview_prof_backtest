import numpy as np
import pandas as pd
import pytest

from factors.opening_commission_ratio import OpeningCommissionRatioFactor


@pytest.fixture
def factor():
    return OpeningCommissionRatioFactor()


class TestOpeningCommissionRatioMetadata:
    def test_name(self, factor):
        assert factor.name == "OPENING_COMMISSION_RATIO"

    def test_category(self, factor):
        assert factor.category == "高频资金流"

    def test_repr(self, factor):
        assert "OPENING_COMMISSION_RATIO" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "OPENING_COMMISSION_RATIO"
        assert meta["category"] == "高频资金流"


class TestOpeningCommissionRatioHandCalculated:
    """用手算数据验证开盘后净委买增额占比因子。"""

    def test_constant_ratio(self, factor):
        """常数: net_commission_increase=100, amount=1000 => ratio=0.1。

        min_periods=1, 所以所有行都有值。
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        net_comm = pd.DataFrame(100.0, index=dates, columns=stocks)
        amount = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(
            net_commission_increase=net_comm, amount=amount, T=3
        )

        for i in range(5):
            assert result.iloc[i, 0] == pytest.approx(0.1, rel=1e-10)

    def test_varying_values_T3(self, factor):
        """T=3, 变化的净委买增额。

        daily_ratio = [0.1, 0.2, -0.05]
        rolling(3, min_periods=1):
          day0: mean([0.1]) = 0.1
          day1: mean([0.1, 0.2]) = 0.15
          day2: mean([0.1, 0.2, -0.05]) = 0.25/3
        """
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        net_comm = pd.DataFrame([100.0, 200.0, -50.0], index=dates, columns=stocks)
        amount = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(
            net_commission_increase=net_comm, amount=amount, T=3
        )

        assert result.iloc[0, 0] == pytest.approx(0.1, rel=1e-10)
        assert result.iloc[1, 0] == pytest.approx(0.15, rel=1e-10)
        assert result.iloc[2, 0] == pytest.approx(0.25 / 3, rel=1e-10)

    def test_two_stocks(self, factor):
        """两只股票并行计算。

        Stock A: ratio = [0.1, 0.2] -> rolling(2): [0.1, 0.15]
        Stock B: ratio = [-0.1, -0.2] -> rolling(2): [-0.1, -0.15]
        """
        dates = pd.date_range("2024-01-01", periods=2, freq="D")
        stocks = ["A", "B"]

        net_comm = pd.DataFrame(
            [[100.0, -100.0], [200.0, -200.0]], index=dates, columns=stocks
        )
        amount = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(
            net_commission_increase=net_comm, amount=amount, T=2
        )

        assert result.loc[dates[0], "A"] == pytest.approx(0.1, rel=1e-10)
        assert result.loc[dates[1], "A"] == pytest.approx(0.15, rel=1e-10)
        assert result.loc[dates[0], "B"] == pytest.approx(-0.1, rel=1e-10)
        assert result.loc[dates[1], "B"] == pytest.approx(-0.15, rel=1e-10)

    def test_rolling_window_slides(self, factor):
        """验证滚动窗口正确滑动 (T=2)。

        daily_ratio = [0.1, 0.2, 0.3, 0.4]
        rolling(2, min_periods=1):
          day0: 0.1
          day1: 0.15
          day2: 0.25
          day3: 0.35
        """
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        stocks = ["A"]

        net_comm = pd.DataFrame(
            [100.0, 200.0, 300.0, 400.0], index=dates, columns=stocks
        )
        amount = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(
            net_commission_increase=net_comm, amount=amount, T=2
        )

        assert result.iloc[0, 0] == pytest.approx(0.1, rel=1e-10)
        assert result.iloc[1, 0] == pytest.approx(0.15, rel=1e-10)
        assert result.iloc[2, 0] == pytest.approx(0.25, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(0.35, rel=1e-10)


class TestOpeningCommissionRatioEdgeCases:
    def test_zero_amount_returns_nan(self, factor):
        """成交额为零时, amount.replace(0, nan) 使比率为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        net_comm = pd.DataFrame([100.0, 200.0, 300.0], index=dates, columns=stocks)
        amount = pd.DataFrame([1000.0, 0.0, 1000.0], index=dates, columns=stocks)

        result = factor.compute(
            net_commission_increase=net_comm, amount=amount, T=3
        )

        # day1 ratio is NaN due to zero amount
        # rolling(3, min_periods=1) skips NaN:
        # day0: mean([0.1]) = 0.1
        # day1: mean([0.1, NaN]) = 0.1
        assert result.iloc[0, 0] == pytest.approx(0.1, rel=1e-10)
        assert result.iloc[1, 0] == pytest.approx(0.1, rel=1e-10)

    def test_nan_in_net_commission(self, factor):
        """net_commission_increase 中含 NaN 时, 不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        net_comm = pd.DataFrame([100.0, np.nan, 300.0], index=dates, columns=stocks)
        amount = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(
            net_commission_increase=net_comm, amount=amount, T=3
        )
        assert isinstance(result.iloc[2, 0], float)

    def test_negative_net_commission(self, factor):
        """净委买为负时, 因子值应为负。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        net_comm = pd.DataFrame([-100.0, -200.0, -300.0], index=dates, columns=stocks)
        amount = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(
            net_commission_increase=net_comm, amount=amount, T=3
        )
        assert (result.values < 0).all()

    def test_t_equals_one(self, factor):
        """T=1 时, 结果等于日度比率本身。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        net_comm = pd.DataFrame([100.0, 200.0, 300.0], index=dates, columns=stocks)
        amount = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(
            net_commission_increase=net_comm, amount=amount, T=1
        )

        np.testing.assert_allclose(result["A"].values, [0.1, 0.2, 0.3], atol=1e-12)


class TestOpeningCommissionRatioOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        stocks = ["A", "B", "C"]

        net_comm = pd.DataFrame(
            np.random.randn(50, 3) * 1e6, index=dates, columns=stocks
        )
        amount = pd.DataFrame(
            np.random.uniform(1e6, 1e7, (50, 3)), index=dates, columns=stocks
        )

        result = factor.compute(
            net_commission_increase=net_comm, amount=amount, T=20
        )

        assert result.shape == net_comm.shape
        assert list(result.columns) == list(net_comm.columns)
        assert list(result.index) == list(net_comm.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        net_comm = pd.DataFrame(100.0, index=dates, columns=stocks)
        amount = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(
            net_commission_increase=net_comm, amount=amount, T=3
        )
        assert isinstance(result, pd.DataFrame)

    def test_min_periods_1_all_rows_have_values(self, factor):
        """min_periods=1, 所以从第一行起就有值。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]

        net_comm = pd.DataFrame(
            np.random.randn(10, 2) * 1e6, index=dates, columns=stocks
        )
        amount = pd.DataFrame(
            np.random.uniform(1e6, 1e7, (10, 2)), index=dates, columns=stocks
        )

        result = factor.compute(
            net_commission_increase=net_comm, amount=amount, T=5
        )

        assert result.notna().all().all()
