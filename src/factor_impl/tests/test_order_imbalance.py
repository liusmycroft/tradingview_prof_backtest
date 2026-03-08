import numpy as np
import pandas as pd
import pytest

from factors.order_imbalance import OrderImbalanceFactor


@pytest.fixture
def factor():
    return OrderImbalanceFactor()


class TestOrderImbalanceMetadata:
    def test_name(self, factor):
        assert factor.name == "ORDER_IMBALANCE"

    def test_category(self, factor):
        assert factor.category == "高频流动性"

    def test_repr(self, factor):
        assert "ORDER_IMBALANCE" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "ORDER_IMBALANCE"
        assert meta["category"] == "高频流动性"


class TestOrderImbalanceHandCalculated:
    """用手算数据验证订单失衡因子。"""

    def test_constant_voi(self, factor):
        """常数 VOI=50, T=3 => 均值=50。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        voi = pd.DataFrame(50.0, index=dates, columns=stocks)

        result = factor.compute(daily_voi=voi, T=3)

        # min_periods=T=3, 前2行NaN
        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(50.0, rel=1e-10)

    def test_varying_voi_T3(self, factor):
        """T=3, 变化的 VOI。

        voi = [100, 200, 300]
        rolling(3, min_periods=3):
          day2: mean([100, 200, 300]) = 200
        """
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        voi = pd.DataFrame([100.0, 200.0, 300.0], index=dates, columns=stocks)

        result = factor.compute(daily_voi=voi, T=3)

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(200.0, rel=1e-10)

    def test_symmetric_voi_cancels(self, factor):
        """前半正后半负, 对称时均值为 0。

        voi = [100, 100, -100, -100]
        T=4: mean = 0
        """
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        stocks = ["A"]

        voi = pd.DataFrame([100.0, 100.0, -100.0, -100.0], index=dates, columns=stocks)

        result = factor.compute(daily_voi=voi, T=4)

        assert result.iloc[3, 0] == pytest.approx(0.0, abs=1e-10)

    def test_two_stocks(self, factor):
        """两只股票并行计算。

        Stock A: voi = [10, 20, 30] -> T=3: mean=20
        Stock B: voi = [-10, -20, -30] -> T=3: mean=-20
        """
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A", "B"]

        voi = pd.DataFrame(
            {"A": [10.0, 20.0, 30.0], "B": [-10.0, -20.0, -30.0]}, index=dates
        )

        result = factor.compute(daily_voi=voi, T=3)

        assert result.loc[dates[2], "A"] == pytest.approx(20.0, rel=1e-10)
        assert result.loc[dates[2], "B"] == pytest.approx(-20.0, rel=1e-10)

    def test_rolling_window_slides(self, factor):
        """验证滚动窗口正确滑动 (T=3)。

        voi = [1, 2, 3, 4, 5]
        rolling(3, min_periods=3):
          day2: mean(1,2,3) = 2
          day3: mean(2,3,4) = 3
          day4: mean(3,4,5) = 4
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        voi = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0], index=dates, columns=stocks)

        result = factor.compute(daily_voi=voi, T=3)

        assert result.iloc[2, 0] == pytest.approx(2.0, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(3.0, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(4.0, rel=1e-10)


class TestOrderImbalanceEdgeCases:
    def test_nan_in_voi(self, factor):
        """VOI 中含 NaN 时, 结果不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        voi = pd.DataFrame([100.0, np.nan, 300.0], index=dates, columns=stocks)

        result = factor.compute(daily_voi=voi, T=3)
        # 窗口内有 NaN, rolling mean 结果为 NaN
        assert np.isnan(result.iloc[2, 0])

    def test_all_zero_voi(self, factor):
        """VOI 全为 0 时, 因子值应为 0。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        voi = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_voi=voi, T=3)
        assert result.iloc[2, 0] == pytest.approx(0.0, abs=1e-15)

    def test_negative_voi(self, factor):
        """全部为负 VOI 时, 因子值应为负。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        voi = pd.DataFrame(-50.0, index=dates, columns=stocks)

        result = factor.compute(daily_voi=voi, T=3)
        assert result.iloc[2, 0] == pytest.approx(-50.0, rel=1e-10)

    def test_insufficient_data(self, factor):
        """数据不足 T 天时, 全部为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        voi = pd.DataFrame(50.0, index=dates, columns=stocks)

        result = factor.compute(daily_voi=voi, T=20)
        assert result.isna().all().all()


class TestOrderImbalanceOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        stocks = ["A", "B", "C"]

        voi = pd.DataFrame(
            np.random.randn(50, 3) * 100, index=dates, columns=stocks
        )

        result = factor.compute(daily_voi=voi, T=20)

        assert result.shape == voi.shape
        assert list(result.columns) == list(voi.columns)
        assert list(result.index) == list(voi.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        voi = pd.DataFrame(50.0, index=dates, columns=stocks)

        result = factor.compute(daily_voi=voi, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        """前 T-1 行应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        T = 5

        voi = pd.DataFrame(
            np.random.randn(10, 2) * 100, index=dates, columns=stocks
        )

        result = factor.compute(daily_voi=voi, T=T)

        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
