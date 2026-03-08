import numpy as np
import pandas as pd
import pytest

from factors.informed_sell_ratio import InformedSellRatioFactor


@pytest.fixture
def factor():
    return InformedSellRatioFactor()


class TestInformedSellRatioMetadata:
    def test_name(self, factor):
        assert factor.name == "INFORMED_SELL_RATIO"

    def test_category(self, factor):
        assert factor.category == "高频资金流"

    def test_repr(self, factor):
        assert "INFORMED_SELL_RATIO" in repr(factor)
        assert "InformedSellRatioFactor" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "INFORMED_SELL_RATIO"
        assert meta["category"] == "高频资金流"
        assert "知情" in meta["description"]


class TestInformedSellRatioHandCalculated:
    def test_T3_single_stock(self, factor):
        """T=3, 单只股票, 手动计算验证。

        informed_sell = [100, 200, 300]
        total         = [1000, 1000, 1000]
        ratio         = [0.1, 0.2, 0.3]
        rolling mean(T=3) at t=2: (0.1 + 0.2 + 0.3) / 3 = 0.2
        """
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        informed = pd.DataFrame([100.0, 200.0, 300.0], index=dates, columns=stocks)
        total = pd.DataFrame([1000.0, 1000.0, 1000.0], index=dates, columns=stocks)

        result = factor.compute(informed_sell_amount=informed, total_amount=total, T=3)

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(0.2, rel=1e-10)

    def test_T2_two_stocks(self, factor):
        """T=2, 两只股票, 验证多列并行计算。"""
        dates = pd.date_range("2024-01-01", periods=2, freq="D")
        stocks = ["A", "B"]

        informed = pd.DataFrame(
            [[100.0, 500.0], [300.0, 100.0]], index=dates, columns=stocks
        )
        total = pd.DataFrame(
            [[1000.0, 1000.0], [1000.0, 1000.0]], index=dates, columns=stocks
        )

        result = factor.compute(informed_sell_amount=informed, total_amount=total, T=2)

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[0, 1])
        assert result.loc[dates[1], "A"] == pytest.approx(0.2, rel=1e-10)
        assert result.loc[dates[1], "B"] == pytest.approx(0.3, rel=1e-10)

    def test_constant_ratio(self, factor):
        """比例恒定时, 滚动均值等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["X"]

        informed = pd.DataFrame([200.0] * 5, index=dates, columns=stocks)
        total = pd.DataFrame([1000.0] * 5, index=dates, columns=stocks)

        result = factor.compute(informed_sell_amount=informed, total_amount=total, T=3)

        assert result.iloc[2, 0] == pytest.approx(0.2, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(0.2, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(0.2, rel=1e-10)

    def test_rolling_window_slides(self, factor):
        """验证滚动窗口正确滑动。

        ratio = [0.1, 0.2, 0.3, 0.4, 0.5], T=3
        t=2: mean(0.1, 0.2, 0.3) = 0.2
        t=3: mean(0.2, 0.3, 0.4) = 0.3
        t=4: mean(0.3, 0.4, 0.5) = 0.4
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        informed = pd.DataFrame([100.0, 200.0, 300.0, 400.0, 500.0], index=dates, columns=stocks)
        total = pd.DataFrame([1000.0] * 5, index=dates, columns=stocks)

        result = factor.compute(informed_sell_amount=informed, total_amount=total, T=3)

        assert result.iloc[2, 0] == pytest.approx(0.2, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(0.3, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(0.4, rel=1e-10)


class TestInformedSellRatioEdgeCases:
    def test_zero_total_amount(self, factor):
        """总成交金额为 0 时, ratio 为 inf/nan, 不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        informed = pd.DataFrame([100.0, 200.0, 300.0], index=dates, columns=stocks)
        total = pd.DataFrame([0.0, 0.0, 0.0], index=dates, columns=stocks)

        result = factor.compute(informed_sell_amount=informed, total_amount=total, T=3)

        assert result.shape == (3, 1)
        assert np.all(np.isinf(result.values) | np.isnan(result.values))

    def test_nan_in_informed(self, factor):
        """informed_sell 中含 NaN 时, 结果不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        informed = pd.DataFrame([100.0, np.nan, 300.0], index=dates, columns=stocks)
        total = pd.DataFrame([1000.0, 1000.0, 1000.0], index=dates, columns=stocks)

        result = factor.compute(informed_sell_amount=informed, total_amount=total, T=3)

        assert isinstance(result, pd.DataFrame)
        assert np.isnan(result.iloc[2, 0])

    def test_zero_informed_sell(self, factor):
        """知情卖出为 0 时, ratio=0, 均值=0。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        informed = pd.DataFrame([0.0, 0.0, 0.0], index=dates, columns=stocks)
        total = pd.DataFrame([1000.0, 1000.0, 1000.0], index=dates, columns=stocks)

        result = factor.compute(informed_sell_amount=informed, total_amount=total, T=3)

        assert result.iloc[2, 0] == pytest.approx(0.0, abs=1e-15)


class TestInformedSellRatioOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]

        informed = pd.DataFrame(
            np.random.uniform(1e6, 5e6, (30, 3)), index=dates, columns=stocks
        )
        total = pd.DataFrame(
            np.random.uniform(1e7, 5e7, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(informed_sell_amount=informed, total_amount=total, T=20)

        assert result.shape == informed.shape
        assert list(result.columns) == list(informed.columns)
        assert list(result.index) == list(informed.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        informed = pd.DataFrame([100.0] * 5, index=dates, columns=stocks)
        total = pd.DataFrame([1000.0] * 5, index=dates, columns=stocks)

        result = factor.compute(informed_sell_amount=informed, total_amount=total, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        """前 T-1 行应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A", "B"]
        T = 20

        informed = pd.DataFrame(
            np.random.uniform(1e6, 5e6, (25, 2)), index=dates, columns=stocks
        )
        total = pd.DataFrame(
            np.random.uniform(1e7, 5e7, (25, 2)), index=dates, columns=stocks
        )

        result = factor.compute(informed_sell_amount=informed, total_amount=total, T=T)

        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
