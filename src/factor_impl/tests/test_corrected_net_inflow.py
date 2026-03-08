import numpy as np
import pandas as pd
import pytest

from factors.corrected_net_inflow import CorrectedNetInflowFactor


@pytest.fixture
def factor():
    return CorrectedNetInflowFactor()


class TestCNIRMetadata:
    def test_name(self, factor):
        assert factor.name == "CNIR"

    def test_category(self, factor):
        assert factor.category == "高频资金流"

    def test_repr(self, factor):
        assert "CNIR" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "CNIR"
        assert meta["category"] == "高频资金流"


class TestCNIRHandCalculated:
    def test_equal_buy_sell_zero_return(self, factor):
        """买卖相等且收益率为0时，残差为0，e^0/(1+e^0)=0.5，
        B_hat=S_hat，CNIR应为0。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        buy = pd.DataFrame(1e6, index=dates, columns=stocks)
        sell = pd.DataFrame(1e6, index=dates, columns=stocks)
        ret = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_buy_amount=buy, daily_sell_amount=sell, daily_return=ret, T=20
        )
        # ln(1) = 0, regression on zeros => eps = 0, CNIR = 0
        assert result.iloc[-1, 0] == pytest.approx(0.0, abs=1e-10)

    def test_output_bounded(self, factor):
        """CNIR = (B_hat - S_hat) / (B_hat + S_hat) 应在 [-1, 1] 范围内。"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        buy = pd.DataFrame(
            np.random.uniform(1e5, 1e7, (30, 2)), index=dates, columns=stocks
        )
        sell = pd.DataFrame(
            np.random.uniform(1e5, 1e7, (30, 2)), index=dates, columns=stocks
        )
        ret = pd.DataFrame(
            np.random.randn(30, 2) * 0.03, index=dates, columns=stocks
        )

        result = factor.compute(
            daily_buy_amount=buy, daily_sell_amount=sell, daily_return=ret, T=20
        )
        valid = result.dropna()
        assert (valid >= -1.0 - 1e-10).all().all()
        assert (valid <= 1.0 + 1e-10).all().all()

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A", "B"]
        buy = pd.DataFrame(
            np.random.uniform(1e5, 1e7, (25, 2)), index=dates, columns=stocks
        )
        sell = pd.DataFrame(
            np.random.uniform(1e5, 1e7, (25, 2)), index=dates, columns=stocks
        )
        ret = pd.DataFrame(
            np.random.randn(25, 2) * 0.02, index=dates, columns=stocks
        )

        result = factor.compute(
            daily_buy_amount=buy, daily_sell_amount=sell, daily_return=ret, T=20
        )
        # Just check they are different (different random data)
        valid_rows = result.dropna()
        if len(valid_rows) > 0:
            assert not np.allclose(
                valid_rows["A"].values, valid_rows["B"].values, atol=1e-10
            )


class TestCNIREdgeCases:
    def test_insufficient_data(self, factor):
        """数据不足T天时，前面应为NaN。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        buy = pd.DataFrame(1e6, index=dates, columns=stocks)
        sell = pd.DataFrame(1e6, index=dates, columns=stocks)
        ret = pd.DataFrame(0.01, index=dates, columns=stocks)

        result = factor.compute(
            daily_buy_amount=buy, daily_sell_amount=sell, daily_return=ret, T=20
        )
        # T=20 but only 10 rows, all should be NaN
        assert result.isna().all().all()

    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        buy_vals = np.ones(25) * 1e6
        buy_vals[5] = np.nan
        buy = pd.DataFrame(buy_vals, index=dates, columns=stocks)
        sell = pd.DataFrame(1e6, index=dates, columns=stocks)
        ret = pd.DataFrame(0.01, index=dates, columns=stocks)

        result = factor.compute(
            daily_buy_amount=buy, daily_sell_amount=sell, daily_return=ret, T=20
        )
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (25, 1)


class TestCNIROutputShape:
    def test_output_shape_matches_input(self, factor):
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        buy = pd.DataFrame(
            np.random.uniform(1e5, 1e7, (30, 3)), index=dates, columns=stocks
        )
        sell = pd.DataFrame(
            np.random.uniform(1e5, 1e7, (30, 3)), index=dates, columns=stocks
        )
        ret = pd.DataFrame(
            np.random.randn(30, 3) * 0.02, index=dates, columns=stocks
        )

        result = factor.compute(
            daily_buy_amount=buy, daily_sell_amount=sell, daily_return=ret, T=20
        )
        assert result.shape == buy.shape
        assert list(result.columns) == list(buy.columns)
        assert list(result.index) == list(buy.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        buy = pd.DataFrame(1e6, index=dates, columns=stocks)
        sell = pd.DataFrame(1e6, index=dates, columns=stocks)
        ret = pd.DataFrame(0.01, index=dates, columns=stocks)

        result = factor.compute(
            daily_buy_amount=buy, daily_sell_amount=sell, daily_return=ret, T=20
        )
        assert isinstance(result, pd.DataFrame)
