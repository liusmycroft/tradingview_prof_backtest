import numpy as np
import pandas as pd
import pytest

from factors.intraday_ret_iv import IntradayRetIVFactor


@pytest.fixture
def factor():
    return IntradayRetIVFactor()


class TestIntradayRetIVMetadata:
    def test_name(self, factor):
        assert factor.name == "INTRADAY_RET_IV"

    def test_category(self, factor):
        assert factor.category == "高频波动跳跃"

    def test_repr(self, factor):
        assert "INTRADAY_RET_IV" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "INTRADAY_RET_IV"


class TestIntradayRetIVCompute:
    def test_output_non_negative(self, factor):
        """标准差应非负。"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=40, freq="D")
        stocks = ["A", "B", "C", "D", "E"]
        intraday_ret = pd.DataFrame(
            np.random.randn(40, 5) * 0.02, index=dates, columns=stocks
        )

        result = factor.compute(intraday_ret=intraday_ret, T=20)
        valid = result.values[~np.isnan(result.values)]
        assert (valid >= -1e-10).all()

    def test_constant_returns(self, factor):
        """常数收益率：残差应为0，std应为0。"""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C", "D"]
        intraday_ret = pd.DataFrame(0.01, index=dates, columns=stocks)

        result = factor.compute(intraday_ret=intraday_ret, T=20)
        valid = result.dropna()
        if len(valid) > 0:
            for col in stocks:
                vals = valid[col].values
                assert all(v == pytest.approx(0.0, abs=1e-8) or np.isnan(v) for v in vals)

    def test_insufficient_stocks(self, factor):
        """股票数不足时结果为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        intraday_ret = pd.DataFrame(
            np.random.randn(30, 2) * 0.02, index=dates, columns=stocks
        )

        result = factor.compute(intraday_ret=intraday_ret, T=20)
        assert result.isna().all().all()


class TestIntradayRetIVEdgeCases:
    def test_nan_in_input(self, factor):
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=40, freq="D")
        stocks = ["A", "B", "C", "D"]
        intraday_ret = pd.DataFrame(
            np.random.randn(40, 4) * 0.02, index=dates, columns=stocks
        )
        intraday_ret.iloc[5, 0] = np.nan

        result = factor.compute(intraday_ret=intraday_ret, T=20)
        assert isinstance(result, pd.DataFrame)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C", "D"]
        intraday_ret = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(intraday_ret=intraday_ret, T=20)
        assert result.isna().all().all()


class TestIntradayRetIVOutputShape:
    def test_output_shape(self, factor):
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=40, freq="D")
        stocks = ["A", "B", "C", "D"]
        intraday_ret = pd.DataFrame(
            np.random.randn(40, 4) * 0.02, index=dates, columns=stocks
        )

        result = factor.compute(intraday_ret=intraday_ret, T=20)
        assert result.shape == intraday_ret.shape

    def test_output_is_dataframe(self, factor):
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C", "D"]
        intraday_ret = pd.DataFrame(
            np.random.randn(30, 4) * 0.02, index=dates, columns=stocks
        )

        result = factor.compute(intraday_ret=intraday_ret, T=20)
        assert isinstance(result, pd.DataFrame)
