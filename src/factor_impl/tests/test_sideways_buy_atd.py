import numpy as np
import pandas as pd
import pytest

from factors.sideways_buy_atd import SidewaysBuyATDFactor


@pytest.fixture
def factor():
    return SidewaysBuyATDFactor()


class TestSidewaysBuyATDMetadata:
    def test_name(self, factor):
        assert factor.name == "SIDEWAYS_BUY_ATD"

    def test_category(self, factor):
        assert factor.category == "高频成交分布"

    def test_repr(self, factor):
        assert "SIDEWAYS_BUY_ATD" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "SIDEWAYS_BUY_ATD"


class TestSidewaysBuyATDCompute:
    def test_known_values(self, factor):
        """手算验证 SATD = (amt_buy/num_buy) / (amt_total/num_total)。"""
        dates = pd.date_range("2024-01-01", periods=3)
        zero_buy_amt = pd.DataFrame({"A": [1000.0, 2000.0, 3000.0]}, index=dates)
        zero_buy_num = pd.DataFrame({"A": [10.0, 20.0, 30.0]}, index=dates)
        total_amt = pd.DataFrame({"A": [5000.0, 10000.0, 15000.0]}, index=dates)
        total_num = pd.DataFrame({"A": [100.0, 200.0, 300.0]}, index=dates)

        result = factor.compute(
            zero_buy_amt=zero_buy_amt,
            zero_buy_deal_num=zero_buy_num,
            total_amt=total_amt,
            total_deal_num=total_num,
        )

        # ATD_buy = 1000/10 = 100, ATD_total = 5000/100 = 50, SATD = 100/50 = 2.0
        assert result.iloc[0, 0] == pytest.approx(2.0)
        # ATD_buy = 2000/20 = 100, ATD_total = 10000/200 = 50, SATD = 2.0
        assert result.iloc[1, 0] == pytest.approx(2.0)

    def test_equal_atd(self, factor):
        """横盘笔均 == 全天笔均时，因子值为1。"""
        dates = pd.date_range("2024-01-01", periods=2)
        amt = pd.DataFrame({"A": [1000.0, 2000.0]}, index=dates)
        num = pd.DataFrame({"A": [10.0, 20.0]}, index=dates)

        result = factor.compute(
            zero_buy_amt=amt, zero_buy_deal_num=num,
            total_amt=amt, total_deal_num=num,
        )
        for i in range(2):
            assert result.iloc[i, 0] == pytest.approx(1.0)


class TestSidewaysBuyATDEdgeCases:
    def test_zero_deal_num(self, factor):
        """成交笔数为0时结果为 inf/NaN。"""
        dates = pd.date_range("2024-01-01", periods=2)
        amt = pd.DataFrame({"A": [1000.0, 2000.0]}, index=dates)
        zero_num = pd.DataFrame({"A": [0.0, 0.0]}, index=dates)
        total_num = pd.DataFrame({"A": [10.0, 20.0]}, index=dates)

        result = factor.compute(
            zero_buy_amt=amt, zero_buy_deal_num=zero_num,
            total_amt=amt, total_deal_num=total_num,
        )
        assert isinstance(result, pd.DataFrame)

    def test_nan_propagation(self, factor):
        dates = pd.date_range("2024-01-01", periods=3)
        amt = pd.DataFrame({"A": [1000.0, np.nan, 3000.0]}, index=dates)
        num = pd.DataFrame({"A": [10.0, 20.0, 30.0]}, index=dates)

        result = factor.compute(
            zero_buy_amt=amt, zero_buy_deal_num=num,
            total_amt=amt, total_deal_num=num,
        )
        assert np.isnan(result.iloc[1, 0])


class TestSidewaysBuyATDOutputShape:
    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=10)
        stocks = ["A", "B"]
        df = pd.DataFrame(np.random.rand(10, 2) * 1000 + 1, index=dates, columns=stocks)
        num = pd.DataFrame(np.random.rand(10, 2) * 100 + 1, index=dates, columns=stocks)

        result = factor.compute(
            zero_buy_amt=df, zero_buy_deal_num=num,
            total_amt=df, total_deal_num=num,
        )
        assert result.shape == (10, 2)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=3)
        df = pd.DataFrame({"A": [100.0, 200.0, 300.0]}, index=dates)
        num = pd.DataFrame({"A": [10.0, 20.0, 30.0]}, index=dates)

        result = factor.compute(
            zero_buy_amt=df, zero_buy_deal_num=num,
            total_amt=df, total_deal_num=num,
        )
        assert isinstance(result, pd.DataFrame)
