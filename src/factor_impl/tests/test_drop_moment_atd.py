import numpy as np
import pandas as pd
import pytest

from factors.drop_moment_atd import DropMomentATDFactor


@pytest.fixture
def factor():
    return DropMomentATDFactor()


class TestDropMomentATDMetadata:
    def test_name(self, factor):
        assert factor.name == "DROP_MOMENT_ATD"

    def test_category(self, factor):
        assert factor.category == "高频因子-成交分布类"

    def test_repr(self, factor):
        assert "DROP_MOMENT_ATD" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "DROP_MOMENT_ATD"


class TestDropMomentATDCompute:
    def test_equal_atd(self, factor):
        """下跌笔均 = 全天笔均时，SATD = 1.0。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        down_amt = pd.DataFrame(500.0, index=dates, columns=stocks)
        down_num = pd.DataFrame(50.0, index=dates, columns=stocks)
        total_amt = pd.DataFrame(1000.0, index=dates, columns=stocks)
        total_num = pd.DataFrame(100.0, index=dates, columns=stocks)

        result = factor.compute(
            down_amount=down_amt, down_deal_num=down_num,
            total_amount=total_amt, total_deal_num=total_num,
        )
        np.testing.assert_array_almost_equal(result["A"].values, 1.0)

    def test_higher_down_atd(self, factor):
        """下跌笔均 > 全天笔均时，SATD > 1.0。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        down_amt = pd.DataFrame(600.0, index=dates, columns=stocks)
        down_num = pd.DataFrame(30.0, index=dates, columns=stocks)  # ATD_down = 20
        total_amt = pd.DataFrame(1000.0, index=dates, columns=stocks)
        total_num = pd.DataFrame(100.0, index=dates, columns=stocks)  # ATD_T = 10

        result = factor.compute(
            down_amount=down_amt, down_deal_num=down_num,
            total_amount=total_amt, total_deal_num=total_num,
        )
        assert (result["A"] > 1.0).all()

    def test_zero_deal_num_nan(self, factor):
        """成交笔数为 0 时应返回 NaN。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        down_amt = pd.DataFrame(500.0, index=dates, columns=stocks)
        down_num = pd.DataFrame(0.0, index=dates, columns=stocks)
        total_amt = pd.DataFrame(1000.0, index=dates, columns=stocks)
        total_num = pd.DataFrame(100.0, index=dates, columns=stocks)

        result = factor.compute(
            down_amount=down_amt, down_deal_num=down_num,
            total_amount=total_amt, total_deal_num=total_num,
        )
        assert result.isna().all().all()

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        down_amt = pd.DataFrame(np.random.rand(10, 2) * 500, index=dates, columns=stocks)
        down_num = pd.DataFrame(np.random.randint(10, 50, (10, 2)).astype(float), index=dates, columns=stocks)
        total_amt = pd.DataFrame(np.random.rand(10, 2) * 1000 + 500, index=dates, columns=stocks)
        total_num = pd.DataFrame(np.random.randint(50, 200, (10, 2)).astype(float), index=dates, columns=stocks)

        result = factor.compute(
            down_amount=down_amt, down_deal_num=down_num,
            total_amount=total_amt, total_deal_num=total_num,
        )
        assert result.shape == down_amt.shape
        assert isinstance(result, pd.DataFrame)
