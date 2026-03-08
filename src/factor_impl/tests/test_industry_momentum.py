import numpy as np
import pandas as pd
import pytest

from factors.industry_momentum import IndustryMomentumFactor


@pytest.fixture
def factor():
    return IndustryMomentumFactor()


class TestIndustryMomentumMetadata:
    def test_name(self, factor):
        assert factor.name == "INDUSTRY_MOMENTUM"

    def test_category(self, factor):
        assert factor.category == "动量溢出"

    def test_repr(self, factor):
        assert "INDUSTRY_MOMENTUM" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "INDUSTRY_MOMENTUM"
        assert meta["category"] == "动量溢出"


class TestIndustryMomentumHandCalculated:
    def test_two_stocks_same_industry(self, factor):
        """同行业两只股票，互为对方的行业动量。

        A: ret=0.02, cap=100
        B: ret=0.03, cap=200
        同行业 "银行"

        A 的行业动量 = B 的 cap 加权 ret = 0.03 (只有 B)
        B 的行业动量 = A 的 cap 加权 ret = 0.02 (只有 A)
        """
        stocks = ["A", "B"]
        returns = pd.Series([0.02, 0.03], index=stocks)
        market_cap = pd.Series([100.0, 200.0], index=stocks)
        industry = pd.Series(["银行", "银行"], index=stocks)

        result = factor.compute(returns=returns, market_cap=market_cap, industry=industry)

        assert result["A"] == pytest.approx(0.03, rel=1e-10)
        assert result["B"] == pytest.approx(0.02, rel=1e-10)

    def test_three_stocks_same_industry(self, factor):
        """同行业三只股票。

        A: ret=0.01, cap=100
        B: ret=0.02, cap=200
        C: ret=0.03, cap=300

        A 的行业动量 = (200*0.02 + 300*0.03) / (200+300) = (4+9)/500 = 0.026
        """
        stocks = ["A", "B", "C"]
        returns = pd.Series([0.01, 0.02, 0.03], index=stocks)
        market_cap = pd.Series([100.0, 200.0, 300.0], index=stocks)
        industry = pd.Series(["科技", "科技", "科技"], index=stocks)

        result = factor.compute(returns=returns, market_cap=market_cap, industry=industry)

        expected_a = (200 * 0.02 + 300 * 0.03) / (200 + 300)
        assert result["A"] == pytest.approx(expected_a, rel=1e-10)

        expected_b = (100 * 0.01 + 300 * 0.03) / (100 + 300)
        assert result["B"] == pytest.approx(expected_b, rel=1e-10)

        expected_c = (100 * 0.01 + 200 * 0.02) / (100 + 200)
        assert result["C"] == pytest.approx(expected_c, rel=1e-10)

    def test_different_industries(self, factor):
        """不同行业的股票互不影响。"""
        stocks = ["A", "B", "C", "D"]
        returns = pd.Series([0.01, 0.05, -0.02, 0.03], index=stocks)
        market_cap = pd.Series([100.0, 200.0, 150.0, 300.0], index=stocks)
        industry = pd.Series(["银行", "银行", "科技", "科技"], index=stocks)

        result = factor.compute(returns=returns, market_cap=market_cap, industry=industry)

        # A 的行业动量 = B 的 ret = 0.05
        assert result["A"] == pytest.approx(0.05, rel=1e-10)
        # B 的行业动量 = A 的 ret = 0.01
        assert result["B"] == pytest.approx(0.01, rel=1e-10)
        # C 的行业动量 = D 的 cap 加权 ret = 0.03
        assert result["C"] == pytest.approx(0.03, rel=1e-10)
        # D 的行业动量 = C 的 cap 加权 ret = -0.02
        assert result["D"] == pytest.approx(-0.02, rel=1e-10)


class TestIndustryMomentumEdgeCases:
    def test_single_stock_in_industry(self, factor):
        """行业内只有一只股票时，结果为 NaN。"""
        stocks = ["A"]
        returns = pd.Series([0.02], index=stocks)
        market_cap = pd.Series([100.0], index=stocks)
        industry = pd.Series(["银行"], index=stocks)

        result = factor.compute(returns=returns, market_cap=market_cap, industry=industry)
        assert np.isnan(result["A"])

    def test_zero_market_cap(self, factor):
        """同行业其他股票市值全为 0 时，结果为 NaN。"""
        stocks = ["A", "B"]
        returns = pd.Series([0.02, 0.03], index=stocks)
        market_cap = pd.Series([100.0, 0.0], index=stocks)
        industry = pd.Series(["银行", "银行"], index=stocks)

        result = factor.compute(returns=returns, market_cap=market_cap, industry=industry)
        assert np.isnan(result["A"])

    def test_output_type(self, factor):
        stocks = ["A", "B"]
        returns = pd.Series([0.02, 0.03], index=stocks)
        market_cap = pd.Series([100.0, 200.0], index=stocks)
        industry = pd.Series(["银行", "银行"], index=stocks)

        result = factor.compute(returns=returns, market_cap=market_cap, industry=industry)
        assert isinstance(result, pd.Series)
        assert len(result) == 2
