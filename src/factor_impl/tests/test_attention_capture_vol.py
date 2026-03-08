import numpy as np
import pandas as pd
import pytest

from factors.attention_capture_vol import AttentionCaptureVolFactor


@pytest.fixture
def factor():
    return AttentionCaptureVolFactor()


@pytest.fixture
def sample_data():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    stocks = ["A", "B", "C"]
    stock_returns = pd.DataFrame(
        np.random.randn(30, 3) * 0.02, index=dates, columns=stocks
    )
    industry_vol = pd.Series(np.random.uniform(0.01, 0.05, 30), index=dates)
    mkt = pd.Series(np.random.randn(30) * 0.01, index=dates)
    smb = pd.Series(np.random.randn(30) * 0.005, index=dates)
    hml = pd.Series(np.random.randn(30) * 0.005, index=dates)
    umd = pd.Series(np.random.randn(30) * 0.005, index=dates)
    return stock_returns, industry_vol, mkt, smb, hml, umd


class TestAttentionCaptureVolMetadata:
    def test_name(self, factor):
        assert factor.name == "ATTENTION_CAPTURE_VOL"

    def test_category(self, factor):
        assert factor.category == "行为金融投资者注意力"

    def test_repr(self, factor):
        assert "ATTENTION_CAPTURE_VOL" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "ATTENTION_CAPTURE_VOL"


class TestAttentionCaptureVolCompute:
    def test_output_non_negative(self, factor, sample_data):
        """绝对值应非负。"""
        stock_returns, industry_vol, mkt, smb, hml, umd = sample_data
        result = factor.compute(
            stock_returns=stock_returns, industry_vol=industry_vol,
            mkt=mkt, smb=smb, hml=hml, umd=umd, T=20
        )
        valid = result.values[~np.isnan(result.values)]
        assert (valid >= -1e-10).all()

    def test_leading_nan(self, factor, sample_data):
        """前 T-1 行应为 NaN。"""
        stock_returns, industry_vol, mkt, smb, hml, umd = sample_data
        result = factor.compute(
            stock_returns=stock_returns, industry_vol=industry_vol,
            mkt=mkt, smb=smb, hml=hml, umd=umd, T=20
        )
        assert result.iloc[:19].isna().all().all()

    def test_zero_industry_vol(self, factor):
        """行业波动率为0时，beta应为0。"""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        stock_returns = pd.DataFrame(
            np.random.randn(30, 2) * 0.02, index=dates, columns=stocks
        )
        industry_vol = pd.Series(0.0, index=dates)
        mkt = pd.Series(np.random.randn(30) * 0.01, index=dates)
        smb = pd.Series(0.0, index=dates)
        hml = pd.Series(0.0, index=dates)
        umd = pd.Series(0.0, index=dates)

        result = factor.compute(
            stock_returns=stock_returns, industry_vol=industry_vol,
            mkt=mkt, smb=smb, hml=hml, umd=umd, T=20
        )
        # industry_vol 全为0，回归中该列全为0，beta[1]应为0
        valid = result.dropna()
        if len(valid) > 0:
            for col in stocks:
                vals = valid[col].values
                assert all(v == pytest.approx(0.0, abs=1e-6) or np.isnan(v) for v in vals)


class TestAttentionCaptureVolEdgeCases:
    def test_nan_in_input(self, factor, sample_data):
        stock_returns, industry_vol, mkt, smb, hml, umd = sample_data
        stock_returns.iloc[5, 0] = np.nan
        result = factor.compute(
            stock_returns=stock_returns, industry_vol=industry_vol,
            mkt=mkt, smb=smb, hml=hml, umd=umd, T=20
        )
        assert isinstance(result, pd.DataFrame)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A"]
        stock_returns = pd.DataFrame(np.nan, index=dates, columns=stocks)
        industry_vol = pd.Series(np.nan, index=dates)
        mkt = pd.Series(np.nan, index=dates)
        smb = pd.Series(np.nan, index=dates)
        hml = pd.Series(np.nan, index=dates)
        umd = pd.Series(np.nan, index=dates)

        result = factor.compute(
            stock_returns=stock_returns, industry_vol=industry_vol,
            mkt=mkt, smb=smb, hml=hml, umd=umd, T=20
        )
        assert result.isna().all().all()


class TestAttentionCaptureVolOutputShape:
    def test_output_shape(self, factor, sample_data):
        stock_returns, industry_vol, mkt, smb, hml, umd = sample_data
        result = factor.compute(
            stock_returns=stock_returns, industry_vol=industry_vol,
            mkt=mkt, smb=smb, hml=hml, umd=umd, T=20
        )
        assert result.shape == stock_returns.shape

    def test_output_is_dataframe(self, factor, sample_data):
        stock_returns, industry_vol, mkt, smb, hml, umd = sample_data
        result = factor.compute(
            stock_returns=stock_returns, industry_vol=industry_vol,
            mkt=mkt, smb=smb, hml=hml, umd=umd, T=20
        )
        assert isinstance(result, pd.DataFrame)
