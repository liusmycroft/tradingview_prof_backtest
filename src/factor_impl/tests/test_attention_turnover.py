import numpy as np
import pandas as pd
import pytest

from factors.attention_turnover import AttentionTurnoverFactor


@pytest.fixture
def factor():
    return AttentionTurnoverFactor()


@pytest.fixture
def simple_data():
    """构造简单的回归数据，使 beta 可手算验证。"""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    stocks = ["A"]

    np.random.seed(42)
    indus_turn = np.random.uniform(0.01, 0.05, 10)
    mkt = np.random.uniform(-0.02, 0.02, 10)
    smb = np.random.uniform(-0.01, 0.01, 10)
    hml = np.random.uniform(-0.01, 0.01, 10)
    umd = np.random.uniform(-0.01, 0.01, 10)

    # 构造 r = 0.01 + 2.0 * indus_turn + 0.5*mkt + noise
    noise = np.random.normal(0, 0.001, 10)
    r = 0.01 + 2.0 * indus_turn + 0.5 * mkt + noise

    returns = pd.DataFrame(r, index=dates, columns=stocks)
    industry_turnover = pd.DataFrame(indus_turn, index=dates, columns=stocks)
    mkt_df = pd.DataFrame(mkt, index=dates, columns=["MKT"])
    smb_df = pd.DataFrame(smb, index=dates, columns=["SMB"])
    hml_df = pd.DataFrame(hml, index=dates, columns=["HML"])
    umd_df = pd.DataFrame(umd, index=dates, columns=["UMD"])

    return returns, industry_turnover, mkt_df, smb_df, hml_df, umd_df


class TestAttentionTurnoverMetadata:
    def test_name(self, factor):
        assert factor.name == "ATTENTION_TURNOVER"

    def test_category(self, factor):
        assert factor.category == "行为金融-投资者注意力"

    def test_repr(self, factor):
        assert "ATTENTION_TURNOVER" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "ATTENTION_TURNOVER"
        assert meta["category"] == "行为金融-投资者注意力"


class TestAttentionTurnoverHandCalculated:
    def test_known_beta(self, factor, simple_data):
        """构造 r = mu + 2*indus_turn + controls + noise, |beta| 应接近 2。"""
        returns, industry_turnover, mkt, smb, hml, umd = simple_data

        result = factor.compute(
            returns=returns,
            industry_turnover=industry_turnover,
            mkt=mkt, smb=smb, hml=hml, umd=umd,
            T=10,
        )

        # 最后一行应有值，且 |beta| 接近 2.0
        val = result.iloc[-1, 0]
        assert not np.isnan(val)
        assert val == pytest.approx(2.0, abs=0.3)

    def test_zero_sensitivity(self, factor):
        """r 与 indus_turn 无关时, |beta| 应接近 0。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]

        np.random.seed(123)
        returns = pd.DataFrame(
            np.random.normal(0, 0.01, (20, 1)), index=dates, columns=stocks
        )
        industry_turnover = pd.DataFrame(
            np.random.uniform(0.01, 0.05, (20, 1)), index=dates, columns=stocks
        )
        mkt = pd.DataFrame(np.zeros(20), index=dates, columns=["MKT"])
        smb = pd.DataFrame(np.zeros(20), index=dates, columns=["SMB"])
        hml = pd.DataFrame(np.zeros(20), index=dates, columns=["HML"])
        umd = pd.DataFrame(np.zeros(20), index=dates, columns=["UMD"])

        result = factor.compute(
            returns=returns,
            industry_turnover=industry_turnover,
            mkt=mkt, smb=smb, hml=hml, umd=umd,
            T=20,
        )

        val = result.iloc[-1, 0]
        assert not np.isnan(val)
        assert val < 1.0  # 应远小于有信号时的值

    def test_two_stocks_independent(self, factor):
        """两只股票应独立回归。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]

        np.random.seed(42)
        indus_turn_a = np.random.uniform(0.01, 0.05, 10)
        indus_turn_b = np.random.uniform(0.01, 0.05, 10)
        mkt = np.random.uniform(-0.02, 0.02, 10)

        r_a = 0.01 + 3.0 * indus_turn_a + 0.5 * mkt
        r_b = 0.01 + 0.5 * indus_turn_b + 0.5 * mkt

        returns = pd.DataFrame({"A": r_a, "B": r_b}, index=dates)
        industry_turnover = pd.DataFrame(
            {"A": indus_turn_a, "B": indus_turn_b}, index=dates
        )
        mkt_df = pd.DataFrame(mkt, index=dates, columns=["MKT"])
        smb_df = pd.DataFrame(np.zeros(10), index=dates, columns=["SMB"])
        hml_df = pd.DataFrame(np.zeros(10), index=dates, columns=["HML"])
        umd_df = pd.DataFrame(np.zeros(10), index=dates, columns=["UMD"])

        result = factor.compute(
            returns=returns,
            industry_turnover=industry_turnover,
            mkt=mkt_df, smb=smb_df, hml=hml_df, umd=umd_df,
            T=10,
        )

        # A 的 |beta| 应大于 B 的 |beta|
        assert result.iloc[-1, 0] > result.iloc[-1, 1]


class TestAttentionTurnoverEdgeCases:
    def test_insufficient_data(self, factor):
        """数据不足 T 天时, 前 T-1 行应为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        returns = pd.DataFrame(np.zeros((5, 1)), index=dates, columns=stocks)
        industry_turnover = pd.DataFrame(
            np.ones((5, 1)) * 0.03, index=dates, columns=stocks
        )
        mkt = pd.DataFrame(np.zeros(5), index=dates, columns=["MKT"])
        smb = pd.DataFrame(np.zeros(5), index=dates, columns=["SMB"])
        hml = pd.DataFrame(np.zeros(5), index=dates, columns=["HML"])
        umd = pd.DataFrame(np.zeros(5), index=dates, columns=["UMD"])

        result = factor.compute(
            returns=returns,
            industry_turnover=industry_turnover,
            mkt=mkt, smb=smb, hml=hml, umd=umd,
            T=20,
        )

        assert result.isna().all().all()

    def test_nan_in_returns(self, factor):
        """returns 含 NaN 时不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]

        np.random.seed(42)
        r = np.random.normal(0, 0.01, 10)
        r[3] = np.nan
        returns = pd.DataFrame(r, index=dates, columns=stocks)
        industry_turnover = pd.DataFrame(
            np.random.uniform(0.01, 0.05, (10, 1)), index=dates, columns=stocks
        )
        mkt = pd.DataFrame(np.zeros(10), index=dates, columns=["MKT"])
        smb = pd.DataFrame(np.zeros(10), index=dates, columns=["SMB"])
        hml = pd.DataFrame(np.zeros(10), index=dates, columns=["HML"])
        umd = pd.DataFrame(np.zeros(10), index=dates, columns=["UMD"])

        result = factor.compute(
            returns=returns,
            industry_turnover=industry_turnover,
            mkt=mkt, smb=smb, hml=hml, umd=umd,
            T=10,
        )

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_result_non_negative(self, factor, simple_data):
        """|beta| 应始终 >= 0。"""
        returns, industry_turnover, mkt, smb, hml, umd = simple_data

        result = factor.compute(
            returns=returns,
            industry_turnover=industry_turnover,
            mkt=mkt, smb=smb, hml=hml, umd=umd,
            T=10,
        )

        valid = result.dropna()
        assert (valid.values >= 0).all()


class TestAttentionTurnoverOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]

        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.normal(0, 0.01, (30, 3)), index=dates, columns=stocks
        )
        industry_turnover = pd.DataFrame(
            np.random.uniform(0.01, 0.05, (30, 3)), index=dates, columns=stocks
        )
        mkt = pd.DataFrame(np.zeros(30), index=dates, columns=["MKT"])
        smb = pd.DataFrame(np.zeros(30), index=dates, columns=["SMB"])
        hml = pd.DataFrame(np.zeros(30), index=dates, columns=["HML"])
        umd = pd.DataFrame(np.zeros(30), index=dates, columns=["UMD"])

        result = factor.compute(
            returns=returns,
            industry_turnover=industry_turnover,
            mkt=mkt, smb=smb, hml=hml, umd=umd,
            T=20,
        )

        assert result.shape == returns.shape
        assert list(result.columns) == list(returns.columns)
        assert list(result.index) == list(returns.index)

    def test_output_is_dataframe(self, factor, simple_data):
        returns, industry_turnover, mkt, smb, hml, umd = simple_data

        result = factor.compute(
            returns=returns,
            industry_turnover=industry_turnover,
            mkt=mkt, smb=smb, hml=hml, umd=umd,
            T=10,
        )
        assert isinstance(result, pd.DataFrame)
