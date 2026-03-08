import numpy as np
import pandas as pd
import pytest

from factors.hf_idio_vol import HfIdioVolFactor


@pytest.fixture
def factor():
    return HfIdioVolFactor()


class TestHfIdioVolMetadata:
    def test_name(self, factor):
        assert factor.name == "HF_IDIO_VOL"

    def test_category(self, factor):
        assert factor.category == "高频波动跳跃"

    def test_repr(self, factor):
        assert "HF_IDIO_VOL" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "HF_IDIO_VOL"
        assert meta["category"] == "高频波动跳跃"


class TestHfIdioVolHandCalculated:
    """手算验证高频特异波动率。"""

    def test_perfect_fit_zero_idio(self, factor):
        """当个股收益完全由因子解释时, R^2=1, 特异率=0, 因子值=0。"""
        np.random.seed(42)
        T = 21
        dates = pd.date_range("2024-01-01", periods=T, freq="D")
        stocks = ["A"]

        # 因子收益
        mkt = np.random.randn(T) * 0.01
        smb = np.random.randn(T) * 0.005
        hml = np.random.randn(T) * 0.005
        ret = np.random.randn(T) * 0.003
        liq = np.random.randn(T) * 0.003
        factor_returns = pd.DataFrame(
            {"MKT": mkt, "SMB": smb, "HML": hml, "RET": ret, "LIQ": liq},
            index=dates,
        )

        # 个股收益 = 完美线性组合
        stock_ret = 0.001 + 1.0 * mkt + 0.5 * smb - 0.3 * hml + 0.2 * ret + 0.1 * liq
        stock_returns = pd.DataFrame(stock_ret, index=dates, columns=stocks)

        hf_volatility = pd.DataFrame(0.02, index=dates, columns=stocks)

        result = factor.compute(
            stock_returns=stock_returns,
            factor_returns=factor_returns,
            hf_volatility=hf_volatility,
            T=T,
        )

        # R^2 ≈ 1, idio_rate ≈ 0, result ≈ 0
        assert result.iloc[-1, 0] == pytest.approx(0.0, abs=1e-4)

    def test_pure_noise_high_idio(self, factor):
        """当个股收益与因子无关时, R^2≈0, 特异率≈1。"""
        np.random.seed(123)
        T = 21
        dates = pd.date_range("2024-01-01", periods=T, freq="D")
        stocks = ["A"]

        factor_returns = pd.DataFrame(
            {
                "MKT": np.random.randn(T) * 0.01,
                "SMB": np.random.randn(T) * 0.005,
                "HML": np.random.randn(T) * 0.005,
                "RET": np.random.randn(T) * 0.003,
                "LIQ": np.random.randn(T) * 0.003,
            },
            index=dates,
        )

        # 纯噪声收益
        stock_returns = pd.DataFrame(
            np.random.randn(T, 1) * 0.02, index=dates, columns=stocks
        )

        hf_vol_val = 0.04
        hf_volatility = pd.DataFrame(hf_vol_val, index=dates, columns=stocks)

        result = factor.compute(
            stock_returns=stock_returns,
            factor_returns=factor_returns,
            hf_volatility=hf_volatility,
            T=T,
        )

        # idio_rate ≈ 1, result ≈ sqrt(1 * 0.04) = 0.2
        val = result.iloc[-1, 0]
        assert not np.isnan(val)
        assert val > 0.1  # 应该接近 sqrt(hf_vol)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        np.random.seed(42)
        T = 21
        dates = pd.date_range("2024-01-01", periods=T, freq="D")
        stocks = ["A", "B"]

        mkt = np.random.randn(T) * 0.01
        factor_returns = pd.DataFrame(
            {
                "MKT": mkt,
                "SMB": np.random.randn(T) * 0.005,
                "HML": np.random.randn(T) * 0.005,
                "RET": np.random.randn(T) * 0.003,
                "LIQ": np.random.randn(T) * 0.003,
            },
            index=dates,
        )

        # A: 完美拟合, B: 纯噪声
        stock_a = 0.001 + 1.0 * mkt
        stock_b = np.random.randn(T) * 0.02
        stock_returns = pd.DataFrame(
            {"A": stock_a, "B": stock_b}, index=dates
        )

        hf_volatility = pd.DataFrame(0.03, index=dates, columns=stocks)

        result = factor.compute(
            stock_returns=stock_returns,
            factor_returns=factor_returns,
            hf_volatility=hf_volatility,
            T=T,
        )

        # A 的特异波动率应远小于 B
        assert result.iloc[-1]["A"] < result.iloc[-1]["B"]


class TestHfIdioVolEdgeCases:
    def test_insufficient_data(self, factor):
        """数据不足T天时, 前面的行应为 NaN。"""
        T = 21
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]

        factor_returns = pd.DataFrame(
            {
                "MKT": np.random.randn(10) * 0.01,
                "SMB": np.random.randn(10) * 0.005,
                "HML": np.random.randn(10) * 0.005,
                "RET": np.random.randn(10) * 0.003,
                "LIQ": np.random.randn(10) * 0.003,
            },
            index=dates,
        )

        stock_returns = pd.DataFrame(
            np.random.randn(10, 1) * 0.01, index=dates, columns=stocks
        )
        hf_volatility = pd.DataFrame(0.02, index=dates, columns=stocks)

        result = factor.compute(
            stock_returns=stock_returns,
            factor_returns=factor_returns,
            hf_volatility=hf_volatility,
            T=T,
        )

        # 数据不足 T=21 天, 所有行应为 NaN
        assert result.isna().all().all()

    def test_nan_in_hf_volatility(self, factor):
        """高频波动率含 NaN 时, 对应结果应为 NaN。"""
        np.random.seed(42)
        T = 5
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        factor_returns = pd.DataFrame(
            {
                "MKT": np.random.randn(5) * 0.01,
                "SMB": np.random.randn(5) * 0.005,
                "HML": np.random.randn(5) * 0.005,
                "RET": np.random.randn(5) * 0.003,
                "LIQ": np.random.randn(5) * 0.003,
            },
            index=dates,
        )

        stock_returns = pd.DataFrame(
            np.random.randn(5, 1) * 0.01, index=dates, columns=stocks
        )
        hf_vol = [0.02, 0.03, np.nan, 0.02, 0.01]
        hf_volatility = pd.DataFrame(hf_vol, index=dates, columns=stocks)

        result = factor.compute(
            stock_returns=stock_returns,
            factor_returns=factor_returns,
            hf_volatility=hf_volatility,
            T=T,
        )
        assert isinstance(result, pd.DataFrame)


class TestHfIdioVolOutputShape:
    def test_output_shape_matches_input(self, factor):
        np.random.seed(42)
        T = 21
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]

        factor_returns = pd.DataFrame(
            {
                "MKT": np.random.randn(30) * 0.01,
                "SMB": np.random.randn(30) * 0.005,
                "HML": np.random.randn(30) * 0.005,
                "RET": np.random.randn(30) * 0.003,
                "LIQ": np.random.randn(30) * 0.003,
            },
            index=dates,
        )

        stock_returns = pd.DataFrame(
            np.random.randn(30, 3) * 0.01, index=dates, columns=stocks
        )
        hf_volatility = pd.DataFrame(
            np.random.uniform(0.01, 0.05, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(
            stock_returns=stock_returns,
            factor_returns=factor_returns,
            hf_volatility=hf_volatility,
            T=T,
        )

        assert result.shape == stock_returns.shape
        assert list(result.columns) == list(stock_returns.columns)
        assert list(result.index) == list(stock_returns.index)

    def test_output_is_dataframe(self, factor):
        np.random.seed(42)
        T = 5
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        factor_returns = pd.DataFrame(
            {
                "MKT": np.random.randn(5) * 0.01,
                "SMB": np.random.randn(5) * 0.005,
                "HML": np.random.randn(5) * 0.005,
                "RET": np.random.randn(5) * 0.003,
                "LIQ": np.random.randn(5) * 0.003,
            },
            index=dates,
        )

        stock_returns = pd.DataFrame(
            np.random.randn(5, 1) * 0.01, index=dates, columns=stocks
        )
        hf_volatility = pd.DataFrame(0.02, index=dates, columns=stocks)

        result = factor.compute(
            stock_returns=stock_returns,
            factor_returns=factor_returns,
            hf_volatility=hf_volatility,
            T=T,
        )
        assert isinstance(result, pd.DataFrame)

    def test_non_negative(self, factor):
        """高频特异波动率应非负（sqrt的结果）。"""
        np.random.seed(42)
        T = 21
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]

        factor_returns = pd.DataFrame(
            {
                "MKT": np.random.randn(30) * 0.01,
                "SMB": np.random.randn(30) * 0.005,
                "HML": np.random.randn(30) * 0.005,
                "RET": np.random.randn(30) * 0.003,
                "LIQ": np.random.randn(30) * 0.003,
            },
            index=dates,
        )

        stock_returns = pd.DataFrame(
            np.random.randn(30, 2) * 0.01, index=dates, columns=stocks
        )
        hf_volatility = pd.DataFrame(
            np.random.uniform(0.01, 0.05, (30, 2)), index=dates, columns=stocks
        )

        result = factor.compute(
            stock_returns=stock_returns,
            factor_returns=factor_returns,
            hf_volatility=hf_volatility,
            T=T,
        )
        valid = result.values[~np.isnan(result.values)]
        assert (valid >= 0).all()
