import numpy as np
import pandas as pd
import pytest

from factors.tidal_price_velocity import TidalPriceVelocityFactor


@pytest.fixture
def factor():
    return TidalPriceVelocityFactor()


class TestTidalPriceVelocityMetadata:
    def test_name(self, factor):
        assert factor.name == "TIDAL_PRICE_VELOCITY"

    def test_category(self, factor):
        assert factor.category == "高频量价相关性"

    def test_repr(self, factor):
        assert "TIDAL_PRICE_VELOCITY" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "TIDAL_PRICE_VELOCITY"
        assert meta["category"] == "高频量价相关性"


class TestTidalPriceVelocityCompute:
    def test_constant_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        daily = pd.DataFrame(0.05, index=dates, columns=["A"])

        result = factor.compute(daily_tidal_velocity=daily, T=20)
        np.testing.assert_array_almost_equal(result["A"].values, 0.05)

    def test_rolling_mean(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        daily = pd.DataFrame([0.1, 0.2, 0.3, 0.4, 0.5], index=dates, columns=["A"])

        result = factor.compute(daily_tidal_velocity=daily, T=3)
        assert result.iloc[0, 0] == pytest.approx(0.1)
        assert result.iloc[1, 0] == pytest.approx(0.15)
        assert result.iloc[2, 0] == pytest.approx(0.2)
        assert result.iloc[3, 0] == pytest.approx(0.3)
        assert result.iloc[4, 0] == pytest.approx(0.4)

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        daily = pd.DataFrame(np.random.randn(30, 2), index=dates, columns=stocks)

        result = factor.compute(daily_tidal_velocity=daily, T=20)
        assert result.shape == daily.shape


class TestTidalPriceVelocityDaily:
    def test_compute_daily_basic(self):
        """验证单日计算逻辑。"""
        n = 233
        # 构造一个中间有成交量高峰的序列
        vol = pd.Series(np.ones(n) * 100, index=range(n))
        vol.iloc[100:120] = 1000  # 中间放量
        close = pd.Series(np.linspace(10, 11, n), index=range(n))

        result = TidalPriceVelocityFactor.compute_daily(vol, close, neighborhood=4)
        assert isinstance(result, float)
        assert not np.isnan(result)

    def test_compute_daily_too_short(self):
        """数据太短时返回NaN。"""
        vol = pd.Series([100, 200, 300], index=range(3))
        close = pd.Series([10, 11, 12], index=range(3))

        result = TidalPriceVelocityFactor.compute_daily(vol, close, neighborhood=4)
        assert np.isnan(result)


class TestTidalPriceVelocityEdgeCases:
    def test_single_value(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        daily = pd.DataFrame([0.05], index=dates, columns=["A"])

        result = factor.compute(daily_tidal_velocity=daily, T=20)
        assert result.iloc[0, 0] == pytest.approx(0.05)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        daily = pd.DataFrame(np.nan, index=dates, columns=["A"])

        result = factor.compute(daily_tidal_velocity=daily, T=5)
        assert result.isna().all().all()
