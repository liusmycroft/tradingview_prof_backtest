import numpy as np
import pandas as pd
import pytest

from factors.weighted_lower_shadow_freq import WeightedLowerShadowFreqFactor


@pytest.fixture
def factor():
    return WeightedLowerShadowFreqFactor()


class TestWeightedLowerShadowFreqMetadata:
    def test_name(self, factor):
        assert factor.name == "WEIGHTED_LOWER_SHADOW_FREQ"

    def test_category(self, factor):
        assert factor.category == "量价因子改进"

    def test_repr(self, factor):
        assert "WEIGHTED_LOWER_SHADOW_FREQ" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "WEIGHTED_LOWER_SHADOW_FREQ"


class TestWeightedLowerShadowFreqCompute:
    def test_no_lower_shadow(self, factor):
        """无下影线时因子应为 0。"""
        dates = pd.date_range("2024-01-01", periods=40, freq="D")
        stocks = ["A"]
        # Low == min(Open, Close) => 下影线 = 0
        open_p = pd.DataFrame(10.0, index=dates, columns=stocks)
        close_p = pd.DataFrame(11.0, index=dates, columns=stocks)
        low_p = pd.DataFrame(10.0, index=dates, columns=stocks)
        prev_close = pd.DataFrame(10.0, index=dates, columns=stocks)

        result = factor.compute(
            open_price=open_p, close_price=close_p,
            low_price=low_p, prev_close=prev_close, M=40,
        )
        assert result.iloc[-1, 0] == pytest.approx(0.0, abs=1e-10)

    def test_all_lower_shadow(self, factor):
        """所有日都有长下影线时因子应为正。"""
        dates = pd.date_range("2024-01-01", periods=40, freq="D")
        stocks = ["A"]
        open_p = pd.DataFrame(10.0, index=dates, columns=stocks)
        close_p = pd.DataFrame(10.0, index=dates, columns=stocks)
        low_p = pd.DataFrame(9.0, index=dates, columns=stocks)  # 下影线 = 1/10 = 10%
        prev_close = pd.DataFrame(10.0, index=dates, columns=stocks)

        result = factor.compute(
            open_price=open_p, close_price=close_p,
            low_price=low_p, prev_close=prev_close, M=40, u=0.01,
        )
        assert result.iloc[-1, 0] > 0

    def test_leading_nan(self, factor):
        """前 M-1 行应为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        stocks = ["A"]
        open_p = pd.DataFrame(10.0, index=dates, columns=stocks)
        close_p = pd.DataFrame(10.0, index=dates, columns=stocks)
        low_p = pd.DataFrame(9.5, index=dates, columns=stocks)
        prev_close = pd.DataFrame(10.0, index=dates, columns=stocks)

        result = factor.compute(
            open_price=open_p, close_price=close_p,
            low_price=low_p, prev_close=prev_close, M=40,
        )
        assert result.iloc[:39]["A"].isna().all()
        assert result.iloc[39:]["A"].notna().all()

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        stocks = ["A", "B"]
        open_p = pd.DataFrame(np.random.uniform(9, 11, (50, 2)), index=dates, columns=stocks)
        close_p = pd.DataFrame(np.random.uniform(9, 11, (50, 2)), index=dates, columns=stocks)
        low_p = pd.DataFrame(np.random.uniform(8, 10, (50, 2)), index=dates, columns=stocks)
        prev_close = pd.DataFrame(np.random.uniform(9, 11, (50, 2)), index=dates, columns=stocks)

        result = factor.compute(
            open_price=open_p, close_price=close_p,
            low_price=low_p, prev_close=prev_close, M=40,
        )
        assert result.shape == open_p.shape
        assert isinstance(result, pd.DataFrame)
