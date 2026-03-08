import numpy as np
import pandas as pd
import pytest

from factors.weighted_upper_shadow_freq import WeightedUpperShadowFreqFactor


@pytest.fixture
def factor():
    return WeightedUpperShadowFreqFactor()


class TestWeightedUpperShadowFreqMetadata:
    def test_name(self, factor):
        assert factor.name == "WEIGHTED_UPPER_SHADOW_FREQ"

    def test_category(self, factor):
        assert factor.category == "量价因子改进"

    def test_repr(self, factor):
        assert "WEIGHTED_UPPER_SHADOW_FREQ" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "WEIGHTED_UPPER_SHADOW_FREQ"
        assert meta["category"] == "量价因子改进"


class TestWeightedUpperShadowFreqCompute:
    def test_no_upper_shadow(self, factor):
        """当最高价等于max(open,close)时，上影线为0，因子应为0。"""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        stocks = ["A"]
        high = pd.DataFrame(10.0, index=dates, columns=stocks)
        open_price = pd.DataFrame(10.0, index=dates, columns=stocks)
        close = pd.DataFrame(9.0, index=dates, columns=stocks)
        prev_close = pd.DataFrame(10.0, index=dates, columns=stocks)

        result = factor.compute(
            high=high, open_price=open_price, close=close,
            prev_close=prev_close, M=40, u=0.01
        )
        # 上影线 = (10 - max(10, 9)) / 10 = 0, 不超过阈值
        assert result.iloc[-1, 0] == pytest.approx(0.0, abs=1e-10)

    def test_all_upper_shadow(self, factor):
        """所有天都有大上影线时，因子应>0且稳定。"""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        stocks = ["A"]
        high = pd.DataFrame(12.0, index=dates, columns=stocks)
        open_price = pd.DataFrame(10.0, index=dates, columns=stocks)
        close = pd.DataFrame(10.0, index=dates, columns=stocks)
        prev_close = pd.DataFrame(10.0, index=dates, columns=stocks)

        result = factor.compute(
            high=high, open_price=open_price, close=close,
            prev_close=prev_close, M=40, u=0.01
        )
        # 上影线 = (12 - 10) / 10 = 0.2 > 0.01, 所有天indicator=1
        # 加权平均 = sum(w_j * 1) / M, 由于衰减权重 w_j < 1, 结果 < 1
        assert result.iloc[-1, 0] > 0.0
        # 最后几行应稳定（窗口已满）
        assert result.iloc[-1, 0] == pytest.approx(result.iloc[-2, 0], rel=1e-6)

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        stocks = ["A", "B"]
        high = pd.DataFrame(np.random.uniform(10, 12, (50, 2)), index=dates, columns=stocks)
        open_price = pd.DataFrame(10.0, index=dates, columns=stocks)
        close = pd.DataFrame(10.0, index=dates, columns=stocks)
        prev_close = pd.DataFrame(10.0, index=dates, columns=stocks)

        result = factor.compute(
            high=high, open_price=open_price, close=close,
            prev_close=prev_close, M=40, u=0.01
        )
        assert result.shape == high.shape

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        stocks = ["A", "B"]
        # A: 无上影线, B: 有上影线
        high = pd.DataFrame({"A": [10.0]*50, "B": [12.0]*50}, index=dates)
        open_price = pd.DataFrame({"A": [10.0]*50, "B": [10.0]*50}, index=dates)
        close = pd.DataFrame({"A": [10.0]*50, "B": [10.0]*50}, index=dates)
        prev_close = pd.DataFrame({"A": [10.0]*50, "B": [10.0]*50}, index=dates)

        result = factor.compute(
            high=high, open_price=open_price, close=close,
            prev_close=prev_close, M=40, u=0.01
        )
        assert result["A"].iloc[-1] == pytest.approx(0.0, abs=1e-10)
        assert result["B"].iloc[-1] > 0.0


class TestWeightedUpperShadowFreqEdgeCases:
    def test_single_row(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A"]
        high = pd.DataFrame([12.0], index=dates, columns=stocks)
        open_price = pd.DataFrame([10.0], index=dates, columns=stocks)
        close = pd.DataFrame([10.0], index=dates, columns=stocks)
        prev_close = pd.DataFrame([10.0], index=dates, columns=stocks)

        result = factor.compute(
            high=high, open_price=open_price, close=close,
            prev_close=prev_close, M=40, u=0.01
        )
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (1, 1)
