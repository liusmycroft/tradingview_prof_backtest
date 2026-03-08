import numpy as np
import pandas as pd
import pytest

from factors.weighted_close_ratio import WeightedCloseRatioFactor


@pytest.fixture
def factor():
    return WeightedCloseRatioFactor()


class TestWeightedCloseRatioMetadata:
    def test_name(self, factor):
        assert factor.name == "WEIGHTED_CLOSE_RATIO"

    def test_category(self, factor):
        assert factor.category == "高频量价"

    def test_repr(self, factor):
        assert "WEIGHTED_CLOSE_RATIO" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "WEIGHTED_CLOSE_RATIO"
        assert meta["category"] == "高频量价"


class TestWeightedCloseRatioHandCalculated:
    def test_rolling_mean_T3(self, factor):
        """T=3 滚动均值手算验证。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        data = pd.DataFrame([1.01, 1.02, 0.99, 1.03, 0.98], index=dates, columns=stocks)

        result = factor.compute(vol_weighted_close=data, T=3)

        # 前 2 行应为 NaN
        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])

        # 第 3 行: mean(1.01, 1.02, 0.99) = 1.006666...
        expected_2 = (1.01 + 1.02 + 0.99) / 3.0
        assert result.iloc[2, 0] == pytest.approx(expected_2, rel=1e-10)

        # 第 4 行: mean(1.02, 0.99, 1.03) = 1.013333...
        expected_3 = (1.02 + 0.99 + 1.03) / 3.0
        assert result.iloc[3, 0] == pytest.approx(expected_3, rel=1e-10)

        # 第 5 行: mean(0.99, 1.03, 0.98) = 1.0
        expected_4 = (0.99 + 1.03 + 0.98) / 3.0
        assert result.iloc[4, 0] == pytest.approx(expected_4, rel=1e-10)

    def test_constant_values(self, factor):
        """常数输入时，滚动均值等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        data = pd.DataFrame([1.05] * 5, index=dates, columns=stocks)

        result = factor.compute(vol_weighted_close=data, T=3)
        assert result.iloc[2, 0] == pytest.approx(1.05, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(1.05, rel=1e-10)


class TestWeightedCloseRatioEdgeCases:
    def test_nan_propagation(self, factor):
        """含 NaN 时，滚动窗口内有 NaN 则结果为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        data = pd.DataFrame([1.0, np.nan, 1.0, 1.0, 1.0], index=dates, columns=stocks)

        result = factor.compute(vol_weighted_close=data, T=3)
        # 窗口 [0,1,2] 含 NaN -> NaN
        assert np.isnan(result.iloc[2, 0])
        # 窗口 [1,2,3] 含 NaN -> NaN
        assert np.isnan(result.iloc[3, 0])
        # 窗口 [2,3,4] 无 NaN -> 有值
        assert not np.isnan(result.iloc[4, 0])

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        data = pd.DataFrame(
            np.random.uniform(0.98, 1.02, (30, 3)), index=dates, columns=stocks
        )
        result = factor.compute(vol_weighted_close=data, T=20)
        assert result.shape == data.shape
        assert list(result.columns) == list(data.columns)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(np.ones(10), index=dates, columns=stocks)
        T = 5
        result = factor.compute(vol_weighted_close=data, T=T)
        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
