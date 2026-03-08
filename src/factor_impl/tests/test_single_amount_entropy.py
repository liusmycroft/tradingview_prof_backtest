import numpy as np
import pandas as pd
import pytest

from factors.single_amount_entropy import SingleAmountEntropyFactor


@pytest.fixture
def factor():
    return SingleAmountEntropyFactor()


class TestSingleAmountEntropyMetadata:
    def test_name(self, factor):
        assert factor.name == "SINGLE_AMOUNT_ENTROPY"

    def test_category(self, factor):
        assert factor.category == "高频量价相关性"

    def test_repr(self, factor):
        assert "SINGLE_AMOUNT_ENTROPY" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "SINGLE_AMOUNT_ENTROPY"
        assert meta["category"] == "高频量价相关性"


class TestSingleAmountEntropyCompute:
    def test_uniform_distribution(self, factor):
        """均匀分布时，p_i = (vol_i*close_i)/(VOL*CLOSE)。
        4个相同时段: p_i = (100*10)/(400*40) = 1/16
        H = -4 * (1/16) * ln(1/16) = 0.25 * ln(16) = ln(2)
        """
        dates = ["2024-01-01"]
        minutes = list(range(4))
        idx = pd.MultiIndex.from_product([dates, minutes])
        close = pd.DataFrame({"A": [10.0, 10.0, 10.0, 10.0]}, index=idx)
        volume = pd.DataFrame({"A": [100.0, 100.0, 100.0, 100.0]}, index=idx)

        result = factor.compute(minute_close=close, minute_volume=volume)
        # p_i = 1/16, H = -4*(1/16)*ln(1/16) = 0.25*ln(16) = ln(2)
        expected = np.log(2)
        assert result.iloc[0, 0] == pytest.approx(expected, rel=1e-6)

    def test_concentrated_distribution(self, factor):
        """成交集中在一个时段时熵较低。"""
        dates = ["2024-01-01"]
        minutes = list(range(4))
        idx = pd.MultiIndex.from_product([dates, minutes])
        close = pd.DataFrame({"A": [10.0, 10.0, 10.0, 10.0]}, index=idx)
        volume = pd.DataFrame({"A": [1000.0, 1.0, 1.0, 1.0]}, index=idx)

        result = factor.compute(minute_close=close, minute_volume=volume)
        # 熵应远小于 ln(4)
        assert result.iloc[0, 0] < np.log(4) * 0.5

    def test_precomputed_passthrough(self, factor):
        """预计算模式直接返回。"""
        dates = pd.date_range("2024-01-01", periods=3)
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=dates)
        result = factor.compute(minute_close=df, minute_volume=df)
        pd.testing.assert_frame_equal(result, df)

    def test_two_stocks_independent(self, factor):
        """两只股票独立计算。"""
        dates = ["2024-01-01"]
        minutes = list(range(4))
        idx = pd.MultiIndex.from_product([dates, minutes])
        close = pd.DataFrame({
            "A": [10.0, 10.0, 10.0, 10.0],
            "B": [20.0, 20.0, 20.0, 20.0],
        }, index=idx)
        volume = pd.DataFrame({
            "A": [100.0, 100.0, 100.0, 100.0],
            "B": [100.0, 100.0, 100.0, 100.0],
        }, index=idx)

        result = factor.compute(minute_close=close, minute_volume=volume)
        # 均匀分布，两只股票熵相同
        assert result.iloc[0, 0] == pytest.approx(result.iloc[0, 1], rel=1e-6)


class TestSingleAmountEntropyEdgeCases:
    def test_zero_volume(self, factor):
        """全零成交量时结果为 NaN。"""
        dates = ["2024-01-01"]
        minutes = list(range(4))
        idx = pd.MultiIndex.from_product([dates, minutes])
        close = pd.DataFrame({"A": [10.0, 10.0, 10.0, 10.0]}, index=idx)
        volume = pd.DataFrame({"A": [0.0, 0.0, 0.0, 0.0]}, index=idx)

        result = factor.compute(minute_close=close, minute_volume=volume)
        assert np.isnan(result.iloc[0, 0])

    def test_nan_in_input(self, factor):
        """输入含 NaN 时不抛异常。"""
        dates = ["2024-01-01"]
        minutes = list(range(4))
        idx = pd.MultiIndex.from_product([dates, minutes])
        close = pd.DataFrame({"A": [10.0, np.nan, 10.0, 10.0]}, index=idx)
        volume = pd.DataFrame({"A": [100.0, 100.0, 100.0, 100.0]}, index=idx)

        result = factor.compute(minute_close=close, minute_volume=volume)
        assert isinstance(result, pd.DataFrame)


class TestSingleAmountEntropyOutputShape:
    def test_output_is_dataframe(self, factor):
        dates = ["2024-01-01"]
        minutes = list(range(4))
        idx = pd.MultiIndex.from_product([dates, minutes])
        close = pd.DataFrame({"A": [10.0] * 4}, index=idx)
        volume = pd.DataFrame({"A": [100.0] * 4}, index=idx)

        result = factor.compute(minute_close=close, minute_volume=volume)
        assert isinstance(result, pd.DataFrame)

    def test_output_shape(self, factor):
        dates = ["2024-01-01", "2024-01-02"]
        minutes = list(range(4))
        idx = pd.MultiIndex.from_product([dates, minutes])
        close = pd.DataFrame({"A": [10.0] * 8, "B": [20.0] * 8}, index=idx)
        volume = pd.DataFrame({"A": [100.0] * 8, "B": [50.0] * 8}, index=idx)

        result = factor.compute(minute_close=close, minute_volume=volume)
        assert result.shape == (2, 2)
