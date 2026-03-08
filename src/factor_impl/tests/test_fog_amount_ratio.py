import numpy as np
import pandas as pd
import pytest

from factors.fog_amount_ratio import FogAmountRatioFactor


@pytest.fixture
def factor():
    return FogAmountRatioFactor()


class TestFogAmountRatioMetadata:
    def test_name(self, factor):
        assert factor.name == "FOG_AMOUNT_RATIO"

    def test_category(self, factor):
        assert factor.category == "高频成交分布"

    def test_repr(self, factor):
        assert "FOG_AMOUNT_RATIO" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "FOG_AMOUNT_RATIO"
        assert meta["category"] == "高频成交分布"


class TestFogAmountRatioCompute:
    def test_known_values(self, factor):
        """手算验证: 0.5 * mean + 0.5 * std"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        fog_mean = pd.DataFrame({"A": [1.0, 1.2, 0.8]}, index=dates)
        fog_std = pd.DataFrame({"A": [0.2, 0.4, 0.1]}, index=dates)

        result = factor.compute(daily_fog_amt_mean=fog_mean, daily_fog_amt_std=fog_std)

        assert result.iloc[0, 0] == pytest.approx(0.6)
        assert result.iloc[1, 0] == pytest.approx(0.8)
        assert result.iloc[2, 0] == pytest.approx(0.45)

    def test_equal_inputs(self, factor):
        """mean == std 时, 结果 = mean。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=dates)

        result = factor.compute(daily_fog_amt_mean=df, daily_fog_amt_std=df)
        pd.testing.assert_frame_equal(result, df)

    def test_zero_std(self, factor):
        """std 全零时, 结果 = 0.5 * mean。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        fog_mean = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=dates)
        fog_std = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=dates)

        result = factor.compute(daily_fog_amt_mean=fog_mean, daily_fog_amt_std=fog_std)
        expected = pd.DataFrame({"A": [0.5, 1.0, 1.5]}, index=dates)
        pd.testing.assert_frame_equal(result, expected)

    def test_symmetry(self, factor):
        """交换 mean 和 std 应得到相同结果（等权）。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        np.random.seed(42)
        fog_mean = pd.DataFrame({"A": np.random.rand(5)}, index=dates)
        fog_std = pd.DataFrame({"A": np.random.rand(5)}, index=dates)

        result1 = factor.compute(daily_fog_amt_mean=fog_mean, daily_fog_amt_std=fog_std)
        result2 = factor.compute(daily_fog_amt_mean=fog_std, daily_fog_amt_std=fog_mean)
        pd.testing.assert_frame_equal(result1, result2)


class TestFogAmountRatioEdgeCases:
    def test_nan_propagation(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=3)
        fog_mean = pd.DataFrame({"A": [1.0, np.nan, 3.0]}, index=dates)
        fog_std = pd.DataFrame({"A": [0.5, 0.5, np.nan]}, index=dates)

        result = factor.compute(daily_fog_amt_mean=fog_mean, daily_fog_amt_std=fog_std)
        assert not np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert np.isnan(result.iloc[2, 0])

    def test_all_zeros(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=3)
        zeros = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=dates)

        result = factor.compute(daily_fog_amt_mean=zeros, daily_fog_amt_std=zeros)
        expected = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=dates)
        pd.testing.assert_frame_equal(result, expected)


class TestFogAmountRatioOutputShape:
    def test_output_is_dataframe(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=5)
        df = pd.DataFrame({"A": np.random.rand(5)}, index=dates)
        result = factor.compute(daily_fog_amt_mean=df, daily_fog_amt_std=df)
        assert isinstance(result, pd.DataFrame)

    def test_output_shape(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=10)
        stocks = ["A", "B", "C"]
        fog_mean = pd.DataFrame(np.random.rand(10, 3), index=dates, columns=stocks)
        fog_std = pd.DataFrame(np.random.rand(10, 3), index=dates, columns=stocks)

        result = factor.compute(daily_fog_amt_mean=fog_mean, daily_fog_amt_std=fog_std)
        assert result.shape == (10, 3)
