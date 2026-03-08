import numpy as np
import pandas as pd
import pytest

from factors.attention_market_volume import AttentionMarketVolumeFactor


@pytest.fixture
def factor():
    return AttentionMarketVolumeFactor()


class TestAttentionMarketVolumeMetadata:
    def test_name(self, factor):
        assert factor.name == "ATTENTION_MARKET_VOLUME"

    def test_category(self, factor):
        assert factor.category == "行为金融注意力"

    def test_repr(self, factor):
        assert "ATTENTION_MARKET_VOLUME" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "ATTENTION_MARKET_VOLUME"
        assert meta["category"] == "行为金融注意力"


class TestAttentionMarketVolumeHandCalculated:
    def test_constant_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        data = pd.DataFrame(0.05, index=dates, columns=["A"])
        result = factor.compute(daily_attention_beta=data, T=20)
        np.testing.assert_array_almost_equal(result["A"].values, 0.05)

    def test_ema_manual_T3(self, factor):
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        data = pd.DataFrame([10.0, 20.0, 30.0, 40.0], index=dates, columns=["A"])
        result = factor.compute(daily_attention_beta=data, T=3)
        assert result.iloc[0, 0] == pytest.approx(10.0, rel=1e-6)
        assert result.iloc[1, 0] == pytest.approx(50 / 3, rel=1e-6)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame({"A": [0.01] * 10, "B": [0.05] * 10}, index=dates)
        result = factor.compute(daily_attention_beta=data, T=5)
        np.testing.assert_array_almost_equal(result["A"].values, 0.01)
        np.testing.assert_array_almost_equal(result["B"].values, 0.05)


class TestAttentionMarketVolumeEdgeCases:
    def test_single_value(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        data = pd.DataFrame([0.03], index=dates, columns=["A"])
        result = factor.compute(daily_attention_beta=data, T=20)
        assert result.iloc[0, 0] == pytest.approx(0.03, rel=1e-10)

    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        values = np.ones(10) * 0.05
        values[3] = np.nan
        data = pd.DataFrame(values, index=dates, columns=["A"])
        result = factor.compute(daily_attention_beta=data, T=5)
        assert isinstance(result, pd.DataFrame)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(np.nan, index=dates, columns=["A"])
        result = factor.compute(daily_attention_beta=data, T=5)
        assert result.isna().all().all()


class TestAttentionMarketVolumeOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        data = pd.DataFrame(
            np.random.uniform(0.01, 0.1, (30, 3)), index=dates, columns=stocks
        )
        result = factor.compute(daily_attention_beta=data, T=20)
        assert result.shape == data.shape

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame([0.01, 0.02, 0.03, 0.04, 0.05], index=dates, columns=["A"])
        result = factor.compute(daily_attention_beta=data, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_no_leading_nan_min_periods_1(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(
            np.random.uniform(0.01, 0.1, (10, 2)), index=dates, columns=["A", "B"]
        )
        result = factor.compute(daily_attention_beta=data, T=20)
        assert result.iloc[0].notna().all()
