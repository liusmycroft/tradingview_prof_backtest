import numpy as np
import pandas as pd
import pytest

from factors.attention_spillover import AttentionSpilloverFactor


@pytest.fixture
def factor():
    return AttentionSpilloverFactor()


class TestAttentionSpilloverMetadata:
    def test_name(self, factor):
        assert factor.name == "SPILL"

    def test_category(self, factor):
        assert factor.category == "行为金融-投资者注意力"

    def test_repr(self, factor):
        assert "SPILL" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "SPILL"
        assert meta["category"] == "行为金融-投资者注意力"


class TestAttentionSpilloverHandCalculated:
    def test_constant_spill(self, factor):
        """peer_avg - daily_attn 恒定时, EMA 等于该常数."""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        daily_attn = pd.DataFrame(0.01, index=dates, columns=stocks)
        peer_avg_attn = pd.DataFrame(0.03, index=dates, columns=stocks)

        result = factor.compute(daily_attn=daily_attn, peer_avg_attn=peer_avg_attn, T=20)

        # spill = 0.03 - 0.01 = 0.02, EMA of constant = constant
        np.testing.assert_array_almost_equal(result["A"].values, 0.02)

    def test_zero_spill(self, factor):
        """peer_avg == daily_attn 时, spill=0, EMA=0."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily_attn = pd.DataFrame(0.05, index=dates, columns=stocks)
        peer_avg_attn = pd.DataFrame(0.05, index=dates, columns=stocks)

        result = factor.compute(daily_attn=daily_attn, peer_avg_attn=peer_avg_attn, T=5)

        np.testing.assert_array_almost_equal(result["A"].values, 0.0, decimal=10)

    def test_ema_manual_T3(self, factor):
        """T=3 手动验证 EMA.

        spill = peer_avg - daily_attn = [2, 4, 6, 8]
        ewm(span=3, adjust=True) alpha = 2/(3+1) = 0.5
          ema_0 = 2.0
          ema_1 = (0.5*2 + 1.0*4) / (0.5+1.0) = 10/3
          ema_2 = (0.25*2 + 0.5*4 + 1.0*6) / (0.25+0.5+1.0) = 34/7
          ema_3 = (0.125*2 + 0.25*4 + 0.5*6 + 1.0*8) / (0.125+0.25+0.5+1.0) = 98/15
        """
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        stocks = ["A"]
        daily_attn = pd.DataFrame([0.0, 0.0, 0.0, 0.0], index=dates, columns=stocks)
        peer_avg_attn = pd.DataFrame([2.0, 4.0, 6.0, 8.0], index=dates, columns=stocks)

        result = factor.compute(daily_attn=daily_attn, peer_avg_attn=peer_avg_attn, T=3)

        assert result.iloc[0, 0] == pytest.approx(2.0, rel=1e-6)
        assert result.iloc[1, 0] == pytest.approx(10.0 / 3, rel=1e-6)
        assert result.iloc[2, 0] == pytest.approx(34.0 / 7, rel=1e-6)
        assert result.iloc[3, 0] == pytest.approx(98.0 / 15, rel=1e-6)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        daily_attn = pd.DataFrame(
            {"A": [0.01] * 10, "B": [0.05] * 10}, index=dates
        )
        peer_avg_attn = pd.DataFrame(
            {"A": [0.03] * 10, "B": [0.03] * 10}, index=dates
        )

        result = factor.compute(daily_attn=daily_attn, peer_avg_attn=peer_avg_attn, T=5)

        # A: spill=0.02, B: spill=-0.02
        np.testing.assert_array_almost_equal(result["A"].values, 0.02)
        np.testing.assert_array_almost_equal(result["B"].values, -0.02)


class TestAttentionSpilloverEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily_vals = np.ones(10) * 0.01
        daily_vals[3] = np.nan
        daily_attn = pd.DataFrame(daily_vals, index=dates, columns=stocks)
        peer_avg_attn = pd.DataFrame(0.03, index=dates, columns=stocks)

        result = factor.compute(daily_attn=daily_attn, peer_avg_attn=peer_avg_attn, T=5)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily_attn = pd.DataFrame(np.nan, index=dates, columns=stocks)
        peer_avg_attn = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_attn=daily_attn, peer_avg_attn=peer_avg_attn, T=5)
        assert result.isna().all().all()

    def test_no_leading_nan_min_periods_1(self, factor):
        """min_periods=1, 第一行就有值."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily_attn = pd.DataFrame(0.01, index=dates, columns=stocks)
        peer_avg_attn = pd.DataFrame(0.03, index=dates, columns=stocks)

        result = factor.compute(daily_attn=daily_attn, peer_avg_attn=peer_avg_attn, T=20)
        assert result.iloc[0].notna().all()
        assert result.notna().all().all()


class TestAttentionSpilloverOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily_attn = pd.DataFrame(
            np.random.uniform(0, 0.1, (30, 3)), index=dates, columns=stocks
        )
        peer_avg_attn = pd.DataFrame(
            np.random.uniform(0, 0.1, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_attn=daily_attn, peer_avg_attn=peer_avg_attn, T=20)
        assert result.shape == daily_attn.shape
        assert list(result.columns) == list(daily_attn.columns)
        assert list(result.index) == list(daily_attn.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_attn = pd.DataFrame(0.01, index=dates, columns=stocks)
        peer_avg_attn = pd.DataFrame(0.03, index=dates, columns=stocks)

        result = factor.compute(daily_attn=daily_attn, peer_avg_attn=peer_avg_attn, T=3)
        assert isinstance(result, pd.DataFrame)
