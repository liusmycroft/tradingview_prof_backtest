import numpy as np
import pandas as pd
import pytest

from factors.attention_decay_panic import AttentionDecayPanicFactor


@pytest.fixture
def factor():
    return AttentionDecayPanicFactor()


class TestAttentionDecayPanicMetadata:
    def test_name(self, factor):
        assert factor.name == "ATTENTION_DECAY_PANIC"

    def test_category(self, factor):
        assert factor.category == "高频动量反转"

    def test_repr(self, factor):
        assert "ATTENTION_DECAY_PANIC" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "ATTENTION_DECAY_PANIC"
        assert meta["category"] == "高频动量反转"


class TestAttentionDecayPanicHandCalculated:
    def test_constant_input(self, factor):
        """常数输入时, mean=常数, std=0, factor = 常数/2."""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily_decay_panic = pd.DataFrame(0.01, index=dates, columns=stocks)

        result = factor.compute(daily_decay_panic=daily_decay_panic, T=20)

        valid = result.dropna()
        # mean=0.01, std=0 => factor = (0.01 + 0) / 2 = 0.005
        np.testing.assert_array_almost_equal(valid["A"].values, 0.005)

    def test_manual_T3(self, factor):
        """T=3 手算验证."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        vals = [0.01, 0.02, 0.03, 0.04, 0.05]
        daily_decay_panic = pd.DataFrame(vals, index=dates, columns=stocks)

        result = factor.compute(daily_decay_panic=daily_decay_panic, T=3)

        # min_periods=3, first 2 rows NaN
        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])

        # index 2: window [0.01, 0.02, 0.03]
        mean_2 = np.mean([0.01, 0.02, 0.03])
        std_2 = np.std([0.01, 0.02, 0.03], ddof=1)
        expected_2 = (mean_2 + std_2) / 2
        assert result.iloc[2, 0] == pytest.approx(expected_2, rel=1e-6)

        # index 3: window [0.02, 0.03, 0.04]
        mean_3 = np.mean([0.02, 0.03, 0.04])
        std_3 = np.std([0.02, 0.03, 0.04], ddof=1)
        expected_3 = (mean_3 + std_3) / 2
        assert result.iloc[3, 0] == pytest.approx(expected_3, rel=1e-6)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        daily_decay_panic = pd.DataFrame(
            {"A": [0.01, 0.01, 0.01, 0.01, 0.01],
             "B": [0.05, 0.05, 0.05, 0.05, 0.05]},
            index=dates,
        )

        result = factor.compute(daily_decay_panic=daily_decay_panic, T=3)

        # Both constant: mean=val, std=0, factor=val/2
        assert result.iloc[2, 0] == pytest.approx(0.005, rel=1e-6)
        assert result.iloc[2, 1] == pytest.approx(0.025, rel=1e-6)


class TestAttentionDecayPanicEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        values = np.ones(10) * 0.01
        values[3] = np.nan
        daily_decay_panic = pd.DataFrame(values, index=dates, columns=stocks)

        result = factor.compute(daily_decay_panic=daily_decay_panic, T=5)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily_decay_panic = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_decay_panic=daily_decay_panic, T=5)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily_decay_panic = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_decay_panic=daily_decay_panic, T=5)
        valid = result.dropna()
        # mean=0, std=0 => factor=0
        np.testing.assert_array_almost_equal(valid["A"].values, 0.0)

    def test_first_T_minus_1_nan(self, factor):
        """前 T-1 行应为 NaN."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily_decay_panic = pd.DataFrame(0.01, index=dates, columns=stocks)
        T = 5

        result = factor.compute(daily_decay_panic=daily_decay_panic, T=T)
        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()


class TestAttentionDecayPanicOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily_decay_panic = pd.DataFrame(
            np.random.uniform(0, 0.05, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_decay_panic=daily_decay_panic, T=20)
        assert result.shape == daily_decay_panic.shape
        assert list(result.columns) == list(daily_decay_panic.columns)
        assert list(result.index) == list(daily_decay_panic.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_decay_panic = pd.DataFrame(0.01, index=dates, columns=stocks)

        result = factor.compute(daily_decay_panic=daily_decay_panic, T=3)
        assert isinstance(result, pd.DataFrame)
