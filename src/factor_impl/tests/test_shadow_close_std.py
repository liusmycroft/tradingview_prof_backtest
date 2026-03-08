import numpy as np
import pandas as pd
import pytest

from factors.shadow_close_std import ShadowCloseStdFactor


@pytest.fixture
def factor():
    return ShadowCloseStdFactor()


class TestShadowCloseStdMetadata:
    def test_name(self, factor):
        assert factor.name == "SHADOW_CLOSE_STD"

    def test_category(self, factor):
        assert factor.category == "量价因子改进"

    def test_repr(self, factor):
        assert "SHADOW_CLOSE_STD" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "SHADOW_CLOSE_STD"
        assert meta["category"] == "量价因子改进"


class TestShadowCloseStdHandCalculated:
    def test_constant_input_zero_std(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        data = pd.DataFrame(0.05, index=dates, columns=["A"])
        result = factor.compute(daily_shadow_ratio=data, T=20)
        assert result.iloc[19, 0] == pytest.approx(0.0, abs=1e-10)

    def test_known_std_T3(self, factor):
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        data = pd.DataFrame([0.01, 0.02, 0.03], index=dates, columns=["A"])
        result = factor.compute(daily_shadow_ratio=data, T=3)
        expected_std = np.std([0.01, 0.02, 0.03], ddof=1)
        assert result.iloc[2, 0] == pytest.approx(expected_std)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame(
            {"A": [0.01] * 5, "B": [0.01, 0.02, 0.03, 0.04, 0.05]}, index=dates
        )
        result = factor.compute(daily_shadow_ratio=data, T=3)
        assert result.iloc[2, 0] == pytest.approx(0.0, abs=1e-10)
        assert result.iloc[2, 1] > 0


class TestShadowCloseStdEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame([0.01, np.nan, 0.03, 0.04, 0.05], index=dates, columns=["A"])
        result = factor.compute(daily_shadow_ratio=data, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(np.nan, index=dates, columns=["A"])
        result = factor.compute(daily_shadow_ratio=data, T=5)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame(0.0, index=dates, columns=["A"])
        result = factor.compute(daily_shadow_ratio=data, T=3)
        assert result.iloc[2, 0] == pytest.approx(0.0, abs=1e-15)


class TestShadowCloseStdOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        data = pd.DataFrame(
            np.random.uniform(0.01, 0.1, (30, 3)), index=dates, columns=stocks
        )
        result = factor.compute(daily_shadow_ratio=data, T=20)
        assert result.shape == data.shape

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame([0.01, 0.02, 0.03, 0.04, 0.05], index=dates, columns=["A"])
        result = factor.compute(daily_shadow_ratio=data, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        data = pd.DataFrame(
            np.random.uniform(0.01, 0.1, (25, 2)), index=dates, columns=["A", "B"]
        )
        result = factor.compute(daily_shadow_ratio=data, T=20)
        assert result.iloc[:19].isna().all().all()
        assert result.iloc[19:].notna().all().all()
