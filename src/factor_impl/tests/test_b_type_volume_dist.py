import numpy as np
import pandas as pd
import pytest

from factors.b_type_volume_dist import BTypeVolumeDistFactor


@pytest.fixture
def factor():
    return BTypeVolumeDistFactor()


class TestBTypeVolumeDistMetadata:
    def test_name(self, factor):
        assert factor.name == "B_TYPE_VOLUME_DIST"

    def test_category(self, factor):
        assert factor.category == "高频成交分布"

    def test_repr(self, factor):
        assert "B_TYPE_VOLUME_DIST" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "B_TYPE_VOLUME_DIST"
        assert meta["category"] == "高频成交分布"


class TestBTypeVolumeDistHandCalculated:
    def test_constant_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        data = pd.DataFrame(0.05, index=dates, columns=["A"])
        result = factor.compute(daily_vsa_high2min=data, T=20)
        assert result.iloc[19, 0] == pytest.approx(0.05)

    def test_rolling_mean_T3(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame([0.01, 0.02, 0.03, 0.04, 0.05], index=dates, columns=["A"])
        result = factor.compute(daily_vsa_high2min=data, T=3)
        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(0.02)
        assert result.iloc[3, 0] == pytest.approx(0.03)
        assert result.iloc[4, 0] == pytest.approx(0.04)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame({"A": [0.01] * 5, "B": [0.05] * 5}, index=dates)
        result = factor.compute(daily_vsa_high2min=data, T=3)
        assert result.iloc[2, 0] == pytest.approx(0.01)
        assert result.iloc[2, 1] == pytest.approx(0.05)


class TestBTypeVolumeDistEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame([0.01, np.nan, 0.03, 0.04, 0.05], index=dates, columns=["A"])
        result = factor.compute(daily_vsa_high2min=data, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(np.nan, index=dates, columns=["A"])
        result = factor.compute(daily_vsa_high2min=data, T=5)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame(0.0, index=dates, columns=["A"])
        result = factor.compute(daily_vsa_high2min=data, T=3)
        assert result.iloc[2, 0] == pytest.approx(0.0, abs=1e-15)


class TestBTypeVolumeDistOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        data = pd.DataFrame(
            np.random.uniform(0.01, 0.1, (30, 3)), index=dates, columns=stocks
        )
        result = factor.compute(daily_vsa_high2min=data, T=20)
        assert result.shape == data.shape

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame([0.01, 0.02, 0.03, 0.04, 0.05], index=dates, columns=["A"])
        result = factor.compute(daily_vsa_high2min=data, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        data = pd.DataFrame(
            np.random.uniform(0.01, 0.1, (25, 2)), index=dates, columns=["A", "B"]
        )
        result = factor.compute(daily_vsa_high2min=data, T=20)
        assert result.iloc[:19].isna().all().all()
        assert result.iloc[19:].notna().all().all()
