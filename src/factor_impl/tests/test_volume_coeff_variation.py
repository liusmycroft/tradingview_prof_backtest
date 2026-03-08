import numpy as np
import pandas as pd
import pytest

from factors.volume_coeff_variation import VolumeCoeffVariationFactor


@pytest.fixture
def factor():
    return VolumeCoeffVariationFactor()


class TestVolumeCoeffVariationMetadata:
    def test_name(self, factor):
        assert factor.name == "VOLUME_COEFF_VARIATION"

    def test_category(self, factor):
        assert factor.category == "高频因子-资金流类"

    def test_repr(self, factor):
        assert "VOLUME_COEFF_VARIATION" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "VOLUME_COEFF_VARIATION"


class TestVolumeCoeffVariationCompute:
    def test_constant_input(self, factor):
        """常数输入时 std=0, VCV=0。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        minute_amount = pd.DataFrame(100.0, index=dates, columns=stocks)

        result = factor.compute(minute_amount=minute_amount, T=5)
        # std of constant = 0, so VCV = 0
        assert result.iloc[-1, 0] == pytest.approx(0.0, abs=1e-10)

    def test_positive_variation(self, factor):
        """有变化的输入应产生正的变异系数。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        vals = [100, 200, 100, 200, 100, 200, 100, 200, 100, 200]
        minute_amount = pd.DataFrame(vals, index=dates, columns=stocks, dtype=float)

        result = factor.compute(minute_amount=minute_amount, T=5)
        assert result.iloc[-1, 0] > 0

    def test_leading_nan(self, factor):
        """前 T-1 行应为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        minute_amount = pd.DataFrame(np.random.rand(10) * 100 + 50, index=dates, columns=stocks)

        result = factor.compute(minute_amount=minute_amount, T=5)
        assert result.iloc[:4]["A"].isna().all()
        assert result.iloc[4:]["A"].notna().all()

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=15, freq="D")
        stocks = ["A", "B"]
        minute_amount = pd.DataFrame(np.random.rand(15, 2) * 100, index=dates, columns=stocks)

        result = factor.compute(minute_amount=minute_amount, T=5)
        assert result.shape == minute_amount.shape
        assert isinstance(result, pd.DataFrame)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        minute_amount = pd.DataFrame(
            {"A": [100.0] * 10, "B": [100, 200, 100, 200, 100, 200, 100, 200, 100, 200]},
            index=dates,
        )

        result = factor.compute(minute_amount=minute_amount, T=5)
        assert result.iloc[-1]["A"] == pytest.approx(0.0, abs=1e-10)
        assert result.iloc[-1]["B"] > 0
