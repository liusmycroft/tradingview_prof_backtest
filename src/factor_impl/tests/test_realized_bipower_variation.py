import numpy as np
import pandas as pd
import pytest
from math import gamma

from factors.realized_bipower_variation import RealizedBipowerVariationFactor, MU_1


@pytest.fixture
def factor():
    return RealizedBipowerVariationFactor()


class TestRBVMetadata:
    def test_name(self, factor):
        assert factor.name == "RBV"

    def test_category(self, factor):
        assert factor.category == "高频波动跳跃"

    def test_repr(self, factor):
        assert "RBV" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "RBV"
        assert meta["category"] == "高频波动跳跃"


class TestRBVMu1:
    def test_mu1_value(self):
        """mu_1 = sqrt(2/pi) ≈ 0.7979."""
        expected = (2.0 / np.pi) ** 0.5
        assert MU_1 == pytest.approx(expected, rel=1e-10)


class TestRBVHandCalculated:
    def test_rolling_mean_T3(self, factor):
        """T=3 滚动均值手算验证。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        data = pd.DataFrame([0.01, 0.02, 0.03, 0.04, 0.05], index=dates, columns=stocks)

        result = factor.compute(daily_rbv=data, T=3)

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx((0.01 + 0.02 + 0.03) / 3, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx((0.02 + 0.03 + 0.04) / 3, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx((0.03 + 0.04 + 0.05) / 3, rel=1e-10)

    def test_constant_input(self, factor):
        """常数输入时，滚动均值等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(0.005, index=dates, columns=stocks)

        result = factor.compute(daily_rbv=data, T=5)
        # 前 4 行为 NaN，第 5 行开始为 0.005
        for i in range(4, 10):
            assert result.iloc[i, 0] == pytest.approx(0.005, rel=1e-10)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame(
            {"A": [0.01, 0.02, 0.03, 0.04, 0.05],
             "B": [0.10, 0.20, 0.30, 0.40, 0.50]},
            index=dates,
        )

        result = factor.compute(daily_rbv=data, T=3)
        assert result.iloc[2, 0] == pytest.approx(0.02, rel=1e-10)
        assert result.iloc[2, 1] == pytest.approx(0.20, rel=1e-10)


class TestRBVComputeDaily:
    def test_simple_intraday(self):
        """手算验证 compute_daily_rbv_from_intraday。"""
        # r = [0.01, -0.02, 0.03]
        # bipower_sum = |0.01|*|-0.02| + |-0.02|*|0.03| = 0.0002 + 0.0006 = 0.0008
        # RBV = mu_1^{-2} * 0.0008
        r = pd.Series([0.01, -0.02, 0.03])
        rbv = RealizedBipowerVariationFactor.compute_daily_rbv_from_intraday(r)
        expected = MU_1 ** (-2) * (0.01 * 0.02 + 0.02 * 0.03)
        assert rbv == pytest.approx(expected, rel=1e-10)


class TestRBVEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        values = np.ones(10) * 0.01
        values[3] = np.nan
        data = pd.DataFrame(values, index=dates, columns=stocks)

        result = factor.compute(daily_rbv=data, T=5)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_rbv=data, T=5)
        assert result.isna().all().all()

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        data = pd.DataFrame(
            np.random.uniform(0.001, 0.01, (30, 3)), index=dates, columns=stocks
        )
        result = factor.compute(daily_rbv=data, T=20)
        assert result.shape == data.shape
        assert list(result.columns) == list(data.columns)

    def test_first_T_minus_1_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(np.ones(10) * 0.01, index=dates, columns=stocks)
        T = 5
        result = factor.compute(daily_rbv=data, T=T)
        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
