import numpy as np
import pandas as pd
import pytest

from factors.nonlinear_hf_volatility import NonlinearHFVolatilityFactor


@pytest.fixture
def factor():
    return NonlinearHFVolatilityFactor()


class TestNonlinearHFVolatilityMetadata:
    def test_name(self, factor):
        assert factor.name == "NONLINEAR_HF_VOLATILITY"

    def test_category(self, factor):
        assert factor.category == "高频波动跳跃"

    def test_repr(self, factor):
        assert "NONLINEAR_HF_VOLATILITY" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "NONLINEAR_HF_VOLATILITY"
        assert meta["category"] == "高频波动跳跃"


class TestNonlinearHFVolatilityCompute:
    def test_basic_computation(self, factor):
        """验证基本计算逻辑。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A", "B"]
        idio_ratio = pd.DataFrame(
            {"A": [0.25, 0.36, 0.49, 0.64, 0.81],
             "B": [0.16, 0.25, 0.36, 0.49, 0.64]},
            index=dates,
        )
        hf_std = pd.DataFrame(
            {"A": [0.01, 0.02, 0.03, 0.04, 0.05],
             "B": [0.02, 0.03, 0.04, 0.05, 0.06]},
            index=dates,
        )

        result = factor.compute(idio_ratio=idio_ratio, hf_std=hf_std)
        assert result.shape == (5, 2)
        assert result.notna().all().all()

    def test_normalization_range(self, factor):
        """归一化后 norm 应在 [0, 1]，exp(norm) 在 [1, e]。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A", "B", "C"]
        idio_ratio = pd.DataFrame(1.0, index=dates, columns=stocks)
        hf_std = pd.DataFrame(
            {"A": [0.01, 0.02, 0.03],
             "B": [0.02, 0.04, 0.06],
             "C": [0.03, 0.06, 0.09]},
            index=dates,
        )

        result = factor.compute(idio_ratio=idio_ratio, hf_std=hf_std)
        # idio_ratio=1 -> sqrt=1, 所以 result = exp(norm)
        # norm 范围 [0,1], exp(norm) 范围 [1, e]
        assert (result >= 1.0 - 1e-10).all().all()
        assert (result <= np.e + 1e-10).all().all()

    def test_zero_idio_ratio(self, factor):
        """特异率为0时，因子应为0。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A", "B"]
        idio_ratio = pd.DataFrame(0.0, index=dates, columns=stocks)
        hf_std = pd.DataFrame(
            {"A": [0.01, 0.02, 0.03], "B": [0.02, 0.03, 0.04]},
            index=dates,
        )

        result = factor.compute(idio_ratio=idio_ratio, hf_std=hf_std)
        np.testing.assert_array_almost_equal(result.values, 0.0)

    def test_single_stock_nan_norm(self, factor):
        """单只股票时，横截面min=max，归一化分母为0，结果为NaN。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]
        idio_ratio = pd.DataFrame(0.5, index=dates, columns=stocks)
        hf_std = pd.DataFrame([0.01, 0.02, 0.03], index=dates, columns=stocks)

        result = factor.compute(idio_ratio=idio_ratio, hf_std=hf_std)
        # 单只股票横截面 min==max, denom=0 -> NaN
        assert result.isna().all().all()


class TestNonlinearHFVolatilityEdgeCases:
    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A", "B"]
        idio_ratio = pd.DataFrame(0.5, index=dates, columns=stocks)
        hf_std = pd.DataFrame(
            np.random.uniform(0.01, 0.1, (5, 2)), index=dates, columns=stocks
        )

        result = factor.compute(idio_ratio=idio_ratio, hf_std=hf_std)
        assert isinstance(result, pd.DataFrame)

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        idio_ratio = pd.DataFrame(
            np.random.uniform(0, 1, (30, 3)), index=dates, columns=stocks
        )
        hf_std = pd.DataFrame(
            np.random.uniform(0.01, 0.1, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(idio_ratio=idio_ratio, hf_std=hf_std)
        assert result.shape == (30, 3)
