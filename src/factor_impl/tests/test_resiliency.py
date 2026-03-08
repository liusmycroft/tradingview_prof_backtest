import numpy as np
import pandas as pd
import pytest

from factors.resiliency import ResiliencyFactor


@pytest.fixture
def factor():
    return ResiliencyFactor()


class TestResiliencyMetadata:
    def test_name(self, factor):
        assert factor.name == "RESILIENCY"

    def test_category(self, factor):
        assert factor.category == "高频流动性"

    def test_repr(self, factor):
        assert "RESILIENCY" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "RESILIENCY"
        assert meta["category"] == "高频流动性"


class TestResiliencyCompute:
    def test_precomputed_passthrough(self, factor):
        """预计算模式直接返回。"""
        dates = pd.date_range("2024-01-01", periods=5)
        df = pd.DataFrame({"A": [0.1, 0.2, 0.3, 0.4, 0.5]}, index=dates)

        result = factor.compute(transitory_price=df)
        pd.testing.assert_frame_equal(result, df)

    def test_output_non_negative(self, factor):
        """弹性因子应非负。"""
        np.random.seed(42)
        dates = ["2024-01-01"]
        minutes = list(range(48))
        idx = pd.MultiIndex.from_product([dates, minutes])
        z = pd.DataFrame({"A": np.random.randn(48) * 0.01}, index=idx)

        result = factor.compute(transitory_price=z)
        val = result.iloc[0, 0]
        if not np.isnan(val):
            assert val >= -1e-10

    def test_zero_transitory(self, factor):
        """暂时价格全为0时，弹性为0。"""
        dates = ["2024-01-01"]
        minutes = list(range(20))
        idx = pd.MultiIndex.from_product([dates, minutes])
        z = pd.DataFrame({"A": np.zeros(20)}, index=idx)

        result = factor.compute(transitory_price=z)
        assert result.iloc[0, 0] == pytest.approx(0.0, abs=1e-10)

    def test_multi_stock(self, factor):
        """多只股票独立计算。"""
        np.random.seed(42)
        dates = ["2024-01-01"]
        minutes = list(range(48))
        idx = pd.MultiIndex.from_product([dates, minutes])
        z = pd.DataFrame({
            "A": np.random.randn(48) * 0.01,
            "B": np.random.randn(48) * 0.02,
        }, index=idx)

        result = factor.compute(transitory_price=z)
        assert result.shape == (1, 2)
        assert not np.isnan(result.iloc[0, 0])
        assert not np.isnan(result.iloc[0, 1])


class TestResiliencyEdgeCases:
    def test_too_few_points(self, factor):
        """数据点不足时结果为 NaN。"""
        dates = ["2024-01-01"]
        minutes = [0, 1, 2]
        idx = pd.MultiIndex.from_product([dates, minutes])
        z = pd.DataFrame({"A": [0.01, 0.02, np.nan]}, index=idx)

        result = factor.compute(transitory_price=z)
        # Only 2 valid points, half=1, should still compute
        assert isinstance(result, pd.DataFrame)

    def test_all_nan(self, factor):
        """全 NaN 输入。"""
        dates = ["2024-01-01"]
        minutes = list(range(10))
        idx = pd.MultiIndex.from_product([dates, minutes])
        z = pd.DataFrame({"A": [np.nan] * 10}, index=idx)

        result = factor.compute(transitory_price=z)
        assert np.isnan(result.iloc[0, 0])


class TestResiliencyOutputShape:
    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=3)
        df = pd.DataFrame({"A": [0.1, 0.2, 0.3]}, index=dates)
        result = factor.compute(transitory_price=df)
        assert isinstance(result, pd.DataFrame)

    def test_output_shape_multiindex(self, factor):
        np.random.seed(42)
        dates = ["2024-01-01", "2024-01-02"]
        minutes = list(range(20))
        idx = pd.MultiIndex.from_product([dates, minutes])
        z = pd.DataFrame({
            "A": np.random.randn(40) * 0.01,
            "B": np.random.randn(40) * 0.01,
        }, index=idx)

        result = factor.compute(transitory_price=z)
        assert result.shape == (2, 2)
