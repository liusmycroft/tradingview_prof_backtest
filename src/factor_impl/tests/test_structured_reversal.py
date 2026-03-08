import numpy as np
import pandas as pd
import pytest

from factors.structured_reversal import StructuredReversalFactor


@pytest.fixture
def factor():
    return StructuredReversalFactor()


class TestStructuredReversalMetadata:
    def test_name(self, factor):
        assert factor.name == "STRUCTURED_REVERSAL"

    def test_category(self, factor):
        assert factor.category == "高频动量反转"

    def test_repr(self, factor):
        assert "STRUCTURED_REVERSAL" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "STRUCTURED_REVERSAL"


class TestStructuredReversalCompute:
    def test_known_values(self, factor):
        """手算验证：所有成交量相同时，阈值=10%分位，动量时段和反转时段的加权收益。"""
        idx = pd.date_range("2024-01-01", periods=10, freq="h")
        # 所有成交量相同 -> 10%分位 = 该值本身
        # 所有 vol <= threshold -> 全部为动量时段
        log_ret = pd.DataFrame({"A": np.ones(10) * 0.01}, index=idx)
        vol = pd.DataFrame({"A": np.ones(10) * 100.0}, index=idx)

        result = factor.compute(log_returns=log_ret, volumes=vol, quantile_threshold=0.1)
        # 所有 vol <= threshold (=100), 全为动量时段
        # rev_mom = equal weighted 0.01 = 0.01
        # rev_rev = 0 (no rev periods with vol > threshold strictly)
        # 实际上 quantile(0.1) 对10个相同值 = 100, 所以 vol<=100 全为动量
        # rev_mask (vol>100) 为空 -> rev_rev=0
        # result = 0 - 0.01 = -0.01
        assert result.iloc[0, 0] == pytest.approx(-0.01, abs=1e-6)

    def test_zero_returns(self, factor):
        """收益率全为0时，因子值为0。"""
        idx = pd.date_range("2024-01-01", periods=20, freq="h")
        log_ret = pd.DataFrame({"A": np.zeros(20)}, index=idx)
        vol = pd.DataFrame({"A": np.random.uniform(1, 100, 20)}, index=idx)

        result = factor.compute(log_returns=log_ret, volumes=vol)
        assert result.iloc[0, 0] == pytest.approx(0.0, abs=1e-10)

    def test_multi_stock(self, factor):
        """多只股票独立计算。"""
        np.random.seed(42)
        idx = pd.date_range("2024-01-01", periods=20, freq="h")
        log_ret = pd.DataFrame({
            "A": np.random.randn(20) * 0.01,
            "B": np.random.randn(20) * 0.02,
        }, index=idx)
        vol = pd.DataFrame({
            "A": np.random.uniform(10, 100, 20),
            "B": np.random.uniform(10, 100, 20),
        }, index=idx)

        result = factor.compute(log_returns=log_ret, volumes=vol)
        assert result.shape == (1, 2)
        assert not np.isnan(result.iloc[0, 0])
        assert not np.isnan(result.iloc[0, 1])


class TestStructuredReversalEdgeCases:
    def test_nan_in_input(self, factor):
        idx = pd.date_range("2024-01-01", periods=10, freq="h")
        log_ret = pd.DataFrame({"A": np.random.randn(10) * 0.01}, index=idx)
        vol = pd.DataFrame({"A": np.random.uniform(10, 100, 10)}, index=idx)
        log_ret.iloc[3, 0] = np.nan

        result = factor.compute(log_returns=log_ret, volumes=vol)
        assert isinstance(result, pd.DataFrame)

    def test_zero_volume(self, factor):
        """成交量全为0时结果为 NaN。"""
        idx = pd.date_range("2024-01-01", periods=10, freq="h")
        log_ret = pd.DataFrame({"A": np.ones(10) * 0.01}, index=idx)
        vol = pd.DataFrame({"A": np.zeros(10)}, index=idx)

        result = factor.compute(log_returns=log_ret, volumes=vol)
        assert np.isnan(result.iloc[0, 0])


class TestStructuredReversalOutputShape:
    def test_output_is_dataframe(self, factor):
        idx = pd.date_range("2024-01-01", periods=10, freq="h")
        log_ret = pd.DataFrame({"A": np.random.randn(10) * 0.01}, index=idx)
        vol = pd.DataFrame({"A": np.random.uniform(10, 100, 10)}, index=idx)

        result = factor.compute(log_returns=log_ret, volumes=vol)
        assert isinstance(result, pd.DataFrame)

    def test_output_columns(self, factor):
        idx = pd.date_range("2024-01-01", periods=10, freq="h")
        stocks = ["A", "B", "C"]
        log_ret = pd.DataFrame(
            np.random.randn(10, 3) * 0.01, index=idx, columns=stocks
        )
        vol = pd.DataFrame(
            np.random.uniform(10, 100, (10, 3)), index=idx, columns=stocks
        )

        result = factor.compute(log_returns=log_ret, volumes=vol)
        assert list(result.columns) == stocks
