"""相似低波因子测试"""

import numpy as np
import pandas as pd
import pytest
from factors.similar_low_vol import SimilarLowVolFactor


@pytest.fixture
def factor():
    return SimilarLowVolFactor()


def _make_data(n=150, n_stocks=2, seed=42):
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n)
    stocks = [f"S{i}" for i in range(n_stocks)]
    close = pd.DataFrame(
        np.cumsum(np.random.randn(n, n_stocks) * 0.02, axis=0) + 10,
        index=dates,
        columns=stocks,
    )
    excess_returns = pd.DataFrame(
        np.random.randn(n, n_stocks) * 0.01, index=dates, columns=stocks
    )
    return close, excess_returns


class TestMetadata:
    def test_name(self, factor):
        assert factor.name == "SIMILAR_LOW_VOL"

    def test_category(self, factor):
        assert factor.category == "量价改进"

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "SIMILAR_LOW_VOL"

    def test_repr(self, factor):
        assert "SimilarLowVolFactor" in repr(factor)


class TestCompute:
    def test_output_shape(self, factor):
        close, er = _make_data(n=150, n_stocks=2)
        result = factor.compute(close=close, excess_returns=er)
        assert result.shape == close.shape

    def test_early_rows_nan(self, factor):
        """前lookback+RW行应为NaN"""
        close, er = _make_data(n=150, n_stocks=1)
        result = factor.compute(close=close, excess_returns=er, lookback=120, RW=6)
        # 前126行应全为NaN
        assert result.iloc[:126, 0].isna().all()

    def test_positive_values(self, factor):
        """因子值 = 1/std，应为正数"""
        close, er = _make_data(n=200, n_stocks=1, seed=123)
        result = factor.compute(
            close=close, excess_returns=er,
            lookback=60, RW=5, threshold=0.2, holding_time=3,
        )
        valid = result.dropna()
        if len(valid) > 0:
            assert (valid > 0).all().all()

    def test_high_threshold_fewer_matches(self, factor):
        """高阈值应产生更少的有效值"""
        close, er = _make_data(n=200, n_stocks=1, seed=99)
        result_low = factor.compute(
            close=close, excess_returns=er, lookback=60, RW=5, threshold=0.1,
        )
        result_high = factor.compute(
            close=close, excess_returns=er, lookback=60, RW=5, threshold=0.8,
        )
        assert result_low.notna().sum().sum() >= result_high.notna().sum().sum()

    def test_short_data_all_nan(self, factor):
        """数据长度不足时应全为NaN"""
        dates = pd.date_range("2024-01-01", periods=10)
        stocks = ["A"]
        close = pd.DataFrame(np.arange(10, 20, dtype=float).reshape(-1, 1), index=dates, columns=stocks)
        er = pd.DataFrame(np.zeros((10, 1)), index=dates, columns=stocks)
        result = factor.compute(close=close, excess_returns=er, lookback=120)
        assert result.isna().all().all()
