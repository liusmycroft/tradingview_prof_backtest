import numpy as np
import pandas as pd
import pytest

from factors.golden_ratio_reversal import GoldenRatioReversalFactor


@pytest.fixture
def factor():
    return GoldenRatioReversalFactor()


class TestGoldenRatioReversalMetadata:
    def test_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "GOLDEN_RATIO_REVERSAL"
        assert meta["category"] == "高频动量反转"
        assert "黄金分割" in meta["description"] or "10:00" in meta["description"]

    def test_repr(self, factor):
        r = repr(factor)
        assert "GoldenRatioReversalFactor" in r
        assert "GOLDEN_RATIO_REVERSAL" in r


class TestGoldenRatioReversalCompute:
    def test_known_values(self, factor):
        """用已知数据验证对数收益率滚动求和。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        close = pd.DataFrame({"A": [110.0, 105.0, 120.0]}, index=dates)
        price_1000 = pd.DataFrame({"A": [100.0, 100.0, 100.0]}, index=dates)

        result = factor.compute(close=close, price_1000=price_1000, T=3)

        # ln(110/100) + ln(105/100) + ln(120/100)
        expected = np.log(1.10) + np.log(1.05) + np.log(1.20)
        np.testing.assert_almost_equal(result.iloc[2, 0], expected, decimal=10)

    def test_single_day_log_return(self, factor):
        """T=1 时，结果应等于单日对数收益率。"""
        dates = pd.bdate_range("2025-01-01", periods=1)
        close = pd.DataFrame({"A": [110.0]}, index=dates)
        price_1000 = pd.DataFrame({"A": [100.0]}, index=dates)

        result = factor.compute(close=close, price_1000=price_1000, T=1)

        np.testing.assert_almost_equal(result.iloc[0, 0], np.log(1.10))

    def test_min_periods(self, factor):
        """窗口不足 T 天时，结果应为 NaN。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        close = pd.DataFrame({"A": [100.0] * 5}, index=dates)
        price_1000 = pd.DataFrame({"A": [100.0] * 5}, index=dates)

        result = factor.compute(close=close, price_1000=price_1000, T=20)

        assert result.isna().all().all()

    def test_multi_stock(self, factor):
        """多只股票同时计算。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        close = pd.DataFrame(
            {"A": [110.0, 105.0, 120.0], "B": [200.0, 210.0, 190.0]}, index=dates
        )
        price_1000 = pd.DataFrame(
            {"A": [100.0, 100.0, 100.0], "B": [195.0, 200.0, 200.0]}, index=dates
        )

        result = factor.compute(close=close, price_1000=price_1000, T=3)

        assert result.shape == (3, 2)
        expected_a = np.log(110 / 100) + np.log(105 / 100) + np.log(120 / 100)
        expected_b = np.log(200 / 195) + np.log(210 / 200) + np.log(190 / 200)
        np.testing.assert_almost_equal(result.loc[dates[2], "A"], expected_a)
        np.testing.assert_almost_equal(result.loc[dates[2], "B"], expected_b)

    def test_nan_propagation(self, factor):
        """输入含 NaN 时，输出对应窗口也应为 NaN。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        close = pd.DataFrame({"A": [110.0, np.nan, 120.0]}, index=dates)
        price_1000 = pd.DataFrame({"A": [100.0, 100.0, 100.0]}, index=dates)

        result = factor.compute(close=close, price_1000=price_1000, T=2)

        # 第 1 行（index=1）的 log_return 为 NaN，所以 window [0,1] 和 [1,2] 都含 NaN
        assert np.isnan(result.iloc[1, 0])
        assert np.isnan(result.iloc[2, 0])

    def test_output_shape(self, factor):
        """输出形状应与输入一致。"""
        dates = pd.bdate_range("2025-01-01", periods=25)
        close = pd.DataFrame(
            np.random.uniform(10, 20, (25, 3)),
            index=dates,
            columns=["A", "B", "C"],
        )
        price_1000 = pd.DataFrame(
            np.random.uniform(10, 20, (25, 3)),
            index=dates,
            columns=["A", "B", "C"],
        )

        result = factor.compute(close=close, price_1000=price_1000, T=20)

        assert result.shape == (25, 3)
        # 前 19 行应为 NaN
        assert result.iloc[:19].isna().all().all()
        # 第 20 行起应有值
        assert result.iloc[19:].notna().all().all()
