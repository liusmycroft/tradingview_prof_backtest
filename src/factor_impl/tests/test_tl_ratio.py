import numpy as np
import pandas as pd
import pytest

from factors.tl_ratio import TLRatioFactor


@pytest.fixture
def factor():
    return TLRatioFactor()


class TestTLRatioMetadata:
    def test_name(self, factor):
        assert factor.name == "TLRatio"

    def test_category(self, factor):
        assert factor.category == "行为金融-筹码分布"

    def test_repr(self, factor):
        assert "TLRatio" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "TLRatio"
        assert meta["category"] == "行为金融-筹码分布"


class TestTLRatioHandCalculated:
    """用手算数据验证 rolling(window=T, min_periods=1).mean() 的正确性。"""

    def test_constant_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        daily_tl_ratio = pd.DataFrame(0.3, index=dates, columns=stocks)

        result = factor.compute(daily_tl_ratio=daily_tl_ratio, T=20)

        np.testing.assert_array_almost_equal(result["A"].values, 0.3)

    def test_rolling_mean_manual_T3(self, factor):
        """T=3, 手动验证滚动均值。

        data = [10, 20, 30, 40]
        rolling(3, min_periods=1):
          mean_0 = 10/1 = 10.0
          mean_1 = (10+20)/2 = 15.0
          mean_2 = (10+20+30)/3 = 20.0
          mean_3 = (20+30+40)/3 = 30.0
        """
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        stocks = ["A"]
        daily_tl_ratio = pd.DataFrame(
            [10.0, 20.0, 30.0, 40.0], index=dates, columns=stocks
        )

        result = factor.compute(daily_tl_ratio=daily_tl_ratio, T=3)

        assert result.iloc[0, 0] == pytest.approx(10.0, rel=1e-6)
        assert result.iloc[1, 0] == pytest.approx(15.0, rel=1e-6)
        assert result.iloc[2, 0] == pytest.approx(20.0, rel=1e-6)
        assert result.iloc[3, 0] == pytest.approx(30.0, rel=1e-6)

    def test_rolling_recent_weight(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        vals = [0.1] * 5 + [0.9] * 5
        daily_tl_ratio = pd.DataFrame(vals, index=dates, columns=stocks)

        result = factor.compute(daily_tl_ratio=daily_tl_ratio, T=5)
        assert result.iloc[-1, 0] == pytest.approx(0.9, rel=1e-6)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        daily_tl_ratio = pd.DataFrame(
            {"A": [0.2] * 10, "B": [0.6] * 10}, index=dates
        )

        result = factor.compute(daily_tl_ratio=daily_tl_ratio, T=5)

        np.testing.assert_array_almost_equal(result["A"].values, 0.2)
        np.testing.assert_array_almost_equal(result["B"].values, 0.6)


class TestTLRatioEdgeCases:
    def test_single_value(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A"]
        daily_tl_ratio = pd.DataFrame([0.35], index=dates, columns=stocks)

        result = factor.compute(daily_tl_ratio=daily_tl_ratio, T=20)
        assert result.iloc[0, 0] == pytest.approx(0.35, rel=1e-10)

    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        values = np.ones(10) * 0.3
        values[3] = np.nan
        daily_tl_ratio = pd.DataFrame(values, index=dates, columns=stocks)

        result = factor.compute(daily_tl_ratio=daily_tl_ratio, T=5)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily_tl_ratio = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_tl_ratio=daily_tl_ratio, T=5)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        daily_tl_ratio = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_tl_ratio=daily_tl_ratio, T=5)
        for val in result["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)


class TestTLRatioOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily_tl_ratio = pd.DataFrame(
            np.random.uniform(0.0, 1.0, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_tl_ratio=daily_tl_ratio, T=20)

        assert result.shape == daily_tl_ratio.shape
        assert list(result.columns) == list(daily_tl_ratio.columns)
        assert list(result.index) == list(daily_tl_ratio.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_tl_ratio = pd.DataFrame(
            [0.1, 0.2, 0.3, 0.4, 0.5], index=dates, columns=stocks
        )

        result = factor.compute(daily_tl_ratio=daily_tl_ratio, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_no_leading_nan_min_periods_1(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        daily_tl_ratio = pd.DataFrame(
            np.random.uniform(0.0, 1.0, (10, 2)), index=dates, columns=stocks
        )

        result = factor.compute(daily_tl_ratio=daily_tl_ratio, T=20)
        assert result.iloc[0].notna().all()
        assert result.notna().all().all()
