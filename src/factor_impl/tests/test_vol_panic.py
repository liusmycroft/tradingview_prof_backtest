import numpy as np
import pandas as pd
import pytest

from factors.vol_panic import VolPanicFactor


@pytest.fixture
def factor():
    return VolPanicFactor()


class TestVolPanicMetadata:
    def test_name(self, factor):
        assert factor.name == "VOL_PANIC"

    def test_category(self, factor):
        assert factor.category == "高频动量反转"

    def test_repr(self, factor):
        assert "VOL_PANIC" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "VOL_PANIC"
        assert meta["category"] == "高频动量反转"


class TestVolPanicHandCalculated:
    """用手算数据验证计算的正确性。"""

    def test_constant_input(self, factor):
        """常数输入时, mean=常数, std=0, 因子=(常数+0)/2=常数/2。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily_vol_panic = pd.DataFrame(4.0, index=dates, columns=stocks)

        result = factor.compute(daily_vol_panic=daily_vol_panic, T=20)

        valid = result.dropna()
        # mean=4.0, std=0.0, result=(4.0+0.0)/2=2.0
        np.testing.assert_array_almost_equal(valid["A"].values, 2.0)

    def test_mean_and_std_combined(self, factor):
        """手动验证 T=3 的均值和标准差。"""
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        stocks = ["A"]
        daily_vol_panic = pd.DataFrame(
            [1.0, 2.0, 3.0, 4.0], index=dates, columns=stocks
        )

        result = factor.compute(daily_vol_panic=daily_vol_panic, T=3)

        # row 2: mean([1,2,3])=2.0, std([1,2,3], ddof=1)=1.0, factor=1.5
        assert result.iloc[2, 0] == pytest.approx((2.0 + 1.0) / 2, rel=1e-6)
        # row 3: mean([2,3,4])=3.0, std([2,3,4], ddof=1)=1.0, factor=2.0
        assert result.iloc[3, 0] == pytest.approx((3.0 + 1.0) / 2, rel=1e-6)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A", "B"]
        daily_vol_panic = pd.DataFrame(
            {"A": [1.0] * 25, "B": [2.0] * 25}, index=dates
        )

        result = factor.compute(daily_vol_panic=daily_vol_panic, T=20)

        valid = result.dropna()
        # 常数: mean=val, std=0, factor=val/2
        np.testing.assert_array_almost_equal(valid["A"].values, 0.5)
        np.testing.assert_array_almost_equal(valid["B"].values, 1.0)

    def test_positive_std_increases_factor(self, factor):
        """有波动时, std>0, 因子应大于 mean/2。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        np.random.seed(42)
        vals = np.random.randn(25) * 0.1 + 1.0
        daily_vol_panic = pd.DataFrame(vals, index=dates, columns=stocks)

        result = factor.compute(daily_vol_panic=daily_vol_panic, T=20)

        last_val = result.iloc[-1, 0]
        # mean ~ 1.0, std > 0, so factor > 0.5
        assert last_val > 0.5


class TestVolPanicEdgeCases:
    def test_nan_in_input(self, factor):
        """输入含 NaN 时不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        values = np.ones(25) * 1.0
        values[10] = np.nan
        daily_vol_panic = pd.DataFrame(values, index=dates, columns=stocks)

        result = factor.compute(daily_vol_panic=daily_vol_panic, T=20)
        assert result.shape == (25, 1)

    def test_all_nan(self, factor):
        """全 NaN 输入时, 结果应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily_vol_panic = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_vol_panic=daily_vol_panic, T=20)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        """全零输入时, mean=0, std=0, 因子=0。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily_vol_panic = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_vol_panic=daily_vol_panic, T=20)
        valid = result.dropna()
        for val in valid["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)


class TestVolPanicOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily_vol_panic = pd.DataFrame(
            np.random.uniform(-1, 1, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_vol_panic=daily_vol_panic, T=20)

        assert result.shape == daily_vol_panic.shape
        assert list(result.columns) == list(daily_vol_panic.columns)
        assert list(result.index) == list(daily_vol_panic.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily_vol_panic = pd.DataFrame([1.0] * 25, index=dates, columns=stocks)

        result = factor.compute(daily_vol_panic=daily_vol_panic, T=20)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        """min_periods=T, 前 T-1 行应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        T = 20
        daily_vol_panic = pd.DataFrame(
            np.random.uniform(-1, 1, (30, 2)), index=dates, columns=stocks
        )

        result = factor.compute(daily_vol_panic=daily_vol_panic, T=T)

        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
