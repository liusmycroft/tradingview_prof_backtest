import numpy as np
import pandas as pd
import pytest

from factors.patv import PATVFactor


@pytest.fixture
def factor():
    return PATVFactor()


class TestPATVMetadata:
    def test_name(self, factor):
        assert factor.name == "PATV"

    def test_category(self, factor):
        assert factor.category == "高频成交分布"

    def test_repr(self, factor):
        assert "PATV" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "PATV"
        assert meta["category"] == "高频成交分布"


class TestPATVHandCalculated:
    """用手算数据验证 EWM(span=T, min_periods=T) 计算的正确性。"""

    def test_constant_input(self, factor):
        """常数输入时, EMA 应等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily_patv = pd.DataFrame(3.0, index=dates, columns=stocks)

        result = factor.compute(daily_patv=daily_patv, T=20)

        valid = result.dropna()
        np.testing.assert_array_almost_equal(valid["A"].values, 3.0)

    def test_ema_weights_recent_more(self, factor):
        """EMA 应对近期数据赋予更高权重。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        values = [0.0] * 20 + [100.0] * 5
        daily_patv = pd.DataFrame(values, index=dates, columns=stocks)

        result = factor.compute(daily_patv=daily_patv, T=20)

        last_val = result.iloc[-1, 0]
        assert 0 < last_val < 100

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算 EMA。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A", "B"]
        daily_patv = pd.DataFrame(
            {"A": [5.0] * 25, "B": [10.0] * 25}, index=dates
        )

        result = factor.compute(daily_patv=daily_patv, T=20)

        valid = result.dropna()
        np.testing.assert_array_almost_equal(valid["A"].values, 5.0)
        np.testing.assert_array_almost_equal(valid["B"].values, 10.0)

    def test_ema_manual_T3(self, factor):
        """T=3, 手动验证 EMA 值。

        ewm(span=3, min_periods=3) -> alpha = 2/(3+1) = 0.5
        data = [10, 20, 30, 40]
        pandas ewm 从第一个非 NaN 开始递推:
          row 0, 1 为 NaN (min_periods=3)
          row 2: ewm of [10,20,30] = (0.25*10+0.5*20+1.0*30)/(0.25+0.5+1.0) = 24.2857
          row 3: ewm of [10,20,30,40] = 32.6667
        """
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        stocks = ["A"]
        daily_patv = pd.DataFrame(
            [10.0, 20.0, 30.0, 40.0], index=dates, columns=stocks
        )

        result = factor.compute(daily_patv=daily_patv, T=3)

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(24.285714285714285, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(32.666666666666667, rel=1e-10)


class TestPATVEdgeCases:
    def test_nan_in_input(self, factor):
        """输入含 NaN 时不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        values = np.ones(25) * 3.0
        values[10] = np.nan
        daily_patv = pd.DataFrame(values, index=dates, columns=stocks)

        result = factor.compute(daily_patv=daily_patv, T=20)
        assert result.shape == (25, 1)

    def test_all_nan(self, factor):
        """全 NaN 输入时, 结果应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily_patv = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_patv=daily_patv, T=20)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        """全零输入时, EMA 应全为 0。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily_patv = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_patv=daily_patv, T=20)
        valid = result.dropna()
        for val in valid["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)


class TestPATVOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily_patv = pd.DataFrame(
            np.random.uniform(0, 10, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_patv=daily_patv, T=20)

        assert result.shape == daily_patv.shape
        assert list(result.columns) == list(daily_patv.columns)
        assert list(result.index) == list(daily_patv.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily_patv = pd.DataFrame([1.0] * 25, index=dates, columns=stocks)

        result = factor.compute(daily_patv=daily_patv, T=20)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        """min_periods=T, 前 T-1 行应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        T = 20
        daily_patv = pd.DataFrame(
            np.random.uniform(0, 10, (30, 2)), index=dates, columns=stocks
        )

        result = factor.compute(daily_patv=daily_patv, T=T)

        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
