import numpy as np
import pandas as pd
import pytest

from factors.follow_leader import FollowLeaderFactor


@pytest.fixture
def factor():
    return FollowLeaderFactor()


class TestFollowLeaderMetadata:
    def test_name(self, factor):
        assert factor.name == "FOLLOW_LEADER"

    def test_category(self, factor):
        assert factor.category == "高频量价相关性"

    def test_repr(self, factor):
        assert "FOLLOW_LEADER" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "FOLLOW_LEADER"


class TestFollowLeaderCompute:
    def test_constant_input(self, factor):
        """常数输入时，std=0，结果 = 0.5 * const。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        daily_fc = pd.DataFrame({"A": [2.0] * 25}, index=dates)

        result = factor.compute(daily_follow_coeff=daily_fc, T=20)
        # rolling_mean = 2.0, rolling_std = 0.0
        # result = 0.5*2.0 + 0.5*0.0 = 1.0
        assert result.iloc[-1, 0] == pytest.approx(1.0)

    def test_leading_nan(self, factor):
        """前 T-1 行应为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        daily_fc = pd.DataFrame({"A": np.random.rand(25)}, index=dates)

        result = factor.compute(daily_follow_coeff=daily_fc, T=20)
        assert result.iloc[:19].isna().all().all()
        assert result.iloc[19:].notna().all().all()

    def test_multi_stock(self, factor):
        """多只股票独立计算。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        daily_fc = pd.DataFrame({
            "A": [1.0] * 25,
            "B": [3.0] * 25,
        }, index=dates)

        result = factor.compute(daily_follow_coeff=daily_fc, T=20)
        # A: 0.5*1 + 0.5*0 = 0.5
        # B: 0.5*3 + 0.5*0 = 1.5
        assert result.iloc[-1, 0] == pytest.approx(0.5)
        assert result.iloc[-1, 1] == pytest.approx(1.5)


class TestFollowLeaderEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        daily_fc = pd.DataFrame({"A": np.random.rand(25)}, index=dates)
        daily_fc.iloc[10, 0] = np.nan

        result = factor.compute(daily_follow_coeff=daily_fc, T=20)
        assert isinstance(result, pd.DataFrame)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        daily_fc = pd.DataFrame({"A": [np.nan] * 25}, index=dates)

        result = factor.compute(daily_follow_coeff=daily_fc, T=20)
        assert result.isna().all().all()


class TestFollowLeaderOutputShape:
    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily_fc = pd.DataFrame(np.random.rand(30, 3), index=dates, columns=stocks)

        result = factor.compute(daily_follow_coeff=daily_fc, T=20)
        assert result.shape == (30, 3)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        daily_fc = pd.DataFrame({"A": np.random.rand(25)}, index=dates)

        result = factor.compute(daily_follow_coeff=daily_fc, T=20)
        assert isinstance(result, pd.DataFrame)
