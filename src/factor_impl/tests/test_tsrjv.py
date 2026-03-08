import numpy as np
import pandas as pd
import pytest

from factors.tsrjv import TSRJVFactor


@pytest.fixture
def factor():
    return TSRJVFactor()


class TestTSRJVMetadata:
    def test_name(self, factor):
        assert factor.name == "TSRJV"

    def test_category(self, factor):
        assert factor.category == "高频波动跳跃"

    def test_repr(self, factor):
        assert "TSRJV" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "TSRJV"
        assert meta["category"] == "高频波动跳跃"


class TestTSRJVCompute:
    def test_known_values(self, factor):
        """手算验证加权均值: sum(T*SRJV) / sum(T)"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        daily_srjv = pd.DataFrame({"A": [0.01, 0.02, 0.03]}, index=dates)
        daily_jump_t = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=dates)

        result = factor.compute(daily_srjv=daily_srjv, daily_jump_t=daily_jump_t, T=3)

        # weighted: 1*0.01 + 2*0.02 + 3*0.03 = 0.01 + 0.04 + 0.09 = 0.14
        # sum_t: 1 + 2 + 3 = 6
        # TSRJV = 0.14 / 6 = 0.02333...
        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(0.14 / 6)

    def test_equal_weights(self, factor):
        """权重相等时, TSRJV = 简单均值。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        daily_srjv = pd.DataFrame({"A": [0.01, 0.02, 0.03]}, index=dates)
        daily_jump_t = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=dates)

        result = factor.compute(daily_srjv=daily_srjv, daily_jump_t=daily_jump_t, T=3)

        # equal weights: mean(0.01, 0.02, 0.03) = 0.02
        assert result.iloc[2, 0] == pytest.approx(0.02)

    def test_single_dominant_weight(self, factor):
        """一个权重远大于其他时, TSRJV 接近该值。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        daily_srjv = pd.DataFrame({"A": [0.01, 0.02, 0.05]}, index=dates)
        daily_jump_t = pd.DataFrame({"A": [0.001, 0.001, 100.0]}, index=dates)

        result = factor.compute(daily_srjv=daily_srjv, daily_jump_t=daily_jump_t, T=3)

        # 第3天权重远大于其他, TSRJV 应接近 0.05
        assert result.iloc[2, 0] == pytest.approx(0.05, rel=1e-3)

    def test_multi_stock(self, factor):
        """多只股票独立计算。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        daily_srjv = pd.DataFrame(
            {"A": [0.01, 0.02, 0.03], "B": [0.10, 0.20, 0.30]}, index=dates
        )
        daily_jump_t = pd.DataFrame(
            {"A": [1.0, 1.0, 1.0], "B": [1.0, 1.0, 1.0]}, index=dates
        )

        result = factor.compute(daily_srjv=daily_srjv, daily_jump_t=daily_jump_t, T=3)

        assert result.shape == (3, 2)
        assert result.iloc[2, 0] == pytest.approx(0.02)
        assert result.iloc[2, 1] == pytest.approx(0.20)

    def test_negative_srjv(self, factor):
        """SRJV 为负时, TSRJV 也可为负。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        daily_srjv = pd.DataFrame({"A": [-0.01, -0.02, -0.03]}, index=dates)
        daily_jump_t = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=dates)

        result = factor.compute(daily_srjv=daily_srjv, daily_jump_t=daily_jump_t, T=3)

        assert result.iloc[2, 0] == pytest.approx(-0.02)


class TestTSRJVEdgeCases:
    def test_nan_in_input(self, factor):
        """输入含 NaN 时不应抛异常。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        daily_srjv = pd.DataFrame({"A": [0.01, np.nan, 0.03, 0.04, 0.05]}, index=dates)
        daily_jump_t = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=dates)

        result = factor.compute(daily_srjv=daily_srjv, daily_jump_t=daily_jump_t, T=3)

        assert isinstance(result, pd.DataFrame)

    def test_zero_weights(self, factor):
        """权重全零时, 结果应为 NaN (0/0)。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        daily_srjv = pd.DataFrame({"A": [0.01, 0.02, 0.03]}, index=dates)
        daily_jump_t = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=dates)

        result = factor.compute(daily_srjv=daily_srjv, daily_jump_t=daily_jump_t, T=3)

        assert np.isnan(result.iloc[2, 0])


class TestTSRJVOutputShape:
    def test_output_is_dataframe(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=5)
        daily_srjv = pd.DataFrame({"A": np.random.randn(5) * 0.01}, index=dates)
        daily_jump_t = pd.DataFrame({"A": np.random.uniform(0.5, 3.0, 5)}, index=dates)

        result = factor.compute(daily_srjv=daily_srjv, daily_jump_t=daily_jump_t, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_output_shape_matches_input(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=30)
        stocks = ["A", "B", "C"]
        daily_srjv = pd.DataFrame(
            np.random.randn(30, 3) * 0.01, index=dates, columns=stocks
        )
        daily_jump_t = pd.DataFrame(
            np.random.uniform(0.5, 3.0, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_srjv=daily_srjv, daily_jump_t=daily_jump_t, T=20)
        assert result.shape == daily_srjv.shape
        assert list(result.columns) == stocks

    def test_first_T_minus_1_rows_are_nan(self, factor):
        """前 T-1 行应全为 NaN。"""
        dates = pd.bdate_range("2025-01-01", periods=25)
        daily_srjv = pd.DataFrame(
            np.random.randn(25, 2) * 0.01, index=dates, columns=["A", "B"]
        )
        daily_jump_t = pd.DataFrame(
            np.random.uniform(0.5, 3.0, (25, 2)), index=dates, columns=["A", "B"]
        )
        T = 20

        result = factor.compute(daily_srjv=daily_srjv, daily_jump_t=daily_jump_t, T=T)

        assert result.iloc[: T - 1].isna().all().all()
