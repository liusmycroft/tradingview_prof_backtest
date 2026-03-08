import numpy as np
import pandas as pd
import pytest

from factors.vsa_close_diff import VSACloseDiffFactor


@pytest.fixture
def factor():
    return VSACloseDiffFactor()


class TestVSACloseDiffMetadata:
    def test_name(self, factor):
        assert factor.name == "VSA_CLOSE_DIFF"

    def test_category(self, factor):
        assert factor.category == "高频成交分布"

    def test_repr(self, factor):
        assert "VSA_CLOSE_DIFF" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "VSA_CLOSE_DIFF"
        assert meta["category"] == "高频成交分布"
        assert "差异" in meta["description"] or "支撑" in meta["description"]


class TestVSACloseDiffCompute:
    """测试 compute 方法。"""

    def test_basic_known_values(self, factor):
        """验证 VSA_Low - Close 的滚动均值。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        vsa_low = pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0, 14.0]}, index=dates)
        close = pd.DataFrame({"A": [9.0, 10.0, 11.0, 12.0, 13.0]}, index=dates)

        result = factor.compute(vsa_low=vsa_low, close=close, T=3)

        # daily_diff = [1, 1, 1, 1, 1]
        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(1.0)
        assert result.iloc[3, 0] == pytest.approx(1.0)
        assert result.iloc[4, 0] == pytest.approx(1.0)

    def test_negative_diff(self, factor):
        """收盘价高于 VSA 下限时差值为负。"""
        dates = pd.bdate_range("2025-01-01", periods=4)
        vsa_low = pd.DataFrame({"A": [9.0, 9.0, 9.0, 9.0]}, index=dates)
        close = pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0]}, index=dates)

        result = factor.compute(vsa_low=vsa_low, close=close, T=3)

        # daily_diff = [-1, -2, -3, -4]
        assert result.iloc[2, 0] == pytest.approx(-2.0)  # mean(-1, -2, -3)
        assert result.iloc[3, 0] == pytest.approx(-3.0)  # mean(-2, -3, -4)

    def test_mixed_diff(self, factor):
        """正负差值混合。"""
        dates = pd.bdate_range("2025-01-01", periods=4)
        vsa_low = pd.DataFrame({"A": [10.0, 8.0, 12.0, 9.0]}, index=dates)
        close = pd.DataFrame({"A": [9.0, 10.0, 10.0, 11.0]}, index=dates)

        result = factor.compute(vsa_low=vsa_low, close=close, T=3)

        # daily_diff = [1, -2, 2, -2]
        assert result.iloc[2, 0] == pytest.approx((1 - 2 + 2) / 3)
        assert result.iloc[3, 0] == pytest.approx((-2 + 2 - 2) / 3)

    def test_multi_stock(self, factor):
        """多只股票同时计算。"""
        dates = pd.bdate_range("2025-01-01", periods=4)
        vsa_low = pd.DataFrame(
            {"A": [10.0, 11.0, 12.0, 13.0], "B": [20.0, 19.0, 18.0, 17.0]},
            index=dates,
        )
        close = pd.DataFrame(
            {"A": [9.0, 9.0, 9.0, 9.0], "B": [20.0, 20.0, 20.0, 20.0]},
            index=dates,
        )

        result = factor.compute(vsa_low=vsa_low, close=close, T=3)

        assert result.shape == (4, 2)
        # A: diff = [1, 2, 3, 4], mean(1,2,3)=2
        assert result.iloc[2, 0] == pytest.approx(2.0)
        # B: diff = [0, -1, -2, -3], mean(0,-1,-2)=-1
        assert result.iloc[2, 1] == pytest.approx(-1.0)


class TestVSACloseDiffEdgeCases:
    def test_nan_in_input(self, factor):
        """输入含 NaN 时不应抛异常。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        vsa_low = pd.DataFrame({"A": [10.0, np.nan, 12.0, 13.0, 14.0]}, index=dates)
        close = pd.DataFrame({"A": [9.0, 10.0, 11.0, 12.0, 13.0]}, index=dates)

        result = factor.compute(vsa_low=vsa_low, close=close, T=3)

        assert isinstance(result, pd.DataFrame)

    def test_zero_diff(self, factor):
        """VSA_Low == Close 时差值为 0。"""
        dates = pd.bdate_range("2025-01-01", periods=4)
        prices = pd.DataFrame({"A": [10.0, 10.0, 10.0, 10.0]}, index=dates)

        result = factor.compute(vsa_low=prices, close=prices, T=3)

        assert result.iloc[2, 0] == pytest.approx(0.0)


class TestVSACloseDiffOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=30)
        stocks = ["A", "B", "C"]
        vsa_low = pd.DataFrame(
            np.random.uniform(9, 11, (30, 3)), index=dates, columns=stocks
        )
        close = pd.DataFrame(
            np.random.uniform(9, 11, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(vsa_low=vsa_low, close=close, T=20)

        assert result.shape == vsa_low.shape
        assert list(result.columns) == list(vsa_low.columns)

    def test_output_is_dataframe(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=5)
        vsa_low = pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0, 14.0]}, index=dates)
        close = pd.DataFrame({"A": [9.0, 10.0, 11.0, 12.0, 13.0]}, index=dates)

        result = factor.compute(vsa_low=vsa_low, close=close, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        """前 T-1 行应全为 NaN。"""
        dates = pd.bdate_range("2025-01-01", periods=25)
        stocks = ["A", "B"]
        vsa_low = pd.DataFrame(
            np.random.uniform(9, 11, (25, 2)), index=dates, columns=stocks
        )
        close = pd.DataFrame(
            np.random.uniform(9, 11, (25, 2)), index=dates, columns=stocks
        )
        T = 20

        result = factor.compute(vsa_low=vsa_low, close=close, T=T)

        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
