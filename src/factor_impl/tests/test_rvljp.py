import numpy as np
import pandas as pd
import pytest

from factors.rvljp import RVLJPFactor


@pytest.fixture
def factor():
    return RVLJPFactor()


class TestRVLJPMetadata:
    def test_name(self, factor):
        assert factor.name == "RVLJP"

    def test_category(self, factor):
        assert factor.category == "高频波动"

    def test_repr(self, factor):
        assert "RVLJP" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "RVLJP"
        assert meta["category"] == "高频波动"
        assert "跳跃" in meta["description"]


class TestRVLJPCompute:
    """测试 compute 方法。"""

    def test_basic_min(self, factor):
        """RVLJP = min(RVJP, large_positive_jump)。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        rvjp = pd.DataFrame({"A": [0.005, 0.010, 0.003]}, index=dates)
        large_pos = pd.DataFrame({"A": [0.003, 0.012, 0.001]}, index=dates)

        result = factor.compute(rvjp=rvjp, large_positive_jump=large_pos)

        expected = pd.DataFrame({"A": [0.003, 0.010, 0.001]}, index=dates)
        pd.testing.assert_frame_equal(result, expected)

    def test_rvjp_always_smaller(self, factor):
        """RVJP 始终小于 large_positive_jump 时，结果等于 RVJP。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        rvjp = pd.DataFrame({"A": [0.001, 0.002, 0.003]}, index=dates)
        large_pos = pd.DataFrame({"A": [0.010, 0.020, 0.030]}, index=dates)

        result = factor.compute(rvjp=rvjp, large_positive_jump=large_pos)

        pd.testing.assert_frame_equal(result, rvjp)

    def test_large_pos_always_smaller(self, factor):
        """large_positive_jump 始终小于 RVJP 时，结果等于 large_positive_jump。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        rvjp = pd.DataFrame({"A": [0.010, 0.020, 0.030]}, index=dates)
        large_pos = pd.DataFrame({"A": [0.001, 0.002, 0.003]}, index=dates)

        result = factor.compute(rvjp=rvjp, large_positive_jump=large_pos)

        pd.testing.assert_frame_equal(result, large_pos)

    def test_equal_values(self, factor):
        """两者相等时，结果也相等。"""
        dates = pd.bdate_range("2025-01-01", periods=2)
        data = pd.DataFrame({"A": [0.005, 0.010]}, index=dates)

        result = factor.compute(rvjp=data, large_positive_jump=data.copy())

        pd.testing.assert_frame_equal(result, data)

    def test_multi_stock(self, factor):
        """多只股票同时计算。"""
        dates = pd.bdate_range("2025-01-01", periods=2)
        rvjp = pd.DataFrame(
            {"A": [0.01, 0.02], "B": [0.005, 0.001]}, index=dates
        )
        large_pos = pd.DataFrame(
            {"A": [0.005, 0.030], "B": [0.010, 0.002]}, index=dates
        )

        result = factor.compute(rvjp=rvjp, large_positive_jump=large_pos)

        assert result.shape == (2, 2)
        np.testing.assert_almost_equal(result.loc[dates[0], "A"], 0.005)
        np.testing.assert_almost_equal(result.loc[dates[0], "B"], 0.005)
        np.testing.assert_almost_equal(result.loc[dates[1], "A"], 0.020)
        np.testing.assert_almost_equal(result.loc[dates[1], "B"], 0.001)

    def test_nan_propagation(self, factor):
        """输入含 NaN 时，输出对应位置也应为 NaN。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        rvjp = pd.DataFrame({"A": [0.01, np.nan, 0.005]}, index=dates)
        large_pos = pd.DataFrame({"A": [0.005, 0.010, np.nan]}, index=dates)

        result = factor.compute(rvjp=rvjp, large_positive_jump=large_pos)

        assert not np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert np.isnan(result.iloc[2, 0])


class TestRVLJPOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=10)
        stocks = ["A", "B", "C"]
        rvjp = pd.DataFrame(
            np.random.uniform(0, 0.01, (10, 3)), index=dates, columns=stocks
        )
        large_pos = pd.DataFrame(
            np.random.uniform(0, 0.01, (10, 3)), index=dates, columns=stocks
        )

        result = factor.compute(rvjp=rvjp, large_positive_jump=large_pos)

        assert result.shape == rvjp.shape
        assert list(result.columns) == list(rvjp.columns)
        assert list(result.index) == list(rvjp.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=3)
        rvjp = pd.DataFrame({"A": [0.01, 0.02, 0.03]}, index=dates)
        large_pos = pd.DataFrame({"A": [0.005, 0.025, 0.01]}, index=dates)

        result = factor.compute(rvjp=rvjp, large_positive_jump=large_pos)
        assert isinstance(result, pd.DataFrame)

    def test_result_non_negative(self, factor):
        """输入非负时，结果也应非负。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        rvjp = pd.DataFrame(
            np.random.uniform(0, 0.01, (5, 2)), index=dates, columns=["A", "B"]
        )
        large_pos = pd.DataFrame(
            np.random.uniform(0, 0.01, (5, 2)), index=dates, columns=["A", "B"]
        )

        result = factor.compute(rvjp=rvjp, large_positive_jump=large_pos)

        assert (result.values >= 0).all()
