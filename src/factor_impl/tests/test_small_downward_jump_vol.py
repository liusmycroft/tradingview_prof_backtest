import numpy as np
import pandas as pd
import pytest

from factors.small_downward_jump_vol import SmallDownwardJumpVolFactor


@pytest.fixture
def factor():
    return SmallDownwardJumpVolFactor()


class TestSmallDownwardJumpVolMetadata:
    def test_name(self, factor):
        assert factor.name == "SMALL_DOWNWARD_JUMP_VOL"

    def test_category(self, factor):
        assert factor.category == "高频波动跳跃"

    def test_repr(self, factor):
        assert "SMALL_DOWNWARD_JUMP_VOL" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "SMALL_DOWNWARD_JUMP_VOL"
        assert meta["category"] == "高频波动跳跃"


class TestSmallDownwardJumpVolHandCalculated:
    def test_basic_subtraction(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        rvjn = pd.DataFrame([0.010, 0.020, 0.015, 0.025, 0.030], index=dates, columns=["A"])
        rvljn = pd.DataFrame([0.003, 0.008, 0.005, 0.010, 0.012], index=dates, columns=["A"])
        result = factor.compute(rvjn=rvjn, rvljn=rvljn)
        expected = [0.007, 0.012, 0.010, 0.015, 0.018]
        np.testing.assert_array_almost_equal(result["A"].values, expected)

    def test_zero_when_equal(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame([0.01, 0.02, 0.03, 0.04, 0.05], index=dates, columns=["A"])
        result = factor.compute(rvjn=data, rvljn=data)
        np.testing.assert_array_almost_equal(result["A"].values, 0.0)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        rvjn = pd.DataFrame({"A": [0.01, 0.02, 0.03], "B": [0.05, 0.06, 0.07]}, index=dates)
        rvljn = pd.DataFrame({"A": [0.005, 0.01, 0.015], "B": [0.02, 0.03, 0.04]}, index=dates)
        result = factor.compute(rvjn=rvjn, rvljn=rvljn)
        np.testing.assert_array_almost_equal(result["A"].values, [0.005, 0.01, 0.015])
        np.testing.assert_array_almost_equal(result["B"].values, [0.03, 0.03, 0.03])


class TestSmallDownwardJumpVolEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        rvjn = pd.DataFrame([0.01, np.nan, 0.03, 0.04, 0.05], index=dates, columns=["A"])
        rvljn = pd.DataFrame([0.005, 0.01, 0.015, 0.02, 0.025], index=dates, columns=["A"])
        result = factor.compute(rvjn=rvjn, rvljn=rvljn)
        assert isinstance(result, pd.DataFrame)
        assert np.isnan(result.iloc[1, 0])

    def test_all_zero(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame(0.0, index=dates, columns=["A"])
        result = factor.compute(rvjn=data, rvljn=data)
        np.testing.assert_array_almost_equal(result["A"].values, 0.0)


class TestSmallDownwardJumpVolOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        rvjn = pd.DataFrame(np.random.uniform(0.01, 0.05, (30, 3)), index=dates, columns=stocks)
        rvljn = pd.DataFrame(np.random.uniform(0.001, 0.02, (30, 3)), index=dates, columns=stocks)
        result = factor.compute(rvjn=rvjn, rvljn=rvljn)
        assert result.shape == rvjn.shape
        assert list(result.columns) == list(rvjn.columns)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        rvjn = pd.DataFrame([0.01, 0.02, 0.03, 0.04, 0.05], index=dates, columns=["A"])
        rvljn = pd.DataFrame([0.005, 0.01, 0.015, 0.02, 0.025], index=dates, columns=["A"])
        result = factor.compute(rvjn=rvjn, rvljn=rvljn)
        assert isinstance(result, pd.DataFrame)
