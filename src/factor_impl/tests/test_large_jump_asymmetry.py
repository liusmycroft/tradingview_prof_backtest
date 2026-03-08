import numpy as np
import pandas as pd
import pytest

from factors.large_jump_asymmetry import LargeJumpAsymmetryFactor


@pytest.fixture
def factor():
    return LargeJumpAsymmetryFactor()


class TestLargeJumpAsymmetryMetadata:
    def test_name(self, factor):
        assert factor.name == "LARGE_JUMP_ASYMMETRY"

    def test_category(self, factor):
        assert factor.category == "高频波动跳跃"

    def test_repr(self, factor):
        assert "LARGE_JUMP_ASYMMETRY" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "LARGE_JUMP_ASYMMETRY"
        assert meta["category"] == "高频波动跳跃"


class TestLargeJumpAsymmetryHandCalculated:
    def test_basic_subtraction(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        rvljp = pd.DataFrame([0.010, 0.020, 0.015, 0.025, 0.030], index=dates, columns=["A"])
        rvljn = pd.DataFrame([0.003, 0.008, 0.005, 0.010, 0.012], index=dates, columns=["A"])
        result = factor.compute(rvljp=rvljp, rvljn=rvljn)
        expected = [0.007, 0.012, 0.010, 0.015, 0.018]
        np.testing.assert_array_almost_equal(result["A"].values, expected)

    def test_zero_when_equal(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame([0.01, 0.02, 0.03, 0.04, 0.05], index=dates, columns=["A"])
        result = factor.compute(rvljp=data, rvljn=data)
        np.testing.assert_array_almost_equal(result["A"].values, 0.0)

    def test_negative_result(self, factor):
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        rvljp = pd.DataFrame([0.01, 0.02, 0.03], index=dates, columns=["A"])
        rvljn = pd.DataFrame([0.05, 0.06, 0.07], index=dates, columns=["A"])
        result = factor.compute(rvljp=rvljp, rvljn=rvljn)
        np.testing.assert_array_almost_equal(result["A"].values, [-0.04, -0.04, -0.04])

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        rvljp = pd.DataFrame({"A": [0.01, 0.02, 0.03], "B": [0.05, 0.06, 0.07]}, index=dates)
        rvljn = pd.DataFrame({"A": [0.005, 0.01, 0.015], "B": [0.02, 0.03, 0.04]}, index=dates)
        result = factor.compute(rvljp=rvljp, rvljn=rvljn)
        np.testing.assert_array_almost_equal(result["A"].values, [0.005, 0.01, 0.015])
        np.testing.assert_array_almost_equal(result["B"].values, [0.03, 0.03, 0.03])


class TestLargeJumpAsymmetryEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        rvljp = pd.DataFrame([0.01, np.nan, 0.03, 0.04, 0.05], index=dates, columns=["A"])
        rvljn = pd.DataFrame([0.005, 0.01, 0.015, 0.02, 0.025], index=dates, columns=["A"])
        result = factor.compute(rvljp=rvljp, rvljn=rvljn)
        assert isinstance(result, pd.DataFrame)
        assert np.isnan(result.iloc[1, 0])

    def test_all_zero(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame(0.0, index=dates, columns=["A"])
        result = factor.compute(rvljp=data, rvljn=data)
        np.testing.assert_array_almost_equal(result["A"].values, 0.0)


class TestLargeJumpAsymmetryOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        rvljp = pd.DataFrame(np.random.uniform(0.01, 0.05, (30, 3)), index=dates, columns=stocks)
        rvljn = pd.DataFrame(np.random.uniform(0.001, 0.02, (30, 3)), index=dates, columns=stocks)
        result = factor.compute(rvljp=rvljp, rvljn=rvljn)
        assert result.shape == rvljp.shape
        assert list(result.columns) == list(rvljp.columns)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        rvljp = pd.DataFrame([0.01, 0.02, 0.03, 0.04, 0.05], index=dates, columns=["A"])
        rvljn = pd.DataFrame([0.005, 0.01, 0.015, 0.02, 0.025], index=dates, columns=["A"])
        result = factor.compute(rvljp=rvljp, rvljn=rvljn)
        assert isinstance(result, pd.DataFrame)
