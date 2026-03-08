import numpy as np
import pandas as pd
import pytest

from factors.modified_business_linkage import ModifiedBusinessLinkageFactor


@pytest.fixture
def factor():
    return ModifiedBusinessLinkageFactor()


class TestModifiedBusinessLinkageMetadata:
    def test_name(self, factor):
        assert factor.name == "MODIFIED_BUSINESS_LINKAGE"

    def test_category(self, factor):
        assert factor.category == "图谱网络-动量溢出"

    def test_repr(self, factor):
        assert "MODIFIED_BUSINESS_LINKAGE" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "MODIFIED_BUSINESS_LINKAGE"
        assert meta["category"] == "图谱网络-动量溢出"


class TestModifiedBusinessLinkageHandCalculated:
    def test_equal_returns(self, factor):
        """sim_weighted_return == own_return 时, linkage=0, rolling mean=0。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        sim_ret = pd.DataFrame(0.05, index=dates, columns=["A"])
        own_ret = pd.DataFrame(0.05, index=dates, columns=["A"])

        result = factor.compute(
            similarity_weighted_return=sim_ret, own_return=own_ret, T=5
        )
        np.testing.assert_array_almost_equal(result["A"].values, 0.0)

    def test_constant_diff(self, factor):
        """sim - own = 0.02 时, rolling mean = 0.02。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        sim_ret = pd.DataFrame(0.07, index=dates, columns=["A"])
        own_ret = pd.DataFrame(0.05, index=dates, columns=["A"])

        result = factor.compute(
            similarity_weighted_return=sim_ret, own_return=own_ret, T=5
        )
        np.testing.assert_array_almost_equal(result["A"].values, 0.02)

    def test_varying_T3(self, factor):
        """T=3, linkage=[0.01, 0.02, 0.03] => mean=0.02。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        sim_ret = pd.DataFrame([0.06, 0.07, 0.08], index=dates, columns=["A"])
        own_ret = pd.DataFrame([0.05, 0.05, 0.05], index=dates, columns=["A"])

        result = factor.compute(
            similarity_weighted_return=sim_ret, own_return=own_ret, T=3
        )
        assert result.iloc[2, 0] == pytest.approx(0.02, rel=1e-6)


class TestModifiedBusinessLinkageEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        sim_vals = np.ones(10) * 0.05
        sim_vals[3] = np.nan
        sim_ret = pd.DataFrame(sim_vals, index=dates, columns=["A"])
        own_ret = pd.DataFrame(0.03, index=dates, columns=["A"])

        result = factor.compute(
            similarity_weighted_return=sim_ret, own_return=own_ret, T=5
        )
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        sim_ret = pd.DataFrame(np.nan, index=dates, columns=["A"])
        own_ret = pd.DataFrame(np.nan, index=dates, columns=["A"])

        result = factor.compute(
            similarity_weighted_return=sim_ret, own_return=own_ret, T=5
        )
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        sim_ret = pd.DataFrame(0.0, index=dates, columns=["A"])
        own_ret = pd.DataFrame(0.0, index=dates, columns=["A"])

        result = factor.compute(
            similarity_weighted_return=sim_ret, own_return=own_ret, T=5
        )
        for val in result["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)


class TestModifiedBusinessLinkageOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        sim_ret = pd.DataFrame(
            np.random.uniform(0, 0.1, (30, 3)), index=dates, columns=stocks
        )
        own_ret = pd.DataFrame(
            np.random.uniform(0, 0.1, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(
            similarity_weighted_return=sim_ret, own_return=own_ret, T=20
        )
        assert result.shape == sim_ret.shape
        assert list(result.columns) == list(sim_ret.columns)
        assert list(result.index) == list(sim_ret.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        sim_ret = pd.DataFrame([0.05, 0.06, 0.07, 0.08, 0.09], index=dates, columns=["A"])
        own_ret = pd.DataFrame(0.05, index=dates, columns=["A"])

        result = factor.compute(
            similarity_weighted_return=sim_ret, own_return=own_ret, T=3
        )
        assert isinstance(result, pd.DataFrame)

    def test_no_leading_nan_min_periods_1(self, factor):
        """min_periods=1, 第一行就有值。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        sim_ret = pd.DataFrame(
            np.random.uniform(0, 0.1, (10, 2)), index=dates, columns=stocks
        )
        own_ret = pd.DataFrame(
            np.random.uniform(0, 0.1, (10, 2)), index=dates, columns=stocks
        )

        result = factor.compute(
            similarity_weighted_return=sim_ret, own_return=own_ret, T=20
        )
        assert result.iloc[0].notna().all()
        assert result.notna().all().all()
