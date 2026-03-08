import numpy as np
import pandas as pd
import pytest

from factors.bottleneck_company import BottleneckCompanyFactor


@pytest.fixture
def factor():
    return BottleneckCompanyFactor()


class TestBottleneckCompanyMetadata:
    def test_name(self, factor):
        assert factor.name == "BOTTLENECK_COMPANY"

    def test_category(self, factor):
        assert factor.category == "图谱网络"

    def test_repr(self, factor):
        assert "BOTTLENECK_COMPANY" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "BOTTLENECK_COMPANY"
        assert meta["category"] == "图谱网络"


class TestBottleneckCompanyCompute:
    def test_passthrough(self, factor):
        """因子应直接返回中介中心性值。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A", "B"]
        bc = pd.DataFrame(
            {"A": [0.1, 0.2, 0.3, 0.4, 0.5],
             "B": [0.5, 0.4, 0.3, 0.2, 0.1]},
            index=dates,
        )

        result = factor.compute(betweenness_centrality=bc)
        pd.testing.assert_frame_equal(result, bc)

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        bc = pd.DataFrame(np.random.rand(30, 3), index=dates, columns=stocks)

        result = factor.compute(betweenness_centrality=bc)
        assert result.shape == (30, 3)

    def test_does_not_modify_input(self, factor):
        """返回的是副本，不应修改原始数据。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        bc = pd.DataFrame([0.1, 0.2, 0.3, 0.4, 0.5], index=dates, columns=stocks)
        original = bc.copy()

        result = factor.compute(betweenness_centrality=bc)
        result.iloc[0, 0] = 999.0
        pd.testing.assert_frame_equal(bc, original)

    def test_zero_centrality(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        bc = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(betweenness_centrality=bc)
        np.testing.assert_array_almost_equal(result.values, 0.0)


class TestBottleneckCompanyEdgeCases:
    def test_single_value(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        bc = pd.DataFrame([0.42], index=dates, columns=["A"])

        result = factor.compute(betweenness_centrality=bc)
        assert result.iloc[0, 0] == pytest.approx(0.42)

    def test_nan_handling(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        bc = pd.DataFrame([0.1, np.nan, 0.3, np.nan, 0.5],
                          index=dates, columns=["A"])

        result = factor.compute(betweenness_centrality=bc)
        assert result.iloc[1, 0] != result.iloc[1, 0]  # NaN check
        assert result.iloc[0, 0] == pytest.approx(0.1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        bc = pd.DataFrame(np.nan, index=dates, columns=["A"])

        result = factor.compute(betweenness_centrality=bc)
        assert result.isna().all().all()
