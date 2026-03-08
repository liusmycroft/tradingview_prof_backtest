import numpy as np
import pandas as pd
import pytest

from factors.katz_centrality import KatzCentralityFactor


@pytest.fixture
def factor():
    return KatzCentralityFactor()


class TestKatzCentralityMetadata:
    def test_name(self, factor):
        assert factor.name == "KATZ_CENTRALITY"

    def test_category(self, factor):
        assert factor.category == "图谱网络-网络结构"

    def test_repr(self, factor):
        assert "KATZ_CENTRALITY" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "KATZ_CENTRALITY"
        assert meta["category"] == "图谱网络-网络结构"


class TestKatzCentralityHandCalculated:
    def test_identity_adjacency(self, factor):
        """A = I (identity), lam=0.1, beta=1.0.
        k = 0.1 * I @ k + 1 => k = 0.1*k + 1 => k = 1/0.9 ~ 1.1111
        """
        stocks = ["A", "B"]
        adj = pd.DataFrame(
            np.eye(2), index=stocks, columns=stocks
        )

        result = factor.compute(adjacency=adj, lam=0.1, beta=1.0)
        expected = 1.0 / 0.9
        np.testing.assert_allclose(result.values[0], expected, atol=1e-5)

    def test_zero_adjacency(self, factor):
        """A = 0, k = lam*0*k + beta = beta for all nodes."""
        stocks = ["A", "B", "C"]
        adj = pd.DataFrame(0.0, index=stocks, columns=stocks)

        result = factor.compute(adjacency=adj, lam=0.1, beta=1.0)
        np.testing.assert_allclose(result.values[0], 1.0, atol=1e-10)

    def test_cycle_graph(self, factor):
        """3-node cycle: A->B->C->A. By symmetry all nodes have same centrality."""
        stocks = ["A", "B", "C"]
        adj = pd.DataFrame(
            [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
            index=stocks, columns=stocks, dtype=float,
        )

        result = factor.compute(adjacency=adj, lam=0.1, beta=1.0)
        vals = result.values[0]
        assert vals[0] == pytest.approx(vals[1], rel=1e-5)
        assert vals[1] == pytest.approx(vals[2], rel=1e-5)


class TestKatzCentralityEdgeCases:
    def test_nan_in_adjacency(self, factor):
        """NaN in adjacency should propagate to result."""
        stocks = ["A", "B"]
        adj = pd.DataFrame(
            [[0.0, np.nan], [1.0, 0.0]], index=stocks, columns=stocks
        )

        result = factor.compute(adjacency=adj, lam=0.1, beta=1.0)
        assert isinstance(result, pd.DataFrame)

    def test_all_nan(self, factor):
        stocks = ["A", "B"]
        adj = pd.DataFrame(np.nan, index=stocks, columns=stocks)

        result = factor.compute(adjacency=adj, lam=0.1, beta=1.0)
        assert result.isna().all().all()

    def test_single_node(self, factor):
        """Single node with no self-loop: k = beta."""
        stocks = ["A"]
        adj = pd.DataFrame([[0.0]], index=stocks, columns=stocks)

        result = factor.compute(adjacency=adj, lam=0.1, beta=1.0)
        assert result.iloc[0, 0] == pytest.approx(1.0, rel=1e-10)


class TestKatzCentralityOutputShape:
    def test_output_shape(self, factor):
        stocks = ["A", "B", "C", "D"]
        adj = pd.DataFrame(
            np.random.rand(4, 4), index=stocks, columns=stocks
        )

        result = factor.compute(adjacency=adj, lam=0.05, beta=1.0)
        assert result.shape == (1, 4)
        assert list(result.columns) == stocks

    def test_output_is_dataframe(self, factor):
        stocks = ["A", "B"]
        adj = pd.DataFrame(
            [[0, 1], [1, 0]], index=stocks, columns=stocks, dtype=float
        )

        result = factor.compute(adjacency=adj, lam=0.1, beta=1.0)
        assert isinstance(result, pd.DataFrame)
