import numpy as np
import pandas as pd
import pytest

from factors.customer_importance import CustomerImportanceFactor


@pytest.fixture
def factor():
    return CustomerImportanceFactor()


class TestCustomerImportanceMetadata:
    def test_name(self, factor):
        assert factor.name == "CUSTOMER_IMPORTANCE"

    def test_category(self, factor):
        assert factor.category == "图谱网络-网络结构"

    def test_repr(self, factor):
        assert "CUSTOMER_IMPORTANCE" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "CUSTOMER_IMPORTANCE"


class TestCustomerImportanceCompute:
    def test_simple_chain(self, factor):
        """A->B->C 链式结构，C 应有最高 PageRank。"""
        adj = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ], dtype=float)
        stocks = ["A", "B", "C"]
        result = factor.compute(adjacency_matrix=adj, stock_list=stocks, q=0.85)
        assert result.loc["C", "pagerank"] > result.loc["A", "pagerank"]

    def test_uniform_graph(self, factor):
        """完全图中所有节点 PageRank 应相等。"""
        adj = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ], dtype=float)
        stocks = ["A", "B", "C"]
        result = factor.compute(adjacency_matrix=adj, stock_list=stocks, q=0.85)
        vals = result["pagerank"].values
        np.testing.assert_array_almost_equal(vals, vals[0], decimal=5)

    def test_isolated_node(self, factor):
        """孤立节点的 PageRank 应为 q/V。"""
        adj = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 0],
        ], dtype=float)
        stocks = ["A", "B", "C"]
        result = factor.compute(adjacency_matrix=adj, stock_list=stocks, q=0.85)
        # A is isolated, only gets q/V
        assert result.loc["A", "pagerank"] == pytest.approx(0.85 / 3, rel=1e-3)

    def test_output_shape(self, factor):
        adj = np.zeros((5, 5))
        stocks = ["A", "B", "C", "D", "E"]
        result = factor.compute(adjacency_matrix=adj, stock_list=stocks)
        assert result.shape == (5, 1)
        assert isinstance(result, pd.DataFrame)
