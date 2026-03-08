import numpy as np
import pandas as pd
import pytest

from factors.supply_chain_position import SupplyChainPositionFactor


@pytest.fixture
def factor():
    return SupplyChainPositionFactor()


class TestSupplyChainPositionMetadata:
    def test_name(self, factor):
        assert factor.name == "SUPPLY_CHAIN_POSITION"

    def test_category(self, factor):
        assert factor.category == "网络结构"

    def test_repr(self, factor):
        assert "SUPPLY_CHAIN_POSITION" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "SUPPLY_CHAIN_POSITION"
        assert meta["category"] == "网络结构"


class TestSupplyChainPositionHandCalculated:
    def test_simple_3x3(self, factor):
        """3 个公司的简单关联矩阵。

        weights:
            A  B  C
        A  [0, 0.3, 0.2]
        B  [0.3, 0, 0.5]
        C  [0.2, 0.5, 0]

        A: 0.3 + 0.2 = 0.5
        B: 0.3 + 0.5 = 0.8
        C: 0.2 + 0.5 = 0.7
        """
        companies = ["A", "B", "C"]
        w = pd.DataFrame(
            [[0.0, 0.3, 0.2],
             [0.3, 0.0, 0.5],
             [0.2, 0.5, 0.0]],
            index=companies, columns=companies,
        )

        result = factor.compute(weights=w)

        assert result["A"] == pytest.approx(0.5, rel=1e-10)
        assert result["B"] == pytest.approx(0.8, rel=1e-10)
        assert result["C"] == pytest.approx(0.7, rel=1e-10)

    def test_asymmetric_weights(self, factor):
        """非对称权重矩阵。

        weights:
            A  B
        A  [0, 0.3]
        B  [0.1, 0]

        A: 0.3
        B: 0.1
        """
        companies = ["A", "B"]
        w = pd.DataFrame(
            [[0.0, 0.3],
             [0.1, 0.0]],
            index=companies, columns=companies,
        )

        result = factor.compute(weights=w)

        assert result["A"] == pytest.approx(0.3, rel=1e-10)
        assert result["B"] == pytest.approx(0.1, rel=1e-10)

    def test_diagonal_ignored(self, factor):
        """对角线上的值应被忽略。"""
        companies = ["A", "B"]
        w = pd.DataFrame(
            [[5.0, 0.3],
             [0.1, 10.0]],
            index=companies, columns=companies,
        )

        result = factor.compute(weights=w)

        assert result["A"] == pytest.approx(0.3, rel=1e-10)
        assert result["B"] == pytest.approx(0.1, rel=1e-10)


class TestSupplyChainPositionEdgeCases:
    def test_all_zeros(self, factor):
        """全零权重矩阵。"""
        companies = ["A", "B"]
        w = pd.DataFrame(
            [[0.0, 0.0],
             [0.0, 0.0]],
            index=companies, columns=companies,
        )

        result = factor.compute(weights=w)
        assert result["A"] == pytest.approx(0.0, abs=1e-15)
        assert result["B"] == pytest.approx(0.0, abs=1e-15)

    def test_nan_in_weights(self, factor):
        """含 NaN 的权重，nansum 应跳过。"""
        companies = ["A", "B", "C"]
        w = pd.DataFrame(
            [[0.0, np.nan, 0.2],
             [0.3, 0.0, 0.5],
             [0.2, 0.5, 0.0]],
            index=companies, columns=companies,
        )

        result = factor.compute(weights=w)
        # A: nansum(nan, 0.2) = 0.2
        assert result["A"] == pytest.approx(0.2, rel=1e-10)

    def test_single_company(self, factor):
        """单个公司时，得分为 0。"""
        companies = ["A"]
        w = pd.DataFrame([[0.0]], index=companies, columns=companies)

        result = factor.compute(weights=w)
        assert result["A"] == pytest.approx(0.0, abs=1e-15)

    def test_output_type(self, factor):
        companies = ["A", "B", "C"]
        w = pd.DataFrame(
            np.random.uniform(0, 1, (3, 3)),
            index=companies, columns=companies,
        )
        result = factor.compute(weights=w)
        assert isinstance(result, pd.Series)
        assert len(result) == 3
