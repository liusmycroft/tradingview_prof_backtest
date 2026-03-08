import numpy as np
import pandas as pd
import pytest

from factors.linkage import LinkageFactor


@pytest.fixture
def factor():
    return LinkageFactor()


class TestLinkageMetadata:
    def test_name(self, factor):
        assert factor.name == "LINKAGE"

    def test_category(self, factor):
        assert factor.category == "图谱网络-动量溢出"

    def test_repr(self, factor):
        assert "LINKAGE" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "LINKAGE"
        assert meta["category"] == "图谱网络-动量溢出"


class TestLinkageHandCalculated:
    def test_simple_three_stocks(self, factor):
        """手动验证三只股票的联动因子。

        similarity = [[1, 0.8, 0.3], [0.8, 1, 0.5], [0.3, 0.5, 1]]
        returns = [0.05, -0.02, 0.10]

        对角线置零后:
        A: weighted_ret = (0.8*(-0.02) + 0.3*0.10) / (0.8+0.3) = 0.014/1.1 = 0.01273
           linkage_A = 0.01273 - 0.05 = -0.03727
        B: weighted_ret = (0.8*0.05 + 0.5*0.10) / (0.8+0.5) = 0.09/1.3 = 0.06923
           linkage_B = 0.06923 - (-0.02) = 0.08923
        C: weighted_ret = (0.3*0.05 + 0.5*(-0.02)) / (0.3+0.5) = 0.005/0.8 = 0.00625
           linkage_C = 0.00625 - 0.10 = -0.09375
        """
        stocks = ["A", "B", "C"]
        similarity = pd.DataFrame(
            [[1.0, 0.8, 0.3], [0.8, 1.0, 0.5], [0.3, 0.5, 1.0]],
            index=stocks, columns=stocks,
        )
        returns = pd.Series({"A": 0.05, "B": -0.02, "C": 0.10})

        result = factor.compute(similarity=similarity, returns=returns)

        assert result["A"] == pytest.approx(0.014 / 1.1 - 0.05, rel=1e-4)
        assert result["B"] == pytest.approx(0.09 / 1.3 - (-0.02), rel=1e-4)
        assert result["C"] == pytest.approx(0.005 / 0.8 - 0.10, rel=1e-4)

    def test_equal_similarity_equal_returns(self, factor):
        """所有相似度相同且收益相同时, linkage 应为 0。"""
        stocks = ["A", "B", "C"]
        similarity = pd.DataFrame(
            [[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]],
            index=stocks, columns=stocks,
        )
        returns = pd.Series({"A": 0.03, "B": 0.03, "C": 0.03})

        result = factor.compute(similarity=similarity, returns=returns)
        for s in stocks:
            assert result[s] == pytest.approx(0.0, abs=1e-10)

    def test_output_is_series(self, factor):
        stocks = ["A", "B"]
        similarity = pd.DataFrame(
            [[1.0, 0.6], [0.6, 1.0]], index=stocks, columns=stocks,
        )
        returns = pd.Series({"A": 0.01, "B": 0.02})

        result = factor.compute(similarity=similarity, returns=returns)
        assert isinstance(result, pd.Series)


class TestLinkageEdgeCases:
    def test_nan_in_returns(self, factor):
        stocks = ["A", "B", "C"]
        similarity = pd.DataFrame(
            [[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]],
            index=stocks, columns=stocks,
        )
        returns = pd.Series({"A": 0.01, "B": np.nan, "C": 0.03})

        result = factor.compute(similarity=similarity, returns=returns)
        assert isinstance(result, pd.Series)

    def test_zero_similarity(self, factor):
        """所有相似度为零时, 加权收益分母为0, 应返回 NaN。"""
        stocks = ["A", "B", "C"]
        similarity = pd.DataFrame(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            index=stocks, columns=stocks,
        )
        returns = pd.Series({"A": 0.01, "B": 0.02, "C": 0.03})

        result = factor.compute(similarity=similarity, returns=returns)
        assert result.isna().all()

    def test_partial_overlap(self, factor):
        """similarity 和 returns 只有部分股票重叠。"""
        stocks_sim = ["A", "B", "C"]
        similarity = pd.DataFrame(
            [[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]],
            index=stocks_sim, columns=stocks_sim,
        )
        returns = pd.Series({"A": 0.01, "B": 0.02, "D": 0.04})

        result = factor.compute(similarity=similarity, returns=returns)
        assert len(result) == 2
        assert "A" in result.index
        assert "B" in result.index


class TestLinkageOutputShape:
    def test_output_shape_matches_common(self, factor):
        stocks = ["A", "B", "C", "D"]
        similarity = pd.DataFrame(
            np.eye(4) + 0.3, index=stocks, columns=stocks,
        )
        returns = pd.Series({"A": 0.01, "B": 0.02, "C": 0.03, "D": 0.04})

        result = factor.compute(similarity=similarity, returns=returns)
        assert len(result) == 4

    def test_output_is_series(self, factor):
        stocks = ["A", "B", "C"]
        similarity = pd.DataFrame(
            [[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]],
            index=stocks, columns=stocks,
        )
        returns = pd.Series({"A": 0.01, "B": 0.02, "C": 0.03})

        result = factor.compute(similarity=similarity, returns=returns)
        assert isinstance(result, pd.Series)
