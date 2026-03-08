import numpy as np
import pandas as pd
import pytest

from factors.search_ratio_downstream import SearchRatioDownstreamFactor


@pytest.fixture
def factor():
    return SearchRatioDownstreamFactor()


class TestSearchRatioDownstreamMetadata:
    def test_name(self, factor):
        assert factor.name == "SEARCH_RATIO_DOWNSTREAM"

    def test_category(self, factor):
        assert factor.category == "动量溢出"

    def test_repr(self, factor):
        assert "SEARCH_RATIO_DOWNSTREAM" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "SEARCH_RATIO_DOWNSTREAM"
        assert meta["category"] == "动量溢出"
        assert "搜索" in meta["description"] or "动量" in meta["description"]


class TestSearchRatioDownstreamCompute:
    """测试 compute 方法。"""

    def test_basic_weighted_return(self, factor):
        """验证加权收益率计算。

        weights:
          A -> B: 0.3, A -> C: 0.7
          B -> A: 0.5, B -> C: 0.5
        returns: A=0.02, B=-0.01, C=0.03

        R_A = (0.3*(-0.01) + 0.7*0.03) / (0.3+0.7) = (-0.003 + 0.021) / 1.0 = 0.018
        R_B = (0.5*0.02 + 0.5*0.03) / (0.5+0.5) = (0.01 + 0.015) / 1.0 = 0.025
        """
        stocks = ["A", "B", "C"]
        weights = pd.DataFrame(
            [[0.0, 0.3, 0.7],
             [0.5, 0.0, 0.5],
             [0.0, 0.0, 0.0]],
            index=stocks, columns=stocks,
        )
        returns = pd.Series([0.02, -0.01, 0.03], index=stocks)

        result = factor.compute(search_weights=weights, returns=returns)

        assert result["A"] == pytest.approx(0.018)
        assert result["B"] == pytest.approx(0.025)

    def test_uniform_weights(self, factor):
        """等权重时，结果为简单平均。"""
        stocks = ["A", "B", "C"]
        weights = pd.DataFrame(
            [[0.0, 1.0, 1.0],
             [1.0, 0.0, 1.0],
             [1.0, 1.0, 0.0]],
            index=stocks, columns=stocks,
        )
        returns = pd.Series([0.01, 0.02, 0.03], index=stocks)

        result = factor.compute(search_weights=weights, returns=returns)

        # R_A = (1*0.02 + 1*0.03) / 2 = 0.025
        assert result["A"] == pytest.approx(0.025)
        # R_B = (1*0.01 + 1*0.03) / 2 = 0.02
        assert result["B"] == pytest.approx(0.02)
        # R_C = (1*0.01 + 1*0.02) / 2 = 0.015
        assert result["C"] == pytest.approx(0.015)

    def test_single_connection(self, factor):
        """只有一个关联公司时，结果等于该公司收益率。"""
        stocks = ["A", "B", "C"]
        weights = pd.DataFrame(
            [[0.0, 0.5, 0.0],
             [0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]],
            index=stocks, columns=stocks,
        )
        returns = pd.Series([0.01, 0.02, 0.03], index=stocks)

        result = factor.compute(search_weights=weights, returns=returns)

        assert result["A"] == pytest.approx(0.02)

    def test_zero_weights_gives_nan(self, factor):
        """权重全为 0 时，结果应为 NaN。"""
        stocks = ["A", "B"]
        weights = pd.DataFrame(
            [[0.0, 0.0],
             [0.0, 0.0]],
            index=stocks, columns=stocks,
        )
        returns = pd.Series([0.01, 0.02], index=stocks)

        result = factor.compute(search_weights=weights, returns=returns)

        assert np.isnan(result["A"])
        assert np.isnan(result["B"])

    def test_asymmetric_weights(self, factor):
        """非对称权重矩阵。"""
        stocks = ["A", "B"]
        weights = pd.DataFrame(
            [[0.0, 0.8],
             [0.2, 0.0]],
            index=stocks, columns=stocks,
        )
        returns = pd.Series([0.05, -0.03], index=stocks)

        result = factor.compute(search_weights=weights, returns=returns)

        # R_A = 0.8 * (-0.03) / 0.8 = -0.03
        assert result["A"] == pytest.approx(-0.03)
        # R_B = 0.2 * 0.05 / 0.2 = 0.05
        assert result["B"] == pytest.approx(0.05)


class TestSearchRatioDownstreamEdgeCases:
    def test_partial_overlap(self, factor):
        """search_weights 的 columns 与 returns 的 index 部分重叠。"""
        stocks_w = ["A", "B", "C"]
        stocks_r = ["B", "C", "D"]
        weights = pd.DataFrame(
            [[0.0, 0.3, 0.7],
             [0.5, 0.0, 0.5],
             [0.2, 0.3, 0.0]],
            index=stocks_w, columns=stocks_w,
        )
        returns = pd.Series([0.02, 0.03, 0.01], index=stocks_r)

        result = factor.compute(search_weights=weights, returns=returns)

        # 只有 B, C 是公共的
        # R_A = (0.3*0.02 + 0.7*0.03) / (0.3+0.7) = 0.027
        assert result["A"] == pytest.approx(0.027)

    def test_returns_is_series(self, factor):
        """确认输入为 Series 时正常工作。"""
        stocks = ["A", "B"]
        weights = pd.DataFrame(
            [[0.0, 1.0], [1.0, 0.0]], index=stocks, columns=stocks
        )
        returns = pd.Series([0.01, 0.02], index=stocks)

        result = factor.compute(search_weights=weights, returns=returns)

        assert isinstance(result, pd.Series)


class TestSearchRatioDownstreamOutputShape:
    def test_output_length(self, factor):
        """输出长度应等于 search_weights 的行数。"""
        stocks = ["A", "B", "C", "D"]
        weights = pd.DataFrame(
            np.random.uniform(0, 1, (4, 4)), index=stocks, columns=stocks
        )
        returns = pd.Series(np.random.randn(4) * 0.02, index=stocks)

        result = factor.compute(search_weights=weights, returns=returns)

        assert len(result) == 4

    def test_output_is_series(self, factor):
        stocks = ["A", "B", "C"]
        weights = pd.DataFrame(
            np.random.uniform(0, 1, (3, 3)), index=stocks, columns=stocks
        )
        returns = pd.Series([0.01, 0.02, 0.03], index=stocks)

        result = factor.compute(search_weights=weights, returns=returns)
        assert isinstance(result, pd.Series)
