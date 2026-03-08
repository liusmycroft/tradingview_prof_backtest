import numpy as np
import pandas as pd
import pytest

from factors.complex_momentum import ComplexMomentumFactor


@pytest.fixture
def factor():
    return ComplexMomentumFactor()


class TestComplexMomentumMetadata:
    def test_name(self, factor):
        assert factor.name == "COMPLEX_MOMENTUM"

    def test_category(self, factor):
        assert factor.category == "动量溢出"

    def test_repr(self, factor):
        assert "COMPLEX_MOMENTUM" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "COMPLEX_MOMENTUM"
        assert meta["category"] == "动量溢出"


class TestComplexMomentumHandCalculated:
    """用手算数据验证 COMRET = sum(W_ij * RET_j) / sum(W_ij) 的正确性。"""

    def test_known_values(self, factor):
        """已知权重和收益率验证加权计算。

        A: (0.8*0.02 + 0.2*(-0.01)) / (0.8+0.2) = 0.014
        B: (0.3*0.02 + 0.7*(-0.01)) / (0.3+0.7) = -0.001
        """
        stocks = ["A", "B"]
        industries = ["银行", "地产"]

        weights = pd.DataFrame(
            [[0.8, 0.2], [0.3, 0.7]], index=stocks, columns=industries
        )
        returns = pd.Series([0.02, -0.01], index=industries)

        result = factor.compute(industry_weights=weights, industry_returns=returns)

        assert result["A"] == pytest.approx(0.014, rel=1e-6)
        assert result["B"] == pytest.approx(-0.001, rel=1e-6)

    def test_single_industry(self, factor):
        """公司只在一个行业有营收时, COMRET 等于该行业收益率。"""
        weights = pd.DataFrame(
            [[1.0, 0.0], [0.0, 1.0]],
            index=["A", "B"],
            columns=["I1", "I2"],
        )
        returns = pd.Series([0.05, -0.03], index=["I1", "I2"])

        result = factor.compute(industry_weights=weights, industry_returns=returns)
        assert result["A"] == pytest.approx(0.05, rel=1e-10)
        assert result["B"] == pytest.approx(-0.03, rel=1e-10)

    def test_partial_industry_overlap(self, factor):
        """权重和收益率行业不完全重叠时, 只用交集。

        weights: I1=0.5, I2=0.3, I3=0.2
        returns: I1=0.02, I2=-0.01 (无 I3)
        result: (0.5*0.02 + 0.3*(-0.01)) / (0.5+0.3) = 0.007/0.8 = 0.00875
        """
        weights = pd.DataFrame(
            [[0.5, 0.3, 0.2]], index=["A"], columns=["I1", "I2", "I3"]
        )
        returns = pd.Series([0.02, -0.01], index=["I1", "I2"])

        result = factor.compute(industry_weights=weights, industry_returns=returns)
        assert result["A"] == pytest.approx(0.00875, rel=1e-6)

    def test_three_stocks_three_industries(self, factor):
        """三只股票三个行业, 来自因子文件示例。

        weights = [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.0, 0.1, 0.9]]
        returns = [0.02, -0.01, 0.03]

        A: (0.8*0.02 + 0.1*(-0.01) + 0.1*0.03) / 1.0 = 0.018
        B: (0.2*0.02 + 0.7*(-0.01) + 0.1*0.03) / 1.0 = 0.0
        C: (0.0*0.02 + 0.1*(-0.01) + 0.9*0.03) / 1.0 = 0.026
        """
        stocks = ["A", "B", "C"]
        industries = ["银行", "地产", "科技"]

        weights = pd.DataFrame(
            [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.0, 0.1, 0.9]],
            index=stocks,
            columns=industries,
        )
        returns = pd.Series([0.02, -0.01, 0.03], index=industries)

        result = factor.compute(industry_weights=weights, industry_returns=returns)

        assert result["A"] == pytest.approx(0.018, rel=1e-10)
        assert result["B"] == pytest.approx(0.0, abs=1e-15)
        assert result["C"] == pytest.approx(0.026, rel=1e-10)


class TestComplexMomentumEdgeCases:
    def test_zero_weights_returns_nan(self, factor):
        """权重全为零时, weight_sum=0 -> NaN。"""
        weights = pd.DataFrame(
            [[0.0, 0.0]], index=["A"], columns=["I1", "I2"]
        )
        returns = pd.Series([0.01, 0.02], index=["I1", "I2"])

        result = factor.compute(industry_weights=weights, industry_returns=returns)
        assert pd.isna(result["A"])

    def test_no_industry_overlap(self, factor):
        """权重和收益率无交集行业时, 结果为 NaN。"""
        weights = pd.DataFrame(
            [[0.5, 0.5]], index=["A"], columns=["I1", "I2"]
        )
        returns = pd.Series([0.01, 0.02], index=["I3", "I4"])

        result = factor.compute(industry_weights=weights, industry_returns=returns)
        assert pd.isna(result["A"])

    def test_nan_in_weights(self, factor):
        """权重含 NaN 时, 不应抛异常。"""
        weights = pd.DataFrame(
            [[0.5, np.nan]], index=["A"], columns=["I1", "I2"]
        )
        returns = pd.Series([0.02, 0.01], index=["I1", "I2"])

        result = factor.compute(industry_weights=weights, industry_returns=returns)
        assert isinstance(result, pd.Series)

    def test_nan_in_returns(self, factor):
        """收益率含 NaN 时, 不应抛异常。"""
        weights = pd.DataFrame(
            [[0.5, 0.5]], index=["A"], columns=["I1", "I2"]
        )
        returns = pd.Series([0.02, np.nan], index=["I1", "I2"])

        result = factor.compute(industry_weights=weights, industry_returns=returns)
        assert isinstance(result, pd.Series)


class TestComplexMomentumOutputShape:
    def test_output_is_series(self, factor):
        """输出应为 pd.Series。"""
        weights = pd.DataFrame(
            [[0.5, 0.5]], index=["A"], columns=["I1", "I2"]
        )
        returns = pd.Series([0.01, 0.02], index=["I1", "I2"])

        result = factor.compute(industry_weights=weights, industry_returns=returns)
        assert isinstance(result, pd.Series)

    def test_output_index_matches_stocks(self, factor):
        """输出 index 应与 industry_weights 的 index 一致。"""
        stocks = ["A", "B", "C"]
        industries = ["I1", "I2"]
        weights = pd.DataFrame(
            [[0.5, 0.5], [0.3, 0.7], [0.8, 0.2]],
            index=stocks,
            columns=industries,
        )
        returns = pd.Series([0.01, 0.02], index=industries)

        result = factor.compute(industry_weights=weights, industry_returns=returns)
        assert list(result.index) == stocks
        assert len(result) == 3

    def test_output_name(self, factor):
        """输出 Series 的 name 应为因子名。"""
        weights = pd.DataFrame(
            [[0.5, 0.5]], index=["A"], columns=["I1", "I2"]
        )
        returns = pd.Series([0.01, 0.02], index=["I1", "I2"])

        result = factor.compute(industry_weights=weights, industry_returns=returns)
        assert result.name == "COMPLEX_MOMENTUM"
