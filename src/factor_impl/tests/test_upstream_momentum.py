import numpy as np
import pandas as pd
import pytest

from factors.upstream_momentum import UpstreamMomentumFactor


@pytest.fixture
def factor():
    return UpstreamMomentumFactor()


class TestUpstreamMomentumMetadata:
    def test_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "UpstreamMomentum"
        assert meta["category"] == "动量"
        assert meta["description"] != ""


class TestUpstreamMomentumCompute:
    def test_three_company_supply_chain(self, factor):
        """用 3 家公司的简单供应链验证加权平均计算。"""
        weights = pd.DataFrame(
            {
                "A": [0.0, 0.3, 0.1],
                "B": [0.4, 0.0, 0.2],
                "C": [0.2, 0.5, 0.0],
            },
            index=["A", "B", "C"],
        )
        returns = pd.Series({"A": 0.05, "B": -0.02, "C": 0.10})

        result = factor.compute(weights=weights, returns=returns)

        # A 的下游传导动量: (0.4*(-0.02) + 0.2*0.10) / (0.4+0.2)
        expected_a = (0.4 * (-0.02) + 0.2 * 0.10) / (0.4 + 0.2)
        # B 的下游传导动量: (0.3*0.05 + 0.5*0.10) / (0.3+0.5)
        expected_b = (0.3 * 0.05 + 0.5 * 0.10) / (0.3 + 0.5)
        # C 的下游传导动量: (0.1*0.05 + 0.2*(-0.02)) / (0.1+0.2)
        expected_c = (0.1 * 0.05 + 0.2 * (-0.02)) / (0.1 + 0.2)

        assert pytest.approx(result["A"], rel=1e-6) == expected_a
        assert pytest.approx(result["B"], rel=1e-6) == expected_b
        assert pytest.approx(result["C"], rel=1e-6) == expected_c

    def test_self_exclusion(self, factor):
        """对角线权重应被置零，自身收益不参与计算。"""
        # 故意在对角线上放非零权重
        weights = pd.DataFrame(
            {
                "A": [0.9, 0.0],
                "B": [0.0, 0.8],
            },
            index=["A", "B"],
        )
        returns = pd.Series({"A": 0.50, "B": 0.01})

        result = factor.compute(weights=weights, returns=returns)

        # A: 对角线 0.9 被清零，只剩 wgt(A,B)=0.0 -> 权重和为 0 -> NaN
        assert pd.isna(result["A"])
        # B: 对角线 0.8 被清零，只剩 wgt(B,A)=0.0 -> 权重和为 0 -> NaN
        assert pd.isna(result["B"])

    def test_zero_weights_row(self, factor):
        """某上游公司对所有下游权重为 0 时，结果应为 NaN。"""
        weights = pd.DataFrame(
            {
                "A": [0.0, 0.0],
                "B": [0.0, 0.0],
            },
            index=["A", "B"],
        )
        returns = pd.Series({"A": 0.05, "B": 0.10})

        result = factor.compute(weights=weights, returns=returns)

        assert pd.isna(result["A"])
        assert pd.isna(result["B"])

    def test_single_company(self, factor):
        """只有一家公司时，排除自身后权重和为 0，结果应为 NaN。"""
        weights = pd.DataFrame({"A": [1.0]}, index=["A"])
        returns = pd.Series({"A": 0.05})

        result = factor.compute(weights=weights, returns=returns)

        assert len(result) == 1
        assert pd.isna(result["A"])

    def test_nan_in_returns(self, factor):
        """下游收益含 NaN 时，加权结果应正确传播 NaN。"""
        weights = pd.DataFrame(
            {
                "A": [0.0, 0.3],
                "B": [0.4, 0.0],
            },
            index=["A", "B"],
        )
        returns = pd.Series({"A": 0.05, "B": np.nan})

        result = factor.compute(weights=weights, returns=returns)

        # A 的唯一下游 B 收益为 NaN -> 结果为 NaN
        assert pd.isna(result["A"])
        # B 的唯一下游 A 收益正常 -> 0.3*0.05 / 0.3 = 0.05
        assert pytest.approx(result["B"], rel=1e-6) == 0.05

    def test_nan_in_weights(self, factor):
        """权重矩阵含 NaN 时，结果应传播 NaN。"""
        weights = pd.DataFrame(
            {
                "A": [0.0, np.nan],
                "B": [0.4, 0.0],
            },
            index=["A", "B"],
        )
        returns = pd.Series({"A": 0.05, "B": 0.10})

        result = factor.compute(weights=weights, returns=returns)

        # A: wgt(A,B)=0.4, rtn_B=0.10 -> 0.4*0.10/0.4 = 0.10
        assert pytest.approx(result["A"], rel=1e-6) == 0.10
        # B: wgt(B,A)=NaN -> dot 产生 NaN
        assert pd.isna(result["B"])

    def test_asymmetric_weights(self, factor):
        """非对称权重矩阵：上下游关系不对称。"""
        weights = pd.DataFrame(
            {
                "A": [0.0, 0.0],
                "B": [0.6, 0.0],
            },
            index=["A", "B"],
        )
        returns = pd.Series({"A": 0.05, "B": -0.03})

        result = factor.compute(weights=weights, returns=returns)

        # A: wgt(A,B)=0.6 -> 0.6*(-0.03)/0.6 = -0.03
        assert pytest.approx(result["A"], rel=1e-6) == -0.03
        # B: 对 A 的权重为 0 -> 权重和为 0 -> NaN
        assert pd.isna(result["B"])

    def test_output_type_and_index(self, factor):
        """输出应为 pd.Series，index 与权重矩阵行索引一致。"""
        weights = pd.DataFrame(
            {
                "X": [0.0, 0.2, 0.1],
                "Y": [0.3, 0.0, 0.4],
                "Z": [0.5, 0.1, 0.0],
            },
            index=["X", "Y", "Z"],
        )
        returns = pd.Series({"X": 0.01, "Y": 0.02, "Z": 0.03})

        result = factor.compute(weights=weights, returns=returns)

        assert isinstance(result, pd.Series)
        assert list(result.index) == ["X", "Y", "Z"]
        assert result.name == "UpstreamMomentum"
