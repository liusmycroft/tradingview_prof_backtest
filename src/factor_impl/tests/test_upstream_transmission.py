import numpy as np
import pandas as pd
import pytest

from factors.upstream_transmission import UpstreamTransmissionFactor


@pytest.fixture
def factor():
    return UpstreamTransmissionFactor()


class TestUpstreamTransmissionMetadata:
    def test_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "UPSTREAM_TRANSMISSION"
        assert meta["category"] == "动量溢出"
        assert "上游" in meta["description"] or "传导" in meta["description"]

    def test_repr(self, factor):
        r = repr(factor)
        assert "UpstreamTransmissionFactor" in r
        assert "UPSTREAM_TRANSMISSION" in r


class TestUpstreamTransmissionCompute:
    def test_three_company(self, factor):
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

        # A: (0.4*(-0.02) + 0.2*0.10) / (0.4+0.2)
        expected_a = (0.4 * (-0.02) + 0.2 * 0.10) / (0.4 + 0.2)
        # B: (0.3*0.05 + 0.5*0.10) / (0.3+0.5)
        expected_b = (0.3 * 0.05 + 0.5 * 0.10) / (0.3 + 0.5)
        # C: (0.1*0.05 + 0.2*(-0.02)) / (0.1+0.2)
        expected_c = (0.1 * 0.05 + 0.2 * (-0.02)) / (0.1 + 0.2)

        assert pytest.approx(result["A"], rel=1e-6) == expected_a
        assert pytest.approx(result["B"], rel=1e-6) == expected_b
        assert pytest.approx(result["C"], rel=1e-6) == expected_c

    def test_self_exclusion(self, factor):
        """对角线权重应被置零，自身收益不参与计算。"""
        weights = pd.DataFrame(
            {"A": [0.9, 0.0], "B": [0.0, 0.8]},
            index=["A", "B"],
        )
        returns = pd.Series({"A": 0.50, "B": 0.01})

        result = factor.compute(weights=weights, returns=returns)

        assert pd.isna(result["A"])
        assert pd.isna(result["B"])

    def test_zero_weights(self, factor):
        """所有权重为 0 时，结果应为 NaN。"""
        weights = pd.DataFrame(
            {"A": [0.0, 0.0], "B": [0.0, 0.0]},
            index=["A", "B"],
        )
        returns = pd.Series({"A": 0.05, "B": 0.10})

        result = factor.compute(weights=weights, returns=returns)

        assert pd.isna(result["A"])
        assert pd.isna(result["B"])

    def test_nan_in_returns(self, factor):
        """上游收益含 NaN 时，加权结果应正确传播 NaN。"""
        weights = pd.DataFrame(
            {"A": [0.0, 0.3], "B": [0.4, 0.0]},
            index=["A", "B"],
        )
        returns = pd.Series({"A": 0.05, "B": np.nan})

        result = factor.compute(weights=weights, returns=returns)

        # A 的唯一上游 B 收益为 NaN -> 结果为 NaN
        assert pd.isna(result["A"])
        # B 的唯一上游 A 收益正常 -> 0.3*0.05 / 0.3 = 0.05
        assert pytest.approx(result["B"], rel=1e-6) == 0.05

    def test_output_type_and_name(self, factor):
        """输出应为 pd.Series，name 应为因子名。"""
        weights = pd.DataFrame(
            {"X": [0.0, 0.2], "Y": [0.3, 0.0]},
            index=["X", "Y"],
        )
        returns = pd.Series({"X": 0.01, "Y": 0.02})

        result = factor.compute(weights=weights, returns=returns)

        assert isinstance(result, pd.Series)
        assert result.name == "UPSTREAM_TRANSMISSION"
        assert list(result.index) == ["X", "Y"]

    def test_asymmetric_weights(self, factor):
        """非对称权重矩阵。"""
        weights = pd.DataFrame(
            {"A": [0.0, 0.0], "B": [0.6, 0.0]},
            index=["A", "B"],
        )
        returns = pd.Series({"A": 0.05, "B": -0.03})

        result = factor.compute(weights=weights, returns=returns)

        # A: wgt(A,B)=0.6 -> 0.6*(-0.03)/0.6 = -0.03
        assert pytest.approx(result["A"], rel=1e-6) == -0.03
        # B: 对 A 的权重为 0 -> NaN
        assert pd.isna(result["B"])
