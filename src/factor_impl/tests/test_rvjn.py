import numpy as np
import pandas as pd
import pytest
from math import gamma

from factors.rvjn import RVJNFactor, MU_2_3


@pytest.fixture
def factor():
    return RVJNFactor()


class TestRVJNMetadata:
    def test_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "RVJN"
        assert meta["category"] == "波动率"
        assert "下行跳跃波动率" in meta["description"]


class TestRVJNCompute:
    """测试 compute 方法（预计算输入）。"""

    def test_basic(self, factor):
        """RS^- > 0.5 * IV_hat 时，RVJN = RS^- - 0.5 * IV_hat。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        rs_neg = pd.DataFrame({"A": [0.010, 0.008, 0.012]}, index=dates)
        iv_hat = pd.DataFrame({"A": [0.004, 0.006, 0.010]}, index=dates)

        result = factor.compute(rs_negative=rs_neg, iv_hat=iv_hat)

        expected = pd.DataFrame(
            {"A": [0.010 - 0.002, 0.008 - 0.003, 0.012 - 0.005]}, index=dates
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_clipping_to_zero(self, factor):
        """RS^- < 0.5 * IV_hat 时，RVJN 应被截断为 0。"""
        dates = pd.bdate_range("2025-01-01", periods=2)
        rs_neg = pd.DataFrame({"A": [0.001, 0.002]}, index=dates)
        iv_hat = pd.DataFrame({"A": [0.010, 0.020]}, index=dates)

        result = factor.compute(rs_negative=rs_neg, iv_hat=iv_hat)

        assert (result.values >= 0).all()
        np.testing.assert_array_almost_equal(result.values, [[0.0], [0.0]])

    def test_mixed_clipping(self, factor):
        """部分行需要截断、部分行不需要。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        rs_neg = pd.DataFrame({"A": [0.010, 0.001, 0.006]}, index=dates)
        iv_hat = pd.DataFrame({"A": [0.004, 0.010, 0.012]}, index=dates)

        result = factor.compute(rs_negative=rs_neg, iv_hat=iv_hat)

        expected_vals = [0.010 - 0.002, 0.0, 0.0]
        np.testing.assert_array_almost_equal(result["A"].values, expected_vals)

    def test_multi_stock(self, factor):
        """多只股票同时计算。"""
        dates = pd.bdate_range("2025-01-01", periods=2)
        rs_neg = pd.DataFrame({"A": [0.01, 0.02], "B": [0.005, 0.001]}, index=dates)
        iv_hat = pd.DataFrame({"A": [0.01, 0.01], "B": [0.002, 0.010]}, index=dates)

        result = factor.compute(rs_negative=rs_neg, iv_hat=iv_hat)

        assert result.shape == (2, 2)
        # A: [0.01-0.005=0.005, 0.02-0.005=0.015]
        np.testing.assert_almost_equal(result.loc[dates[0], "A"], 0.005)
        np.testing.assert_almost_equal(result.loc[dates[1], "A"], 0.015)
        # B: [0.005-0.001=0.004, max(0.001-0.005,0)=0]
        np.testing.assert_almost_equal(result.loc[dates[0], "B"], 0.004)
        np.testing.assert_almost_equal(result.loc[dates[1], "B"], 0.0)

    def test_nan_propagation(self, factor):
        """输入含 NaN 时，输出对应位置也应为 NaN（clip 不影响 NaN）。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        rs_neg = pd.DataFrame({"A": [0.01, np.nan, 0.005]}, index=dates)
        iv_hat = pd.DataFrame({"A": [0.004, 0.006, np.nan]}, index=dates)

        result = factor.compute(rs_negative=rs_neg, iv_hat=iv_hat)

        assert not np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert np.isnan(result.iloc[2, 0])


class TestComputeFromIntraday:
    """测试 compute_from_intraday 静态方法。"""

    def test_rs_negative_all_positive(self):
        """所有日内收益为正时，RS^- 应为 0。"""
        dates = pd.bdate_range("2025-01-01", periods=2)
        intraday = pd.DataFrame(
            np.abs(np.random.randn(2, 10)) * 0.001 + 0.0001,
            index=dates,
            columns=range(10),
        )
        result = RVJNFactor.compute_from_intraday(intraday)
        np.testing.assert_array_almost_equal(result["rs_negative"].values, [0.0, 0.0])

    def test_rs_negative_known_values(self):
        """用已知数据验证 RS^- 计算。"""
        dates = pd.bdate_range("2025-01-01", periods=1)
        # 5 个日内收益：-0.01, 0.02, -0.03, 0.01, -0.02
        returns = [[-0.01, 0.02, -0.03, 0.01, -0.02]]
        intraday = pd.DataFrame(returns, index=dates, columns=range(5))

        result = RVJNFactor.compute_from_intraday(intraday)

        # RS^- = (-0.01)^2 + (-0.03)^2 + (-0.02)^2 = 0.0001 + 0.0009 + 0.0004 = 0.0014
        np.testing.assert_almost_equal(result["rs_negative"].iloc[0], 0.0014)

    def test_iv_hat_known_values(self):
        """用已知数据验证 IV_hat 计算（k=3）。"""
        dates = pd.bdate_range("2025-01-01", periods=1)
        # 使用简单数值便于手算
        r = [0.01, -0.02, 0.03, -0.01, 0.02]
        intraday = pd.DataFrame([r], index=dates, columns=range(5))

        result = RVJNFactor.compute_from_intraday(intraday, k=3)

        # 手动计算 IV_hat:
        abs_r_23 = np.abs(r) ** (2 / 3)
        # k=3 的滑动乘积，从 index 2 开始
        products = []
        for i in range(2, 5):
            products.append(abs_r_23[i] * abs_r_23[i - 1] * abs_r_23[i - 2])
        expected_iv = MU_2_3 ** (-3) * sum(products)

        np.testing.assert_almost_equal(result["iv_hat"].iloc[0], expected_iv)

    def test_iv_hat_fewer_cols_than_k(self):
        """列数少于 k 时，IV_hat 应为 NaN。"""
        dates = pd.bdate_range("2025-01-01", periods=1)
        intraday = pd.DataFrame([[0.01, -0.02]], index=dates, columns=range(2))

        result = RVJNFactor.compute_from_intraday(intraday, k=3)

        assert np.isnan(result["iv_hat"].iloc[0])

    def test_roundtrip_single_stock(self):
        """从日内收益 -> compute_from_intraday -> compute 的完整流程。"""
        np.random.seed(42)
        dates = pd.bdate_range("2025-01-01", periods=5)
        intraday = pd.DataFrame(
            np.random.randn(5, 48) * 0.002,
            index=dates,
            columns=range(48),
        )

        parts = RVJNFactor.compute_from_intraday(intraday)

        rs_neg_df = parts["rs_negative"].to_frame("STOCK")
        iv_hat_df = parts["iv_hat"].to_frame("STOCK")

        factor = RVJNFactor()
        rvjn = factor.compute(rs_negative=rs_neg_df, iv_hat=iv_hat_df)

        assert rvjn.shape == (5, 1)
        assert (rvjn.values >= 0).all()

    def test_mu_2_3_constant(self):
        """验证 MU_2_3 常量的数值正确性。"""
        expected = 2 ** (1 / 3) * gamma(5 / 6) / gamma(1 / 2)
        np.testing.assert_almost_equal(MU_2_3, expected)
