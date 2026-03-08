import numpy as np
import pandas as pd
import pytest

from factors.chip_return_enhance import ChipReturnEnhanceFactor


@pytest.fixture
def factor():
    return ChipReturnEnhanceFactor()


class TestChipReturnEnhanceMetadata:
    def test_name(self, factor):
        assert factor.name == "CHIP_RETURN_ENHANCE"

    def test_category(self, factor):
        assert factor.category == "量价因子改进"

    def test_repr(self, factor):
        assert "CHIP_RETURN_ENHANCE" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "CHIP_RETURN_ENHANCE"
        assert meta["category"] == "量价因子改进"


class TestChipReturnEnhanceCompute:
    def test_known_values(self, factor):
        """手算验证: (1 - ret20) * ret20 + (1 - ret20) * holding_ret_adj"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        holding_ret_adj = pd.DataFrame({"A": [0.1, 0.2, -0.1]}, index=dates)
        ret20 = pd.DataFrame({"A": [0.05, -0.1, 0.2]}, index=dates)

        result = factor.compute(holding_ret_adj=holding_ret_adj, ret20=ret20)

        # row 0: (1-0.05)*0.05 + (1-0.05)*0.1 = 0.95*0.05 + 0.95*0.1 = 0.0475 + 0.095 = 0.1425
        assert result.iloc[0, 0] == pytest.approx(0.1425)
        # row 1: (1-(-0.1))*(-0.1) + (1-(-0.1))*0.2 = 1.1*(-0.1) + 1.1*0.2 = -0.11 + 0.22 = 0.11
        assert result.iloc[1, 0] == pytest.approx(0.11)
        # row 2: (1-0.2)*0.2 + (1-0.2)*(-0.1) = 0.8*0.2 + 0.8*(-0.1) = 0.16 - 0.08 = 0.08
        assert result.iloc[2, 0] == pytest.approx(0.08)

    def test_ret20_zero(self, factor):
        """ret20=0 时, 因子 = holding_ret_adj"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        holding_ret_adj = pd.DataFrame({"A": [0.1, 0.2, 0.3]}, index=dates)
        ret20 = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=dates)

        result = factor.compute(holding_ret_adj=holding_ret_adj, ret20=ret20)

        pd.testing.assert_frame_equal(result, holding_ret_adj)

    def test_holding_ret_adj_zero(self, factor):
        """holding_ret_adj=0 时, 因子 = (1-ret20)*ret20"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        holding_ret_adj = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=dates)
        ret20 = pd.DataFrame({"A": [0.1, 0.2, 0.3]}, index=dates)

        result = factor.compute(holding_ret_adj=holding_ret_adj, ret20=ret20)

        expected = (1 - ret20) * ret20
        pd.testing.assert_frame_equal(result, expected)

    def test_multi_stock(self, factor):
        """多只股票独立计算。"""
        dates = pd.bdate_range("2025-01-01", periods=2)
        holding_ret_adj = pd.DataFrame({"A": [0.1, 0.2], "B": [0.3, 0.4]}, index=dates)
        ret20 = pd.DataFrame({"A": [0.05, 0.1], "B": [0.0, -0.1]}, index=dates)

        result = factor.compute(holding_ret_adj=holding_ret_adj, ret20=ret20)

        assert result.shape == (2, 2)
        # A, row 0: 0.95*0.05 + 0.95*0.1 = 0.1425
        assert result.loc[dates[0], "A"] == pytest.approx(0.1425)
        # B, row 0: 1.0*0.0 + 1.0*0.3 = 0.3
        assert result.loc[dates[0], "B"] == pytest.approx(0.3)


class TestChipReturnEnhanceEdgeCases:
    def test_nan_propagation(self, factor):
        """输入含 NaN 时，输出对应位置也应为 NaN。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        holding_ret_adj = pd.DataFrame({"A": [0.1, np.nan, 0.3]}, index=dates)
        ret20 = pd.DataFrame({"A": [0.05, 0.1, np.nan]}, index=dates)

        result = factor.compute(holding_ret_adj=holding_ret_adj, ret20=ret20)

        assert not np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert np.isnan(result.iloc[2, 0])

    def test_all_zeros(self, factor):
        """全零输入，结果应全为零。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        holding_ret_adj = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=dates)
        ret20 = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=dates)

        result = factor.compute(holding_ret_adj=holding_ret_adj, ret20=ret20)

        expected = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=dates)
        pd.testing.assert_frame_equal(result, expected)


class TestChipReturnEnhanceOutputShape:
    def test_output_is_dataframe(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=5)
        holding_ret_adj = pd.DataFrame({"A": np.random.randn(5) * 0.1}, index=dates)
        ret20 = pd.DataFrame({"A": np.random.randn(5) * 0.1}, index=dates)

        result = factor.compute(holding_ret_adj=holding_ret_adj, ret20=ret20)
        assert isinstance(result, pd.DataFrame)

    def test_output_shape_matches_input(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=10)
        stocks = ["A", "B", "C"]
        holding_ret_adj = pd.DataFrame(
            np.random.randn(10, 3) * 0.1, index=dates, columns=stocks
        )
        ret20 = pd.DataFrame(
            np.random.randn(10, 3) * 0.1, index=dates, columns=stocks
        )

        result = factor.compute(holding_ret_adj=holding_ret_adj, ret20=ret20)
        assert result.shape == holding_ret_adj.shape
        assert list(result.columns) == stocks
