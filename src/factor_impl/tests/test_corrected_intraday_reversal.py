import numpy as np
import pandas as pd
import pytest

from factors.corrected_intraday_reversal import CorrectedIntradayReversalFactor


@pytest.fixture
def factor():
    return CorrectedIntradayReversalFactor()


class TestCorrectedIntradayReversalMetadata:
    def test_name(self, factor):
        assert factor.name == "CORRECTED_INTRADAY_REVERSAL"

    def test_category(self, factor):
        assert factor.category == "高频动量反转"

    def test_repr(self, factor):
        assert "CORRECTED_INTRADAY_REVERSAL" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "CORRECTED_INTRADAY_REVERSAL"
        assert meta["category"] == "高频动量反转"
        assert "反转" in meta["description"]


class TestCorrectedIntradayReversalCompute:
    """测试 compute 方法。"""

    def test_low_vol_flips_sign(self, factor):
        """低波动股票的日内收益应被翻转符号。

        3 只股票, 1 天:
          intraday_return = [0.001, 0.01, 0.005]
          abs = [0.001, 0.01, 0.005]
          cross_section_mean = (0.001 + 0.01 + 0.005) / 3 = 0.00533...

        A (0.001 < 0.00533): 翻转 -> -0.001
        B (0.01 > 0.00533): 保持 -> 0.01
        C (0.005 < 0.00533): 翻转 -> -0.005
        """
        dates = pd.bdate_range("2025-01-01", periods=1)
        data = pd.DataFrame(
            {"A": [0.001], "B": [0.01], "C": [0.005]}, index=dates
        )

        # T=1 以便直接看到修正后的值
        result = factor.compute(intraday_return=data, T=1)

        assert result.iloc[0, 0] == pytest.approx(-0.001)  # A flipped
        assert result.iloc[0, 1] == pytest.approx(0.01)    # B kept
        assert result.iloc[0, 2] == pytest.approx(-0.005)  # C flipped

    def test_negative_return_low_vol(self, factor):
        """负收益 + 低波动 -> 翻转为正。"""
        dates = pd.bdate_range("2025-01-01", periods=1)
        data = pd.DataFrame(
            {"A": [-0.001], "B": [0.01], "C": [-0.005]}, index=dates
        )

        result = factor.compute(intraday_return=data, T=1)

        # abs: [0.001, 0.01, 0.005], mean=0.00533
        # A: 0.001 < 0.00533 -> flip: 0.001
        # B: 0.01 > 0.00533 -> keep: 0.01
        # C: 0.005 < 0.00533 -> flip: 0.005
        assert result.iloc[0, 0] == pytest.approx(0.001)
        assert result.iloc[0, 1] == pytest.approx(0.01)
        assert result.iloc[0, 2] == pytest.approx(0.005)

    def test_rolling_mean(self, factor):
        """验证滚动均值。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        # 单只股票，无截面比较 -> abs 始终等于 mean -> 不翻转
        # 用两只股票
        data = pd.DataFrame(
            {"A": [0.001, 0.002, 0.001, 0.002, 0.001],
             "B": [0.01, 0.01, 0.01, 0.01, 0.01]},
            index=dates,
        )

        result = factor.compute(intraday_return=data, T=3)

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        # A 的 abs 始终 < cross_section_mean, 所以被翻转
        # corrected A = [-0.001, -0.002, -0.001, -0.002, -0.001]
        # rolling mean T=3 for A: mean(-0.001, -0.002, -0.001) = -0.001333...
        expected_a = np.mean([-0.001, -0.002, -0.001])
        assert result.iloc[2, 0] == pytest.approx(expected_a)

    def test_all_same_vol(self, factor):
        """所有股票波动率相同时，无翻转（都等于均值）。"""
        dates = pd.bdate_range("2025-01-01", periods=1)
        data = pd.DataFrame(
            {"A": [0.01], "B": [-0.01], "C": [0.01]}, index=dates
        )

        result = factor.compute(intraday_return=data, T=1)

        # abs = [0.01, 0.01, 0.01], mean = 0.01
        # 没有 < mean 的，所以都不翻转
        assert result.iloc[0, 0] == pytest.approx(0.01)
        assert result.iloc[0, 1] == pytest.approx(-0.01)
        assert result.iloc[0, 2] == pytest.approx(0.01)


class TestCorrectedIntradayReversalEdgeCases:
    def test_nan_in_input(self, factor):
        """输入含 NaN 时不应抛异常。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        data = pd.DataFrame(
            {"A": [0.01, np.nan, 0.01, 0.01, 0.01],
             "B": [0.02, 0.02, 0.02, 0.02, 0.02]},
            index=dates,
        )

        result = factor.compute(intraday_return=data, T=3)

        assert isinstance(result, pd.DataFrame)

    def test_single_stock(self, factor):
        """单只股票时，abs 始终等于 mean，不翻转。"""
        dates = pd.bdate_range("2025-01-01", periods=4)
        data = pd.DataFrame({"A": [0.01, -0.02, 0.03, -0.01]}, index=dates)

        result = factor.compute(intraday_return=data, T=3)

        # 单只股票: abs == mean, 不满足 < mean, 不翻转
        expected = np.mean([0.01, -0.02, 0.03])
        assert result.iloc[2, 0] == pytest.approx(expected)

    def test_zero_return(self, factor):
        """零收益的处理。"""
        dates = pd.bdate_range("2025-01-01", periods=1)
        data = pd.DataFrame(
            {"A": [0.0], "B": [0.01]}, index=dates
        )

        result = factor.compute(intraday_return=data, T=1)

        # A: abs=0 < mean=0.005, flip -> -0 = 0
        assert result.iloc[0, 0] == pytest.approx(0.0)


class TestCorrectedIntradayReversalOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=30)
        stocks = ["A", "B", "C"]
        data = pd.DataFrame(
            np.random.randn(30, 3) * 0.01, index=dates, columns=stocks
        )

        result = factor.compute(intraday_return=data, T=20)

        assert result.shape == data.shape
        assert list(result.columns) == list(data.columns)

    def test_output_is_dataframe(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=5)
        data = pd.DataFrame(
            {"A": [0.01, -0.01, 0.02, -0.02, 0.01],
             "B": [0.02, 0.01, -0.01, 0.03, -0.02]},
            index=dates,
        )

        result = factor.compute(intraday_return=data, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        """前 T-1 行应全为 NaN。"""
        dates = pd.bdate_range("2025-01-01", periods=25)
        data = pd.DataFrame(
            np.random.randn(25, 3) * 0.01, index=dates, columns=["A", "B", "C"]
        )
        T = 20

        result = factor.compute(intraday_return=data, T=T)

        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
