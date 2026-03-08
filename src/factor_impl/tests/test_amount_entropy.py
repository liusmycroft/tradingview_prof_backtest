import numpy as np
import pandas as pd
import pytest

from factors.amount_entropy import AmountEntropyFactor


@pytest.fixture
def factor():
    return AmountEntropyFactor()


class TestAmountEntropyMetadata:
    def test_name(self, factor):
        assert factor.name == "AMOUNT_ENTROPY"

    def test_category(self, factor):
        assert factor.category == "高频量价"

    def test_repr(self, factor):
        assert "AMOUNT_ENTROPY" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "AMOUNT_ENTROPY"
        assert meta["category"] == "高频量价"


class TestAmountEntropyHandCalculated:
    """用手算数据验证滚动均值计算的正确性。"""

    def test_T3_single_stock(self, factor):
        """T=3, 单只股票, 手动验证滚动均值。

        daily_entropy = [1.0, 2.0, 3.0, 4.0, 5.0]
        T=3, min_periods=1:
          row 0: mean([1.0]) = 1.0
          row 1: mean([1.0, 2.0]) = 1.5
          row 2: mean([1.0, 2.0, 3.0]) = 2.0
          row 3: mean([2.0, 3.0, 4.0]) = 3.0
          row 4: mean([3.0, 4.0, 5.0]) = 4.0
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_entropy = pd.DataFrame(
            [1.0, 2.0, 3.0, 4.0, 5.0], index=dates, columns=stocks
        )

        result = factor.compute(daily_entropy=daily_entropy, T=3)

        expected = [1.0, 1.5, 2.0, 3.0, 4.0]
        np.testing.assert_array_almost_equal(result["A"].values, expected)

    def test_T2_two_stocks(self, factor):
        """T=2, 两只股票, 验证多列并行计算。"""
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        stocks = ["A", "B"]
        daily_entropy = pd.DataFrame(
            [[1.0, 4.0], [3.0, 2.0], [5.0, 6.0], [7.0, 8.0]],
            index=dates,
            columns=stocks,
        )

        result = factor.compute(daily_entropy=daily_entropy, T=2)

        # Stock A: row1 = mean(1,3)=2.0, row2 = mean(3,5)=4.0, row3 = mean(5,7)=6.0
        assert result.loc[dates[1], "A"] == pytest.approx(2.0, rel=1e-10)
        assert result.loc[dates[2], "A"] == pytest.approx(4.0, rel=1e-10)
        assert result.loc[dates[3], "A"] == pytest.approx(6.0, rel=1e-10)

        # Stock B: row1 = mean(4,2)=3.0, row2 = mean(2,6)=4.0, row3 = mean(6,8)=7.0
        assert result.loc[dates[1], "B"] == pytest.approx(3.0, rel=1e-10)
        assert result.loc[dates[2], "B"] == pytest.approx(4.0, rel=1e-10)
        assert result.loc[dates[3], "B"] == pytest.approx(7.0, rel=1e-10)

    def test_constant_entropy(self, factor):
        """恒定熵值时, 滚动均值应等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["X"]
        daily_entropy = pd.DataFrame(2.5, index=dates, columns=stocks)

        result = factor.compute(daily_entropy=daily_entropy, T=5)

        for i in range(10):
            assert result.iloc[i, 0] == pytest.approx(2.5, rel=1e-10)


class TestAmountEntropyEdgeCases:
    def test_nan_in_entropy(self, factor):
        """daily_entropy 中含 NaN 时, 不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_entropy = pd.DataFrame(
            [1.0, np.nan, 3.0, 4.0, 5.0], index=dates, columns=stocks
        )

        result = factor.compute(daily_entropy=daily_entropy, T=3)
        assert isinstance(result, pd.DataFrame)
        # row 2: mean of [1.0, NaN, 3.0] with min_periods=1 -> mean of valid = 2.0
        assert result.iloc[2, 0] == pytest.approx(2.0, rel=1e-10)

    def test_all_nan(self, factor):
        """全 NaN 输入时, 结果应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_entropy = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_entropy=daily_entropy, T=3)
        assert result.isna().all().all()

    def test_zero_entropy(self, factor):
        """全零输入时, 结果应全为 0。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_entropy = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_entropy=daily_entropy, T=3)
        for i in range(5):
            assert result.iloc[i, 0] == pytest.approx(0.0, abs=1e-15)


class TestAmountEntropyOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily_entropy = pd.DataFrame(
            np.random.uniform(1, 3, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_entropy=daily_entropy, T=10)

        assert result.shape == daily_entropy.shape
        assert list(result.columns) == list(daily_entropy.columns)
        assert list(result.index) == list(daily_entropy.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily_entropy = pd.DataFrame(
            [1.0, 2.0, 3.0, 4.0, 5.0], index=dates, columns=stocks
        )

        result = factor.compute(daily_entropy=daily_entropy, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_min_periods_1_no_leading_nan(self, factor):
        """min_periods=1, 所以第一行就有值, 不存在前导 NaN。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        daily_entropy = pd.DataFrame(
            np.random.uniform(1, 3, (10, 2)), index=dates, columns=stocks
        )

        result = factor.compute(daily_entropy=daily_entropy, T=5)
        assert result.iloc[0].notna().all()
        assert result.notna().all().all()
