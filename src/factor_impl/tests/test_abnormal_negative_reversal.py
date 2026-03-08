import numpy as np
import pandas as pd
import pytest

from factors.abnormal_negative_reversal import AbnormalNegativeReversalFactor


@pytest.fixture
def factor():
    return AbnormalNegativeReversalFactor()


class TestABNRMetadata:
    def test_name(self, factor):
        assert factor.name == "AB_NR"

    def test_category(self, factor):
        assert factor.category == "高频收益分布"

    def test_repr(self, factor):
        assert "AB_NR" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "AB_NR"
        assert meta["category"] == "高频收益分布"


class TestABNRHandCalculated:
    def test_constant_nr(self, factor):
        """NR 恒定时, AB_NR = NR / mean(NR) = 1.0。"""
        dates = pd.date_range("2024-01-01", periods=15, freq="ME")
        stocks = ["A"]
        monthly_nr = pd.DataFrame(0.3, index=dates, columns=stocks)

        result = factor.compute(monthly_nr=monthly_nr, K=12)

        # 前 11 行为 NaN (min_periods=12), 第 12 行起为 1.0
        valid = result.dropna()
        np.testing.assert_array_almost_equal(valid["A"].values, 1.0)

    def test_increasing_nr(self, factor):
        """NR 递增时, 当月 NR > 均值, AB_NR > 1。"""
        dates = pd.date_range("2024-01-01", periods=13, freq="ME")
        stocks = ["A"]
        vals = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5]
        monthly_nr = pd.DataFrame(vals, index=dates, columns=stocks)

        result = factor.compute(monthly_nr=monthly_nr, K=12)

        # At index 12: NR=0.5, mean of indices 1..12 = (0.1*11 + 0.5)/12
        nr_mean = (0.1 * 11 + 0.5) / 12
        expected = 0.5 / nr_mean
        assert result.iloc[12, 0] == pytest.approx(expected, rel=1e-10)
        assert result.iloc[12, 0] > 1.0

    def test_manual_K3(self, factor):
        """K=3 手算验证。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="ME")
        stocks = ["A"]
        vals = [0.2, 0.4, 0.6, 0.3, 0.5]
        monthly_nr = pd.DataFrame(vals, index=dates, columns=stocks)

        result = factor.compute(monthly_nr=monthly_nr, K=3)

        # index 2: mean(0.2,0.4,0.6) = 0.4, AB_NR = 0.6/0.4 = 1.5
        assert result.iloc[2, 0] == pytest.approx(0.6 / 0.4, rel=1e-10)
        # index 3: mean(0.4,0.6,0.3) = 1.3/3, AB_NR = 0.3/(1.3/3)
        assert result.iloc[3, 0] == pytest.approx(0.3 / (1.3 / 3), rel=1e-10)
        # index 4: mean(0.6,0.3,0.5) = 1.4/3, AB_NR = 0.5/(1.4/3)
        assert result.iloc[4, 0] == pytest.approx(0.5 / (1.4 / 3), rel=1e-10)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="ME")
        monthly_nr = pd.DataFrame(
            {"A": [0.3, 0.3, 0.3, 0.3, 0.3],
             "B": [0.1, 0.2, 0.3, 0.4, 0.5]},
            index=dates,
        )

        result = factor.compute(monthly_nr=monthly_nr, K=3)

        # A is constant => AB_NR = 1.0
        assert result.iloc[2, 0] == pytest.approx(1.0, rel=1e-10)
        # B at index 2: mean(0.1,0.2,0.3)=0.2, AB_NR=0.3/0.2=1.5
        assert result.iloc[2, 1] == pytest.approx(1.5, rel=1e-10)


class TestABNREdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="ME")
        stocks = ["A"]
        vals = [0.3, np.nan, 0.3, 0.3, 0.3]
        monthly_nr = pd.DataFrame(vals, index=dates, columns=stocks)

        result = factor.compute(monthly_nr=monthly_nr, K=3)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (5, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="ME")
        stocks = ["A"]
        monthly_nr = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(monthly_nr=monthly_nr, K=3)
        assert result.isna().all().all()

    def test_zero_nr_gives_nan_or_inf(self, factor):
        """NR 均值为 0 时, 除法结果为 inf 或 NaN。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="ME")
        stocks = ["A"]
        monthly_nr = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(monthly_nr=monthly_nr, K=3)
        # 0/0 = NaN
        assert result.iloc[2:].isna().all().all() or np.isinf(result.iloc[2:].values).all()

    def test_first_K_minus_1_nan(self, factor):
        """前 K-1 行应为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="ME")
        stocks = ["A"]
        monthly_nr = pd.DataFrame(0.5, index=dates, columns=stocks)
        K = 5

        result = factor.compute(monthly_nr=monthly_nr, K=K)
        assert result.iloc[: K - 1].isna().all().all()
        assert result.iloc[K - 1 :].notna().all().all()


class TestABNROutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=15, freq="ME")
        stocks = ["A", "B", "C"]
        monthly_nr = pd.DataFrame(
            np.random.uniform(0.1, 0.5, (15, 3)), index=dates, columns=stocks
        )

        result = factor.compute(monthly_nr=monthly_nr, K=12)
        assert result.shape == monthly_nr.shape
        assert list(result.columns) == list(monthly_nr.columns)
        assert list(result.index) == list(monthly_nr.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="ME")
        stocks = ["A"]
        monthly_nr = pd.DataFrame(0.3, index=dates, columns=stocks)

        result = factor.compute(monthly_nr=monthly_nr, K=3)
        assert isinstance(result, pd.DataFrame)
