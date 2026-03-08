import numpy as np
import pandas as pd
import pytest

from factors.cancel_rate import CancelRateFactor


@pytest.fixture
def factor():
    return CancelRateFactor()


class TestCancelRateMetadata:
    def test_name(self, factor):
        assert factor.name == "CANCEL_RATE"

    def test_category(self, factor):
        assert factor.category == "高频流动性"

    def test_repr(self, factor):
        assert "CANCEL_RATE" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "CANCEL_RATE"
        assert meta["category"] == "高频流动性"
        assert "撤单" in meta["description"]


class TestCancelRateHandCalculated:
    def test_T3_single_stock(self, factor):
        """T=3, 单只股票, 手动计算验证。

        full    = [0.03, 0.06, 0.09]
        partial = [0.01, 0.02, 0.03]
        invalid = [0.02, 0.04, 0.06]
        composite = [(0.03+0.01+0.02)/3, (0.06+0.02+0.04)/3, (0.09+0.03+0.06)/3]
                  = [0.02, 0.04, 0.06]
        rolling mean(T=3) at t=2: (0.02 + 0.04 + 0.06) / 3 = 0.04
        """
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        full = pd.DataFrame([0.03, 0.06, 0.09], index=dates, columns=stocks)
        partial = pd.DataFrame([0.01, 0.02, 0.03], index=dates, columns=stocks)
        invalid = pd.DataFrame([0.02, 0.04, 0.06], index=dates, columns=stocks)

        result = factor.compute(
            full_cancel_rate=full,
            partial_cancel_rate=partial,
            invalid_cancel_rate=invalid,
            T=3,
        )

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(0.04, rel=1e-10)

    def test_equal_components(self, factor):
        """三个分量相等时，合成值等于单个分量。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        rate = pd.DataFrame([0.1, 0.2, 0.3], index=dates, columns=stocks)

        result = factor.compute(
            full_cancel_rate=rate,
            partial_cancel_rate=rate.copy(),
            invalid_cancel_rate=rate.copy(),
            T=3,
        )

        # composite = rate, rolling mean = (0.1+0.2+0.3)/3 = 0.2
        assert result.iloc[2, 0] == pytest.approx(0.2, rel=1e-10)

    def test_constant_input(self, factor):
        """常数输入时，滚动均值等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]

        rate = pd.DataFrame([0.05] * 5, index=dates, columns=stocks)

        result = factor.compute(
            full_cancel_rate=rate,
            partial_cancel_rate=rate.copy(),
            invalid_cancel_rate=rate.copy(),
            T=3,
        )

        assert result.iloc[2, 0] == pytest.approx(0.05, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(0.05, rel=1e-10)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A", "B"]

        full = pd.DataFrame({"A": [0.03, 0.03, 0.03], "B": [0.09, 0.09, 0.09]}, index=dates)
        partial = pd.DataFrame({"A": [0.03, 0.03, 0.03], "B": [0.09, 0.09, 0.09]}, index=dates)
        invalid = pd.DataFrame({"A": [0.03, 0.03, 0.03], "B": [0.09, 0.09, 0.09]}, index=dates)

        result = factor.compute(
            full_cancel_rate=full,
            partial_cancel_rate=partial,
            invalid_cancel_rate=invalid,
            T=3,
        )

        assert result.iloc[2, 0] == pytest.approx(0.03, rel=1e-10)
        assert result.iloc[2, 1] == pytest.approx(0.09, rel=1e-10)


class TestCancelRateEdgeCases:
    def test_short_data(self, factor):
        """数据不足 T 时，结果应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        rate = pd.DataFrame([0.01] * 5, index=dates, columns=stocks)

        result = factor.compute(
            full_cancel_rate=rate,
            partial_cancel_rate=rate,
            invalid_cancel_rate=rate,
            T=20,
        )
        assert result.isna().all().all()

    def test_nan_in_input(self, factor):
        """输入含 NaN 时不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        full = pd.DataFrame([0.01, np.nan, 0.03, 0.04, 0.05], index=dates, columns=stocks)
        partial = pd.DataFrame([0.01] * 5, index=dates, columns=stocks)
        invalid = pd.DataFrame([0.01] * 5, index=dates, columns=stocks)

        result = factor.compute(
            full_cancel_rate=full,
            partial_cancel_rate=partial,
            invalid_cancel_rate=invalid,
            T=3,
        )
        assert isinstance(result, pd.DataFrame)
        # 窗口含 NaN 的行结果也为 NaN
        assert np.isnan(result.iloc[2, 0])

    def test_all_zero(self, factor):
        """全零输入时，结果应全为 0。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        rate = pd.DataFrame([0.0] * 5, index=dates, columns=stocks)

        result = factor.compute(
            full_cancel_rate=rate,
            partial_cancel_rate=rate,
            invalid_cancel_rate=rate,
            T=3,
        )
        assert result.iloc[2, 0] == pytest.approx(0.0, abs=1e-15)


class TestCancelRateOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        rate = pd.DataFrame(
            np.random.uniform(0, 0.1, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(
            full_cancel_rate=rate,
            partial_cancel_rate=rate,
            invalid_cancel_rate=rate,
            T=20,
        )
        assert result.shape == rate.shape
        assert list(result.columns) == list(rate.columns)
        assert list(result.index) == list(rate.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        rate = pd.DataFrame([0.01] * 5, index=dates, columns=stocks)

        result = factor.compute(
            full_cancel_rate=rate,
            partial_cancel_rate=rate,
            invalid_cancel_rate=rate,
            T=3,
        )
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        """前 T-1 行应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A", "B"]
        T = 20
        rate = pd.DataFrame(
            np.random.uniform(0, 0.1, (25, 2)), index=dates, columns=stocks
        )

        result = factor.compute(
            full_cancel_rate=rate,
            partial_cancel_rate=rate,
            invalid_cancel_rate=rate,
            T=T,
        )
        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
