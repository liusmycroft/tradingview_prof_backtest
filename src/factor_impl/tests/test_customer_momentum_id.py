import numpy as np
import pandas as pd
import pytest

from factors.customer_momentum_id import CustomerMomentumIDFactor


@pytest.fixture
def factor():
    return CustomerMomentumIDFactor()


class TestCustomerMomentumIDMetadata:
    def test_name(self, factor):
        assert factor.name == "CUSTOMER_MOMENTUM_ID"

    def test_category(self, factor):
        assert factor.category == "图谱网络-动量溢出"

    def test_repr(self, factor):
        assert "CUSTOMER_MOMENTUM_ID" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "CUSTOMER_MOMENTUM_ID"
        assert meta["category"] == "图谱网络-动量溢出"


class TestCustomerMomentumIDHandCalculated:
    """用手算数据验证计算的正确性。"""

    def test_constant_input(self, factor):
        """常数输入时, combined = mom * (-ID), EMA 应等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily_customer_momentum = pd.DataFrame(2.0, index=dates, columns=stocks)
        daily_info_discreteness = pd.DataFrame(-3.0, index=dates, columns=stocks)
        # combined = 2.0 * (--3.0) = 2.0 * 3.0 = 6.0

        result = factor.compute(
            daily_customer_momentum=daily_customer_momentum,
            daily_info_discreteness=daily_info_discreteness,
            T=20,
        )

        valid = result.dropna()
        np.testing.assert_array_almost_equal(valid["A"].values, 6.0)

    def test_negative_id_amplifies_momentum(self, factor):
        """负 ID（信息连续）应放大正动量。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily_customer_momentum = pd.DataFrame(1.0, index=dates, columns=stocks)
        daily_info_discreteness = pd.DataFrame(-0.5, index=dates, columns=stocks)
        # combined = 1.0 * 0.5 = 0.5

        result = factor.compute(
            daily_customer_momentum=daily_customer_momentum,
            daily_info_discreteness=daily_info_discreteness,
            T=20,
        )

        valid = result.dropna()
        np.testing.assert_array_almost_equal(valid["A"].values, 0.5)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A", "B"]
        daily_customer_momentum = pd.DataFrame(
            {"A": [1.0] * 25, "B": [2.0] * 25}, index=dates
        )
        daily_info_discreteness = pd.DataFrame(
            {"A": [-1.0] * 25, "B": [-2.0] * 25}, index=dates
        )
        # A: 1.0 * 1.0 = 1.0, B: 2.0 * 2.0 = 4.0

        result = factor.compute(
            daily_customer_momentum=daily_customer_momentum,
            daily_info_discreteness=daily_info_discreteness,
            T=20,
        )

        valid = result.dropna()
        np.testing.assert_array_almost_equal(valid["A"].values, 1.0)
        np.testing.assert_array_almost_equal(valid["B"].values, 4.0)

    def test_ema_weights_recent_more(self, factor):
        """EMA 应对近期数据赋予更高权重。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        mom_vals = [0.0] * 20 + [10.0] * 5
        daily_customer_momentum = pd.DataFrame(mom_vals, index=dates, columns=stocks)
        daily_info_discreteness = pd.DataFrame(-1.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_customer_momentum=daily_customer_momentum,
            daily_info_discreteness=daily_info_discreteness,
            T=20,
        )

        last_val = result.iloc[-1, 0]
        assert 0 < last_val < 10


class TestCustomerMomentumIDEdgeCases:
    def test_nan_in_input(self, factor):
        """输入含 NaN 时不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        mom = np.ones(25) * 1.0
        mom[10] = np.nan
        daily_customer_momentum = pd.DataFrame(mom, index=dates, columns=stocks)
        daily_info_discreteness = pd.DataFrame(-1.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_customer_momentum=daily_customer_momentum,
            daily_info_discreteness=daily_info_discreteness,
            T=20,
        )
        assert result.shape == (25, 1)

    def test_all_nan(self, factor):
        """全 NaN 输入时, 结果应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily_customer_momentum = pd.DataFrame(np.nan, index=dates, columns=stocks)
        daily_info_discreteness = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(
            daily_customer_momentum=daily_customer_momentum,
            daily_info_discreteness=daily_info_discreteness,
            T=20,
        )
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        """全零输入时, 结果应全为 0。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily_customer_momentum = pd.DataFrame(0.0, index=dates, columns=stocks)
        daily_info_discreteness = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(
            daily_customer_momentum=daily_customer_momentum,
            daily_info_discreteness=daily_info_discreteness,
            T=20,
        )
        valid = result.dropna()
        for val in valid["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)


class TestCustomerMomentumIDOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily_customer_momentum = pd.DataFrame(
            np.random.uniform(-1, 1, (30, 3)), index=dates, columns=stocks
        )
        daily_info_discreteness = pd.DataFrame(
            np.random.uniform(-1, 1, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(
            daily_customer_momentum=daily_customer_momentum,
            daily_info_discreteness=daily_info_discreteness,
            T=20,
        )

        assert result.shape == daily_customer_momentum.shape
        assert list(result.columns) == list(daily_customer_momentum.columns)
        assert list(result.index) == list(daily_customer_momentum.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        daily_customer_momentum = pd.DataFrame([1.0] * 25, index=dates, columns=stocks)
        daily_info_discreteness = pd.DataFrame([-1.0] * 25, index=dates, columns=stocks)

        result = factor.compute(
            daily_customer_momentum=daily_customer_momentum,
            daily_info_discreteness=daily_info_discreteness,
            T=20,
        )
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        """min_periods=T, 前 T-1 行应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        T = 20
        daily_customer_momentum = pd.DataFrame(
            np.random.uniform(-1, 1, (30, 2)), index=dates, columns=stocks
        )
        daily_info_discreteness = pd.DataFrame(
            np.random.uniform(-1, 1, (30, 2)), index=dates, columns=stocks
        )

        result = factor.compute(
            daily_customer_momentum=daily_customer_momentum,
            daily_info_discreteness=daily_info_discreteness,
            T=T,
        )

        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
