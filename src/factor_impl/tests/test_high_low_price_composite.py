import numpy as np
import pandas as pd
import pytest

from factors.high_low_price_composite import HighLowPriceCompositeRatioFactor


@pytest.fixture
def factor():
    return HighLowPriceCompositeRatioFactor()


class TestHighLowPriceCompositeMetadata:
    def test_name(self, factor):
        assert factor.name == "HIGH_LOW_PRICE_COMPOSITE_RATIO"

    def test_category(self, factor):
        assert factor.category == "高频资金流"

    def test_repr(self, factor):
        assert "HIGH_LOW_PRICE_COMPOSITE_RATIO" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "HIGH_LOW_PRICE_COMPOSITE_RATIO"


class TestHighLowPriceCompositeCompute:
    def test_known_values(self, factor):
        """手算验证。"""
        dates = pd.date_range("2024-01-01", periods=3)
        non_extreme = pd.DataFrame({"A": [500.0, 600.0, 700.0]}, index=dates)
        high_price = pd.DataFrame({"A": [100.0, 200.0, 300.0]}, index=dates)
        total = pd.DataFrame({"A": [1000.0, 1000.0, 1000.0]}, index=dates)

        result = factor.compute(
            non_extreme_volume=non_extreme,
            high_price_volume=high_price,
            total_volume=total,
        )

        assert result.iloc[0, 0] == pytest.approx(0.4)
        assert result.iloc[1, 0] == pytest.approx(0.4)
        assert result.iloc[2, 0] == pytest.approx(0.4)

    def test_equal_volumes(self, factor):
        """非极端和高价成交量相等时，因子值为0。"""
        dates = pd.date_range("2024-01-01", periods=3)
        vol = pd.DataFrame({"A": [100.0, 200.0, 300.0]}, index=dates)
        total = pd.DataFrame({"A": [1000.0, 1000.0, 1000.0]}, index=dates)

        result = factor.compute(
            non_extreme_volume=vol,
            high_price_volume=vol,
            total_volume=total,
        )
        for i in range(3):
            assert result.iloc[i, 0] == pytest.approx(0.0)

    def test_multi_stock(self, factor):
        """多只股票独立计算。"""
        dates = pd.date_range("2024-01-01", periods=2)
        non_extreme = pd.DataFrame({"A": [800.0, 900.0], "B": [600.0, 700.0]}, index=dates)
        high_price = pd.DataFrame({"A": [200.0, 100.0], "B": [400.0, 300.0]}, index=dates)
        total = pd.DataFrame({"A": [1000.0, 1000.0], "B": [1000.0, 1000.0]}, index=dates)

        result = factor.compute(
            non_extreme_volume=non_extreme,
            high_price_volume=high_price,
            total_volume=total,
        )
        assert result.loc[dates[0], "A"] == pytest.approx(0.6)
        assert result.loc[dates[0], "B"] == pytest.approx(0.2)


class TestHighLowPriceCompositeEdgeCases:
    def test_zero_total_volume(self, factor):
        """总成交量为0时结果为 inf/NaN。"""
        dates = pd.date_range("2024-01-01", periods=2)
        non_extreme = pd.DataFrame({"A": [100.0, 200.0]}, index=dates)
        high_price = pd.DataFrame({"A": [50.0, 100.0]}, index=dates)
        total = pd.DataFrame({"A": [0.0, 0.0]}, index=dates)

        result = factor.compute(
            non_extreme_volume=non_extreme,
            high_price_volume=high_price,
            total_volume=total,
        )
        assert isinstance(result, pd.DataFrame)

    def test_nan_propagation(self, factor):
        dates = pd.date_range("2024-01-01", periods=3)
        non_extreme = pd.DataFrame({"A": [100.0, np.nan, 300.0]}, index=dates)
        high_price = pd.DataFrame({"A": [50.0, 100.0, 150.0]}, index=dates)
        total = pd.DataFrame({"A": [1000.0, 1000.0, 1000.0]}, index=dates)

        result = factor.compute(
            non_extreme_volume=non_extreme,
            high_price_volume=high_price,
            total_volume=total,
        )
        assert np.isnan(result.iloc[1, 0])


class TestHighLowPriceCompositeOutputShape:
    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=10)
        stocks = ["A", "B", "C"]
        non_extreme = pd.DataFrame(np.random.rand(10, 3) * 500, index=dates, columns=stocks)
        high_price = pd.DataFrame(np.random.rand(10, 3) * 200, index=dates, columns=stocks)
        total = pd.DataFrame(np.random.rand(10, 3) * 1000 + 500, index=dates, columns=stocks)

        result = factor.compute(
            non_extreme_volume=non_extreme,
            high_price_volume=high_price,
            total_volume=total,
        )
        assert result.shape == (10, 3)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=3)
        df = pd.DataFrame({"A": [100.0, 200.0, 300.0]}, index=dates)
        result = factor.compute(
            non_extreme_volume=df, high_price_volume=df, total_volume=df
        )
        assert isinstance(result, pd.DataFrame)
