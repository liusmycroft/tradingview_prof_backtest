"""筹码集中度因子测试"""

import numpy as np
import pandas as pd
import pytest
from factors.chip_concentration import ChipConcentrationFactor


@pytest.fixture
def factor():
    return ChipConcentrationFactor()


@pytest.fixture
def sample_data():
    dates = pd.date_range("2024-01-01", periods=5)
    stocks = ["000001", "000002", "000003"]
    price95 = pd.DataFrame(
        [[15.0, 20.0, 25.0]] * 5, index=dates, columns=stocks
    )
    price05 = pd.DataFrame(
        [[10.0, 10.0, 15.0]] * 5, index=dates, columns=stocks
    )
    return price95, price05


class TestMetadata:
    def test_name(self, factor):
        assert factor.name == "CHIP_CONCENTRATION"

    def test_category(self, factor):
        assert factor.category == "行为金融"

    def test_description(self, factor):
        assert "筹码集中度" in factor.description

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "CHIP_CONCENTRATION"
        assert meta["category"] == "行为金融"

    def test_repr(self, factor):
        r = repr(factor)
        assert "ChipConcentrationFactor" in r
        assert "CHIP_CONCENTRATION" in r


class TestCompute:
    def test_known_values(self, factor, sample_data):
        price95, price05 = sample_data
        result = factor.compute(price95=price95, price05=price05)
        # 2*(15-10)/(15+10) = 10/25 = 0.4
        assert pytest.approx(result.iloc[0, 0], rel=1e-6) == 0.4
        # 2*(20-10)/(20+10) = 20/30 = 0.6667
        assert pytest.approx(result.iloc[0, 1], rel=1e-4) == 2 / 3
        # 2*(25-15)/(25+15) = 20/40 = 0.5
        assert pytest.approx(result.iloc[0, 2], rel=1e-6) == 0.5

    def test_output_shape(self, factor, sample_data):
        price95, price05 = sample_data
        result = factor.compute(price95=price95, price05=price05)
        assert result.shape == price95.shape

    def test_equal_prices_zero(self, factor):
        dates = pd.date_range("2024-01-01", periods=3)
        stocks = ["A"]
        price = pd.DataFrame([[10.0]] * 3, index=dates, columns=stocks)
        result = factor.compute(price95=price, price05=price)
        assert (result == 0).all().all()

    def test_output_range(self, factor):
        """筹码集中度应在[0, 2)范围内（price95 >= price05 > 0时）"""
        dates = pd.date_range("2024-01-01", periods=10)
        stocks = ["A", "B"]
        np.random.seed(42)
        price05 = pd.DataFrame(
            np.random.uniform(5, 10, (10, 2)), index=dates, columns=stocks
        )
        price95 = price05 + np.random.uniform(0, 10, (10, 2))
        result = factor.compute(price95=price95, price05=price05)
        assert (result >= 0).all().all()
        assert (result < 2).all().all()

    def test_nan_handling(self, factor):
        dates = pd.date_range("2024-01-01", periods=3)
        stocks = ["A"]
        price95 = pd.DataFrame([[10.0], [np.nan], [12.0]], index=dates, columns=stocks)
        price05 = pd.DataFrame([[5.0], [6.0], [np.nan]], index=dates, columns=stocks)
        result = factor.compute(price95=price95, price05=price05)
        assert not np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert np.isnan(result.iloc[2, 0])
