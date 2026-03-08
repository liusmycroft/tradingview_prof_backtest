import numpy as np
import pandas as pd
import pytest

from factors.acma import ACMAFactor


@pytest.fixture
def factor():
    return ACMAFactor()


class TestACMAMetadata:
    def test_name(self, factor):
        assert factor.name == "ACMA"

    def test_category(self, factor):
        assert factor.category == "高频量价相关性"

    def test_repr(self, factor):
        assert "ACMA" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "ACMA"
        assert meta["category"] == "高频量价相关性"


class TestACMACompute:
    def test_basic_rolling_mean(self, factor):
        """验证滚动均值计算的正确性。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        data = pd.DataFrame({"A": [0.5, 0.6, 0.7, 0.8, 0.9]}, index=dates)

        result = factor.compute(daily_acma=data, T=3)

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(0.6)  # mean(0.5, 0.6, 0.7)
        assert result.iloc[3, 0] == pytest.approx(0.7)  # mean(0.6, 0.7, 0.8)
        assert result.iloc[4, 0] == pytest.approx(0.8)  # mean(0.7, 0.8, 0.9)

    def test_T20_known_values(self, factor):
        """T=20 时验证已知值。"""
        dates = pd.bdate_range("2025-01-01", periods=25)
        np.random.seed(77)
        vals = np.random.uniform(0.1, 0.9, 25)
        data = pd.DataFrame({"A": vals}, index=dates)

        result = factor.compute(daily_acma=data, T=20)

        assert result.iloc[:19].isna().all().all()
        expected = vals[:20].mean()
        assert result.iloc[19, 0] == pytest.approx(expected)

    def test_multi_stock(self, factor):
        """多只股票同时计算。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        data = pd.DataFrame(
            {"A": [0.5, 0.6, 0.7, 0.8, 0.9], "B": [0.9, 0.8, 0.7, 0.6, 0.5]},
            index=dates,
        )

        result = factor.compute(daily_acma=data, T=3)

        assert result.iloc[2, 0] == pytest.approx(0.6)
        assert result.iloc[2, 1] == pytest.approx(0.8)

    def test_constant_values(self, factor):
        """常数输入时均值等于该常数。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        data = pd.DataFrame({"A": [0.75] * 5}, index=dates)

        result = factor.compute(daily_acma=data, T=3)

        assert result.iloc[2, 0] == pytest.approx(0.75)


class TestACMAEdgeCases:
    def test_nan_in_input(self, factor):
        """输入含 NaN 时不应抛异常。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        data = pd.DataFrame({"A": [0.5, np.nan, 0.7, 0.8, 0.9]}, index=dates)

        result = factor.compute(daily_acma=data, T=3)

        assert isinstance(result, pd.DataFrame)
        assert np.isnan(result.iloc[2, 0])


class TestACMAOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=30)
        stocks = ["A", "B", "C"]
        data = pd.DataFrame(
            np.random.uniform(0.1, 0.9, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_acma=data, T=20)

        assert result.shape == data.shape
        assert list(result.columns) == list(data.columns)

    def test_output_is_dataframe(self, factor):
        dates = pd.bdate_range("2025-01-01", periods=5)
        data = pd.DataFrame({"A": [0.5, 0.6, 0.7, 0.8, 0.9]}, index=dates)

        result = factor.compute(daily_acma=data, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        """前 T-1 行应全为 NaN。"""
        dates = pd.bdate_range("2025-01-01", periods=25)
        data = pd.DataFrame(
            np.random.uniform(0.1, 0.9, (25, 2)), index=dates, columns=["A", "B"]
        )
        T = 20

        result = factor.compute(daily_acma=data, T=T)

        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
