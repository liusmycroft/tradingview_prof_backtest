import numpy as np
import pandas as pd
import pytest

from factors.rsj import RSJFactor


@pytest.fixture
def factor():
    return RSJFactor()


class TestRSJMetadata:
    def test_name(self, factor):
        assert factor.name == "RSJ"

    def test_category(self, factor):
        assert factor.category == "高频波动跳跃"

    def test_repr(self, factor):
        assert "RSJ" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "RSJ"
        assert meta["category"] == "高频波动跳跃"


class TestRSJHandCalculated:
    def test_constant_input(self, factor):
        """常数输入时, 滚动均值等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(0.3, index=dates, columns=stocks)

        result = factor.compute(daily_rsj=data, T=20)
        assert result.iloc[-1, 0] == pytest.approx(0.3, rel=1e-6)

    def test_varying_input_T3(self, factor):
        """T=3 手动验证。

        data = [0.1, 0.2, 0.3, 0.4, 0.5]
        T=3:
          row 2: mean(0.1, 0.2, 0.3) = 0.2
          row 3: mean(0.2, 0.3, 0.4) = 0.3
          row 4: mean(0.3, 0.4, 0.5) = 0.4
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        data = pd.DataFrame([0.1, 0.2, 0.3, 0.4, 0.5], index=dates, columns=stocks)

        result = factor.compute(daily_rsj=data, T=3)

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(0.2, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(0.3, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(0.4, rel=1e-10)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        data = pd.DataFrame({"A": [0.1] * 25, "B": [-0.2] * 25}, index=dates)

        result = factor.compute(daily_rsj=data, T=20)
        assert result.iloc[-1, 0] == pytest.approx(0.1, rel=1e-6)
        assert result.iloc[-1, 1] == pytest.approx(-0.2, rel=1e-6)


class TestRSJEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        values = np.ones(10) * 0.3
        values[3] = np.nan
        data = pd.DataFrame(values, index=dates, columns=stocks)

        result = factor.compute(daily_rsj=data, T=5)
        assert isinstance(result, pd.DataFrame)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_rsj=data, T=5)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_rsj=data, T=20)
        assert result.iloc[-1, 0] == pytest.approx(0.0, abs=1e-15)


class TestRSJOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        data = pd.DataFrame(
            np.random.uniform(-0.5, 0.5, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_rsj=data, T=20)
        assert result.shape == data.shape
        assert list(result.columns) == list(data.columns)
        assert list(result.index) == list(data.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        stocks = ["A"]
        data = pd.DataFrame(0.1, index=dates, columns=stocks)

        result = factor.compute(daily_rsj=data, T=20)
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        T = 20
        data = pd.DataFrame(
            np.random.uniform(-0.5, 0.5, (30, 2)), index=dates, columns=stocks
        )

        result = factor.compute(daily_rsj=data, T=T)
        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
