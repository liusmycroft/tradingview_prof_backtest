import numpy as np
import pandas as pd
import pytest

from factors.idio_turnover_vol import IdioTurnoverVolFactor


@pytest.fixture
def factor():
    return IdioTurnoverVolFactor()


@pytest.fixture
def sample_data():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    stocks = ["A", "B", "C"]
    turnover = pd.DataFrame(
        np.random.uniform(0.01, 0.1, (30, 3)), index=dates, columns=stocks
    )
    tmkt = pd.Series(np.random.uniform(0.03, 0.06, 30), index=dates)
    tsmb = pd.Series(np.random.uniform(-0.01, 0.01, 30), index=dates)
    thml = pd.Series(np.random.uniform(-0.01, 0.01, 30), index=dates)
    tmom = pd.Series(np.random.uniform(-0.01, 0.01, 30), index=dates)
    return turnover, tmkt, tsmb, thml, tmom


class TestIdioTurnoverVolMetadata:
    def test_name(self, factor):
        assert factor.name == "IDIO_TURNOVER_VOL"

    def test_category(self, factor):
        assert factor.category == "量价因子改进"

    def test_repr(self, factor):
        assert "IDIO_TURNOVER_VOL" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "IDIO_TURNOVER_VOL"


class TestIdioTurnoverVolCompute:
    def test_output_non_negative(self, factor, sample_data):
        """标准差应非负。"""
        turnover, tmkt, tsmb, thml, tmom = sample_data
        result = factor.compute(
            turnover=turnover, tmkt=tmkt, tsmb=tsmb, thml=thml, tmom=tmom, T=20
        )
        valid = result.values[~np.isnan(result.values)]
        assert (valid >= -1e-10).all()

    def test_constant_turnover(self, factor):
        """常数换手率：残差应为0，std应为0。"""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        turnover = pd.DataFrame(0.05, index=dates, columns=stocks)
        tmkt = pd.Series(0.05, index=dates)
        tsmb = pd.Series(0.0, index=dates)
        thml = pd.Series(0.0, index=dates)
        tmom = pd.Series(0.0, index=dates)

        result = factor.compute(
            turnover=turnover, tmkt=tmkt, tsmb=tsmb, thml=thml, tmom=tmom, T=20
        )
        valid = result.dropna()
        if len(valid) > 0:
            for col in stocks:
                vals = valid[col].values
                assert all(v == pytest.approx(0.0, abs=1e-8) or np.isnan(v) for v in vals)

    def test_leading_nan(self, factor, sample_data):
        """前 T-1 行应为 NaN。"""
        turnover, tmkt, tsmb, thml, tmom = sample_data
        result = factor.compute(
            turnover=turnover, tmkt=tmkt, tsmb=tsmb, thml=thml, tmom=tmom, T=20
        )
        assert result.iloc[:19].isna().all().all()


class TestIdioTurnoverVolEdgeCases:
    def test_nan_in_input(self, factor, sample_data):
        turnover, tmkt, tsmb, thml, tmom = sample_data
        turnover.iloc[5, 0] = np.nan
        result = factor.compute(
            turnover=turnover, tmkt=tmkt, tsmb=tsmb, thml=thml, tmom=tmom, T=20
        )
        assert isinstance(result, pd.DataFrame)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A"]
        turnover = pd.DataFrame(np.nan, index=dates, columns=stocks)
        tmkt = pd.Series(np.nan, index=dates)
        tsmb = pd.Series(np.nan, index=dates)
        thml = pd.Series(np.nan, index=dates)
        tmom = pd.Series(np.nan, index=dates)

        result = factor.compute(
            turnover=turnover, tmkt=tmkt, tsmb=tsmb, thml=thml, tmom=tmom, T=20
        )
        assert result.isna().all().all()


class TestIdioTurnoverVolOutputShape:
    def test_output_shape(self, factor, sample_data):
        turnover, tmkt, tsmb, thml, tmom = sample_data
        result = factor.compute(
            turnover=turnover, tmkt=tmkt, tsmb=tsmb, thml=thml, tmom=tmom, T=20
        )
        assert result.shape == turnover.shape

    def test_output_is_dataframe(self, factor, sample_data):
        turnover, tmkt, tsmb, thml, tmom = sample_data
        result = factor.compute(
            turnover=turnover, tmkt=tmkt, tsmb=tsmb, thml=thml, tmom=tmom, T=20
        )
        assert isinstance(result, pd.DataFrame)
