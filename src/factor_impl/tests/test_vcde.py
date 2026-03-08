import numpy as np
import pandas as pd
import pytest

from factors.vcde import VcdeFactor


@pytest.fixture
def factor():
    return VcdeFactor()


class TestVcdeMetadata:
    def test_name(self, factor):
        assert factor.name == "VCDE"

    def test_category(self, factor):
        assert factor.category == "行为金融-处置效应"

    def test_repr(self, factor):
        assert "VCDE" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "VCDE"
        assert meta["category"] == "行为金融-处置效应"


class TestVcdeHandCalculated:
    """手算验证三种 VCDE 变体。"""

    def test_vcde1_basic(self, factor):
        """VCDE1 = |CPGR - CPLR|

        CPGR=0.6, CPLR=0.2 => |0.6-0.2| = 0.4
        T=1 时 EMA 就是自身。
        """
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A"]
        cpgr = pd.DataFrame([0.6], index=dates, columns=stocks)
        cplr = pd.DataFrame([0.2], index=dates, columns=stocks)

        result = factor.compute(cpgr=cpgr, cplr=cplr, variant=1, T=1)
        assert result.iloc[0, 0] == pytest.approx(0.4, rel=1e-10)

    def test_vcde1_symmetric(self, factor):
        """VCDE1 对称: |0.2-0.6| = |0.6-0.2| = 0.4"""
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A"]
        cpgr = pd.DataFrame([0.2], index=dates, columns=stocks)
        cplr = pd.DataFrame([0.6], index=dates, columns=stocks)

        result = factor.compute(cpgr=cpgr, cplr=cplr, variant=1, T=1)
        assert result.iloc[0, 0] == pytest.approx(0.4, rel=1e-10)

    def test_vcde2_basic(self, factor):
        """VCDE2 = CPGR + 0.23 * CPLR

        CPGR=0.5, CPLR=1.0 => 0.5 + 0.23*1.0 = 0.73
        """
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A"]
        cpgr = pd.DataFrame([0.5], index=dates, columns=stocks)
        cplr = pd.DataFrame([1.0], index=dates, columns=stocks)

        result = factor.compute(cpgr=cpgr, cplr=cplr, variant=2, T=1)
        assert result.iloc[0, 0] == pytest.approx(0.73, rel=1e-10)

    def test_vcde3_basic(self, factor):
        """VCDE3 = CPGR + CPLR

        CPGR=0.3, CPLR=0.7 => 1.0
        """
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A"]
        cpgr = pd.DataFrame([0.3], index=dates, columns=stocks)
        cplr = pd.DataFrame([0.7], index=dates, columns=stocks)

        result = factor.compute(cpgr=cpgr, cplr=cplr, variant=3, T=1)
        assert result.iloc[0, 0] == pytest.approx(1.0, rel=1e-10)

    def test_ema_smoothing_T3(self, factor):
        """验证 EMA 平滑 (T=3)。

        VCDE1 daily = [0.4, 0.2, 0.6, 0.8]
        ewm(span=3, adjust=True), alpha=0.5
        ema_0 = 0.4
        ema_1 = (0.5*0.4 + 1.0*0.2) / 1.5 = 4/15
        ema_2 = (0.25*0.4 + 0.5*0.2 + 1.0*0.6) / 1.75 = 0.8/1.75 = 16/35
        ema_3 = (0.125*0.4 + 0.25*0.2 + 0.5*0.6 + 1.0*0.8) / 1.875 = 1.2/1.875 = 0.64
        """
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        stocks = ["A"]
        cpgr = pd.DataFrame([0.5, 0.3, 0.7, 0.9], index=dates, columns=stocks)
        cplr = pd.DataFrame([0.1, 0.1, 0.1, 0.1], index=dates, columns=stocks)

        result = factor.compute(cpgr=cpgr, cplr=cplr, variant=1, T=3)

        # VCDE1 = |CPGR - CPLR| = [0.4, 0.2, 0.6, 0.8]
        assert result.iloc[0, 0] == pytest.approx(0.4, rel=1e-6)
        assert result.iloc[1, 0] == pytest.approx(4.0 / 15, rel=1e-6)
        assert result.iloc[2, 0] == pytest.approx(16.0 / 35, rel=1e-6)
        assert result.iloc[3, 0] == pytest.approx(0.64, rel=1e-6)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A", "B"]
        cpgr = pd.DataFrame({"A": [0.5] * 5, "B": [0.3] * 5}, index=dates)
        cplr = pd.DataFrame({"A": [0.1] * 5, "B": [0.1] * 5}, index=dates)

        result = factor.compute(cpgr=cpgr, cplr=cplr, variant=1, T=5)

        # A: |0.5-0.1|=0.4 constant => EMA=0.4
        # B: |0.3-0.1|=0.2 constant => EMA=0.2
        np.testing.assert_array_almost_equal(result["A"].values, 0.4)
        np.testing.assert_array_almost_equal(result["B"].values, 0.2)

    def test_invalid_variant_raises(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A"]
        cpgr = pd.DataFrame([0.5], index=dates, columns=stocks)
        cplr = pd.DataFrame([0.1], index=dates, columns=stocks)

        with pytest.raises(ValueError, match="variant must be 1, 2, or 3"):
            factor.compute(cpgr=cpgr, cplr=cplr, variant=4)


class TestVcdeEdgeCases:
    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        cpgr_vals = [0.5, np.nan, 0.3, 0.4, 0.6]
        cpgr = pd.DataFrame(cpgr_vals, index=dates, columns=stocks)
        cplr = pd.DataFrame([0.1] * 5, index=dates, columns=stocks)

        result = factor.compute(cpgr=cpgr, cplr=cplr, variant=1, T=3)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (5, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        cpgr = pd.DataFrame(np.nan, index=dates, columns=stocks)
        cplr = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(cpgr=cpgr, cplr=cplr, variant=1, T=3)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        cpgr = pd.DataFrame(0.0, index=dates, columns=stocks)
        cplr = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(cpgr=cpgr, cplr=cplr, variant=1, T=3)
        for val in result["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)

    def test_equal_cpgr_cplr_vcde1_zero(self, factor):
        """CPGR == CPLR 时, VCDE1 = 0。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]
        cpgr = pd.DataFrame([0.5, 0.5, 0.5], index=dates, columns=stocks)
        cplr = pd.DataFrame([0.5, 0.5, 0.5], index=dates, columns=stocks)

        result = factor.compute(cpgr=cpgr, cplr=cplr, variant=1, T=3)
        for val in result["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)


class TestVcdeOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        cpgr = pd.DataFrame(
            np.random.uniform(0, 1, (30, 3)), index=dates, columns=stocks
        )
        cplr = pd.DataFrame(
            np.random.uniform(0, 1, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(cpgr=cpgr, cplr=cplr, variant=1, T=20)

        assert result.shape == cpgr.shape
        assert list(result.columns) == list(cpgr.columns)
        assert list(result.index) == list(cpgr.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        cpgr = pd.DataFrame([0.5] * 5, index=dates, columns=stocks)
        cplr = pd.DataFrame([0.1] * 5, index=dates, columns=stocks)

        for v in [1, 2, 3]:
            result = factor.compute(cpgr=cpgr, cplr=cplr, variant=v, T=3)
            assert isinstance(result, pd.DataFrame)

    def test_no_leading_nan_min_periods_1(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        cpgr = pd.DataFrame(
            np.random.uniform(0, 1, (10, 2)), index=dates, columns=stocks
        )
        cplr = pd.DataFrame(
            np.random.uniform(0, 1, (10, 2)), index=dates, columns=stocks
        )

        result = factor.compute(cpgr=cpgr, cplr=cplr, variant=1, T=20)
        assert result.iloc[0].notna().all()
        assert result.notna().all().all()
