import numpy as np
import pandas as pd
import pytest

from factors.v_shaped_disposition import VShapedDispositionFactor


@pytest.fixture
def factor():
    return VShapedDispositionFactor()


class TestVShapedDispositionMetadata:
    def test_name(self, factor):
        assert factor.name == "VNSP"

    def test_category(self, factor):
        assert factor.category == "行为金融-处置效应"

    def test_repr(self, factor):
        assert "VNSP" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "VNSP"
        assert meta["category"] == "行为金融-处置效应"


class TestVShapedDispositionHandCalculated:
    """手算验证 EWM(span=T) of (Gain + sigma * |Loss|)。"""

    def test_constant_input(self, factor):
        """常数输入时, EMA 应等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        gain = pd.DataFrame(0.3, index=dates, columns=stocks)
        loss = pd.DataFrame(-0.2, index=dates, columns=stocks)

        result = factor.compute(daily_gain=gain, daily_loss=loss, sigma=1.0, T=20)
        # VNSP = 0.3 + 1.0 * |-0.2| = 0.5
        np.testing.assert_array_almost_equal(result["A"].values, 0.5)

    def test_sigma_scaling(self, factor):
        """sigma 参数应正确缩放亏损端。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        gain = pd.DataFrame(0.3, index=dates, columns=stocks)
        loss = pd.DataFrame(-0.2, index=dates, columns=stocks)

        result = factor.compute(daily_gain=gain, daily_loss=loss, sigma=2.0, T=20)
        # VNSP = 0.3 + 2.0 * 0.2 = 0.7
        np.testing.assert_array_almost_equal(result["A"].values, 0.7)

    def test_ema_manual_T3(self, factor):
        """T=3, 手动验证 EMA 值。

        gain = [0.1, 0.2, 0.3, 0.4]
        loss = [-0.1, -0.1, -0.1, -0.1]
        sigma = 1.0
        vnsp = [0.2, 0.3, 0.4, 0.5]

        ewm(span=3, adjust=True), alpha = 2/(3+1) = 0.5
          ema_0 = 0.2
          ema_1 = (0.5*0.2 + 1.0*0.3) / (0.5+1.0) = 0.4/1.5 = 0.26667
          ema_2 = (0.25*0.2 + 0.5*0.3 + 1.0*0.4) / (0.25+0.5+1.0) = 0.6/1.75 = 0.34286
          ema_3 = (0.125*0.2 + 0.25*0.3 + 0.5*0.4 + 1.0*0.5) / (0.125+0.25+0.5+1.0) = 0.8/1.875 = 0.42667
        """
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        stocks = ["A"]
        gain = pd.DataFrame([0.1, 0.2, 0.3, 0.4], index=dates, columns=stocks)
        loss = pd.DataFrame([-0.1, -0.1, -0.1, -0.1], index=dates, columns=stocks)

        result = factor.compute(daily_gain=gain, daily_loss=loss, sigma=1.0, T=3)

        assert result.iloc[0, 0] == pytest.approx(0.2, rel=1e-6)
        assert result.iloc[1, 0] == pytest.approx(0.4 / 1.5, rel=1e-6)
        assert result.iloc[2, 0] == pytest.approx(0.6 / 1.75, rel=1e-6)
        assert result.iloc[3, 0] == pytest.approx(0.8 / 1.875, rel=1e-6)

    def test_loss_already_positive(self, factor):
        """如果 loss 传入正值，abs() 仍然正确。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        stocks = ["A"]
        gain = pd.DataFrame(0.3, index=dates, columns=stocks)
        loss = pd.DataFrame(0.2, index=dates, columns=stocks)

        result = factor.compute(daily_gain=gain, daily_loss=loss, sigma=1.0, T=20)
        np.testing.assert_array_almost_equal(result["A"].values, 0.5)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        gain = pd.DataFrame({"A": [0.1] * 20, "B": [0.5] * 20}, index=dates)
        loss = pd.DataFrame({"A": [-0.1] * 20, "B": [-0.3] * 20}, index=dates)

        result = factor.compute(daily_gain=gain, daily_loss=loss, sigma=1.0, T=20)
        np.testing.assert_array_almost_equal(result["A"].values, 0.2)
        np.testing.assert_array_almost_equal(result["B"].values, 0.8)


class TestVShapedDispositionEdgeCases:
    def test_single_value(self, factor):
        """单个数据点的 EMA 应等于该值 (min_periods=1)。"""
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A"]
        gain = pd.DataFrame([0.5], index=dates, columns=stocks)
        loss = pd.DataFrame([-0.3], index=dates, columns=stocks)

        result = factor.compute(daily_gain=gain, daily_loss=loss, sigma=1.0, T=20)
        assert result.iloc[0, 0] == pytest.approx(0.8, rel=1e-10)

    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        gain = pd.DataFrame(np.ones(10) * 0.1, index=dates, columns=stocks)
        gain.iloc[3, 0] = np.nan
        loss = pd.DataFrame(-0.1, index=dates, columns=stocks)

        result = factor.compute(daily_gain=gain, daily_loss=loss, sigma=1.0, T=5)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        gain = pd.DataFrame(np.nan, index=dates, columns=stocks)
        loss = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_gain=gain, daily_loss=loss, sigma=1.0, T=5)
        assert result.isna().all().all()

    def test_zero_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A"]
        gain = pd.DataFrame(0.0, index=dates, columns=stocks)
        loss = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(daily_gain=gain, daily_loss=loss, sigma=1.0, T=5)
        for val in result["A"].values:
            assert val == pytest.approx(0.0, abs=1e-15)


class TestVShapedDispositionOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        gain = pd.DataFrame(
            np.random.uniform(0, 1, (30, 3)), index=dates, columns=stocks
        )
        loss = pd.DataFrame(
            np.random.uniform(-1, 0, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_gain=gain, daily_loss=loss, sigma=1.0, T=20)

        assert result.shape == gain.shape
        assert list(result.columns) == list(gain.columns)
        assert list(result.index) == list(gain.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        gain = pd.DataFrame([0.1, 0.2, 0.3, 0.4, 0.5], index=dates, columns=stocks)
        loss = pd.DataFrame([-0.1] * 5, index=dates, columns=stocks)

        result = factor.compute(daily_gain=gain, daily_loss=loss, sigma=1.0, T=3)
        assert isinstance(result, pd.DataFrame)

    def test_no_leading_nan_min_periods_1(self, factor):
        """min_periods=1, 所以第一行就有值。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        gain = pd.DataFrame(
            np.random.uniform(0, 1, (10, 2)), index=dates, columns=stocks
        )
        loss = pd.DataFrame(
            np.random.uniform(-1, 0, (10, 2)), index=dates, columns=stocks
        )

        result = factor.compute(daily_gain=gain, daily_loss=loss, sigma=1.0, T=20)
        assert result.iloc[0].notna().all()
        assert result.notna().all().all()
