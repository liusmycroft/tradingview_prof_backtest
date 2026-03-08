import numpy as np
import pandas as pd
import pytest

from factors.volume_proportion_composite import VolumeProportionCompositeFactor


@pytest.fixture
def factor():
    return VolumeProportionCompositeFactor()


class TestVolumeProportionCompositeMetadata:
    def test_name(self, factor):
        assert factor.name == "OBCVP"

    def test_category(self, factor):
        assert factor.category == "高频成交分布"

    def test_repr(self, factor):
        assert "OBCVP" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "OBCVP"
        assert meta["category"] == "高频成交分布"


class TestVolumeProportionCompositeHandCalculated:
    """手算验证 OBCVP = alpha * OCVP + (1-alpha) * BCVP"""

    def test_constant_ratio_T3(self, factor):
        """常数占比时，滚动均值等于该常数。

        opening = 100, closing = 200, daily = 1000
        OCVP = 0.1, BCVP = 0.2
        alpha=0.5: OBCVP = 0.5*0.1 + 0.5*0.2 = 0.15
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        opening = pd.DataFrame(100.0, index=dates, columns=stocks)
        closing = pd.DataFrame(200.0, index=dates, columns=stocks)
        daily = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(
            opening_auction_volume=opening,
            closing_auction_volume=closing,
            daily_volume=daily,
            T=3,
            alpha=0.5,
        )
        assert result.iloc[2, 0] == pytest.approx(0.15, rel=1e-10)
        assert result.iloc[4, 0] == pytest.approx(0.15, rel=1e-10)

    def test_varying_ratio_T3(self, factor):
        """T=3, 手动验证不同占比的滚动均值。

        opening = [100, 200, 300, 400, 500]
        closing = [50,  100, 150, 200, 250]
        daily   = [1000]*5

        o_ratio = [0.1, 0.2, 0.3, 0.4, 0.5]
        c_ratio = [0.05, 0.1, 0.15, 0.2, 0.25]

        T=3, alpha=0.5:
          row 2: OCVP=mean(0.1,0.2,0.3)=0.2, BCVP=mean(0.05,0.1,0.15)=0.1
                 OBCVP = 0.5*0.2 + 0.5*0.1 = 0.15
          row 3: OCVP=mean(0.2,0.3,0.4)=0.3, BCVP=mean(0.1,0.15,0.2)=0.15
                 OBCVP = 0.5*0.3 + 0.5*0.15 = 0.225
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        opening = pd.DataFrame([100.0, 200.0, 300.0, 400.0, 500.0], index=dates, columns=stocks)
        closing = pd.DataFrame([50.0, 100.0, 150.0, 200.0, 250.0], index=dates, columns=stocks)
        daily = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(
            opening_auction_volume=opening,
            closing_auction_volume=closing,
            daily_volume=daily,
            T=3,
            alpha=0.5,
        )

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert result.iloc[2, 0] == pytest.approx(0.15, rel=1e-10)
        assert result.iloc[3, 0] == pytest.approx(0.225, rel=1e-10)

    def test_alpha_zero(self, factor):
        """alpha=0 时只看收盘占比。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]
        opening = pd.DataFrame(100.0, index=dates, columns=stocks)
        closing = pd.DataFrame(300.0, index=dates, columns=stocks)
        daily = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(
            opening_auction_volume=opening,
            closing_auction_volume=closing,
            daily_volume=daily,
            T=3,
            alpha=0.0,
        )
        assert result.iloc[2, 0] == pytest.approx(0.3, rel=1e-10)

    def test_alpha_one(self, factor):
        """alpha=1 时只看开盘占比。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]
        opening = pd.DataFrame(100.0, index=dates, columns=stocks)
        closing = pd.DataFrame(300.0, index=dates, columns=stocks)
        daily = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(
            opening_auction_volume=opening,
            closing_auction_volume=closing,
            daily_volume=daily,
            T=3,
            alpha=1.0,
        )
        assert result.iloc[2, 0] == pytest.approx(0.1, rel=1e-10)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A", "B"]
        opening = pd.DataFrame({"A": [100.0] * 5, "B": [200.0] * 5}, index=dates)
        closing = pd.DataFrame({"A": [50.0] * 5, "B": [100.0] * 5}, index=dates)
        daily = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(
            opening_auction_volume=opening,
            closing_auction_volume=closing,
            daily_volume=daily,
            T=3,
            alpha=0.5,
        )
        # A: 0.5*0.1 + 0.5*0.05 = 0.075
        assert result.iloc[2, 0] == pytest.approx(0.075, rel=1e-10)
        # B: 0.5*0.2 + 0.5*0.1 = 0.15
        assert result.iloc[2, 1] == pytest.approx(0.15, rel=1e-10)


class TestVolumeProportionCompositeEdgeCases:
    def test_nan_in_input(self, factor):
        """输入含 NaN 时不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        opening = pd.DataFrame([100.0, np.nan, 300.0, 400.0, 500.0], index=dates, columns=stocks)
        closing = pd.DataFrame(200.0, index=dates, columns=stocks)
        daily = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(
            opening_auction_volume=opening,
            closing_auction_volume=closing,
            daily_volume=daily,
            T=3,
        )
        assert isinstance(result, pd.DataFrame)

    def test_zero_daily_volume(self, factor):
        """daily_volume 为 0 时不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        opening = pd.DataFrame(100.0, index=dates, columns=stocks)
        closing = pd.DataFrame(200.0, index=dates, columns=stocks)
        daily = pd.DataFrame([0.0, 1000.0, 1000.0, 1000.0, 1000.0], index=dates, columns=stocks)

        result = factor.compute(
            opening_auction_volume=opening,
            closing_auction_volume=closing,
            daily_volume=daily,
            T=3,
        )
        assert isinstance(result, pd.DataFrame)

    def test_insufficient_window(self, factor):
        """数据不足 T 天时应返回 NaN。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]
        opening = pd.DataFrame(100.0, index=dates, columns=stocks)
        closing = pd.DataFrame(200.0, index=dates, columns=stocks)
        daily = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(
            opening_auction_volume=opening,
            closing_auction_volume=closing,
            daily_volume=daily,
            T=5,
        )
        assert result.isna().all().all()


class TestVolumeProportionCompositeOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        opening = pd.DataFrame(np.random.uniform(50, 200, (30, 3)), index=dates, columns=stocks)
        closing = pd.DataFrame(np.random.uniform(50, 200, (30, 3)), index=dates, columns=stocks)
        daily = pd.DataFrame(np.random.uniform(500, 2000, (30, 3)), index=dates, columns=stocks)

        result = factor.compute(
            opening_auction_volume=opening,
            closing_auction_volume=closing,
            daily_volume=daily,
            T=20,
        )
        assert result.shape == opening.shape
        assert list(result.columns) == list(opening.columns)
        assert list(result.index) == list(opening.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        opening = pd.DataFrame(100.0, index=dates, columns=stocks)
        closing = pd.DataFrame(200.0, index=dates, columns=stocks)
        daily = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(
            opening_auction_volume=opening,
            closing_auction_volume=closing,
            daily_volume=daily,
            T=3,
        )
        assert isinstance(result, pd.DataFrame)

    def test_first_T_minus_1_rows_are_nan(self, factor):
        """前 T-1 行应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        stocks = ["A", "B"]
        T = 5
        opening = pd.DataFrame(100.0, index=dates, columns=stocks)
        closing = pd.DataFrame(200.0, index=dates, columns=stocks)
        daily = pd.DataFrame(1000.0, index=dates, columns=stocks)

        result = factor.compute(
            opening_auction_volume=opening,
            closing_auction_volume=closing,
            daily_volume=daily,
            T=T,
        )
        assert result.iloc[: T - 1].isna().all().all()
        assert result.iloc[T - 1 :].notna().all().all()
