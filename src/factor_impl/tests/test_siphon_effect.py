import numpy as np
import pandas as pd
import pytest

from factors.siphon_effect import SiphonEffectFactor


@pytest.fixture
def factor():
    return SiphonEffectFactor()


class TestSiphonEffectMetadata:
    def test_name(self, factor):
        assert factor.name == "SIPHON_EFFECT"

    def test_category(self, factor):
        assert factor.category == "高频资金流"

    def test_repr(self, factor):
        assert "SIPHON_EFFECT" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "SIPHON_EFFECT"
        assert meta["category"] == "高频资金流"


class TestSiphonEffectHandCalculated:
    """手算验证 0.5*rolling_mean + 0.5*rolling_std。"""

    def test_constant_input(self, factor):
        """常数输入时, std=0, 因子值 = 0.5*常数 + 0.5*0 = 0.5*常数。
        注意: rolling(min_periods=1).std 对单个值返回 NaN, 对2+个相同值返回 0。
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(0.4, index=dates, columns=stocks)

        result = factor.compute(daily_net_siphon=daily, T=3)

        # day0: mean=0.4, std=NaN => 0.5*0.4 + 0.5*NaN = NaN
        assert np.isnan(result.iloc[0, 0])
        # day1+: mean=0.4, std=0.0 => 0.5*0.4 + 0.5*0.0 = 0.2
        for i in range(1, 5):
            assert result.iloc[i, 0] == pytest.approx(0.2, rel=1e-10)

    def test_varying_input_T3(self, factor):
        """T=3, data=[0.1, 0.2, 0.3, 0.4, 0.5]

        rolling(3, min_periods=1) mean:
          day0: 0.1
          day1: 0.15
          day2: 0.2
          day3: 0.3
          day4: 0.4

        rolling(3, min_periods=1) std (ddof=1):
          day0: NaN (单个值)
          day1: std([0.1, 0.2]) = 0.07071...
          day2: std([0.1, 0.2, 0.3]) = 0.1
          day3: std([0.2, 0.3, 0.4]) = 0.1
          day4: std([0.3, 0.4, 0.5]) = 0.1
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(
            [0.1, 0.2, 0.3, 0.4, 0.5], index=dates, columns=stocks
        )

        result = factor.compute(daily_net_siphon=daily, T=3)

        # day0: 0.5*0.1 + 0.5*NaN = NaN
        assert np.isnan(result.iloc[0, 0])
        # day1: 0.5*0.15 + 0.5*0.07071 = 0.075 + 0.03536 = 0.11036
        std_day1 = np.std([0.1, 0.2], ddof=1)
        expected_day1 = 0.5 * 0.15 + 0.5 * std_day1
        assert result.iloc[1, 0] == pytest.approx(expected_day1, rel=1e-6)
        # day2: 0.5*0.2 + 0.5*0.1 = 0.15
        assert result.iloc[2, 0] == pytest.approx(0.15, rel=1e-6)
        # day3: 0.5*0.3 + 0.5*0.1 = 0.2
        assert result.iloc[3, 0] == pytest.approx(0.2, rel=1e-6)
        # day4: 0.5*0.4 + 0.5*0.1 = 0.25
        assert result.iloc[4, 0] == pytest.approx(0.25, rel=1e-6)

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A", "B"]
        daily = pd.DataFrame(
            {"A": [0.2] * 5, "B": [0.5] * 5}, index=dates
        )

        result = factor.compute(daily_net_siphon=daily, T=3)

        # 常数输入, std=0 (从 day1 开始), mean=常数
        # day1+: A = 0.5*0.2 = 0.1, B = 0.5*0.5 = 0.25
        for i in range(1, 5):
            assert result.iloc[i]["A"] == pytest.approx(0.1, rel=1e-10)
            assert result.iloc[i]["B"] == pytest.approx(0.25, rel=1e-10)


class TestSiphonEffectEdgeCases:
    def test_single_value(self, factor):
        """单个数据点: mean=val, std=NaN => result=NaN。"""
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame([0.42], index=dates, columns=stocks)

        result = factor.compute(daily_net_siphon=daily, T=20)
        assert np.isnan(result.iloc[0, 0])

    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        values = [0.3, np.nan, 0.4, 0.5, 0.2]
        daily = pd.DataFrame(values, index=dates, columns=stocks)

        result = factor.compute(daily_net_siphon=daily, T=3)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (5, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(daily_net_siphon=daily, T=3)
        assert result.isna().all().all()

    def test_negative_input(self, factor):
        """净虹吸效应可以为负值。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame(
            [-0.3, -0.2, -0.1, -0.4, -0.5], index=dates, columns=stocks
        )

        result = factor.compute(daily_net_siphon=daily, T=3)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (5, 1)


class TestSiphonEffectOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        daily = pd.DataFrame(
            np.random.uniform(-1, 1, (30, 3)), index=dates, columns=stocks
        )

        result = factor.compute(daily_net_siphon=daily, T=20)

        assert result.shape == daily.shape
        assert list(result.columns) == list(daily.columns)
        assert list(result.index) == list(daily.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        daily = pd.DataFrame([0.1, 0.2, 0.3, 0.4, 0.5], index=dates, columns=stocks)

        result = factor.compute(daily_net_siphon=daily, T=3)
        assert isinstance(result, pd.DataFrame)
