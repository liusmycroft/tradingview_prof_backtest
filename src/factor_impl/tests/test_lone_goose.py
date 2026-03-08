import numpy as np
import pandas as pd
import pytest

from factors.lone_goose import LoneGooseFactor


@pytest.fixture
def factor():
    return LoneGooseFactor()


class TestLoneGooseMetadata:
    def test_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "LONE_GOOSE"
        assert meta["category"] == "高频量价"
        assert "孤雁" in meta["description"]

    def test_repr(self, factor):
        r = repr(factor)
        assert "LoneGooseFactor" in r
        assert "LONE_GOOSE" in r


class TestLoneGooseCompute:
    def test_known_values(self, factor):
        """用已知数据验证等权组合计算。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        daily = pd.DataFrame({"A": [0.2, 0.4, 0.6]}, index=dates)

        result = factor.compute(daily_lone_goose=daily, T=3)

        # 均值 = (0.2+0.4+0.6)/3 = 0.4
        # 标准差 = std([0.2, 0.4, 0.6], ddof=1) = 0.2
        expected_mean = 0.4
        expected_std = np.std([0.2, 0.4, 0.6], ddof=1)
        expected = 0.5 * expected_mean + 0.5 * expected_std

        np.testing.assert_almost_equal(result.iloc[2, 0], expected)

    def test_constant_input(self, factor):
        """输入为常数时，标准差为 0，结果应为 0.5 * mean。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        daily = pd.DataFrame({"A": [0.5, 0.5, 0.5]}, index=dates)

        result = factor.compute(daily_lone_goose=daily, T=3)

        expected = 0.5 * 0.5 + 0.5 * 0.0
        np.testing.assert_almost_equal(result.iloc[2, 0], expected)

    def test_min_periods(self, factor):
        """窗口不足 T 天时，结果应为 NaN。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        daily = pd.DataFrame({"A": [0.1] * 5}, index=dates)

        result = factor.compute(daily_lone_goose=daily, T=20)

        assert result.isna().all().all()

    def test_multi_stock(self, factor):
        """多只股票同时计算。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        daily = pd.DataFrame(
            {"A": [0.2, 0.4, 0.6], "B": [0.1, 0.1, 0.1]}, index=dates
        )

        result = factor.compute(daily_lone_goose=daily, T=3)

        assert result.shape == (3, 2)
        # B 列标准差为 0
        expected_b = 0.5 * 0.1 + 0.5 * 0.0
        np.testing.assert_almost_equal(result.loc[dates[2], "B"], expected_b)

    def test_output_shape(self, factor):
        """输出形状应与输入一致。"""
        dates = pd.bdate_range("2025-01-01", periods=25)
        daily = pd.DataFrame(
            np.random.rand(25, 4),
            index=dates,
            columns=["A", "B", "C", "D"],
        )

        result = factor.compute(daily_lone_goose=daily, T=20)

        assert result.shape == (25, 4)
        assert result.iloc[:19].isna().all().all()
        assert result.iloc[19:].notna().all().all()

    def test_nan_propagation(self, factor):
        """输入含 NaN 时，对应窗口结果也应为 NaN。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        daily = pd.DataFrame({"A": [0.2, np.nan, 0.6]}, index=dates)

        result = factor.compute(daily_lone_goose=daily, T=2)

        # window [0,1] 含 NaN -> NaN; window [1,2] 含 NaN -> NaN
        assert np.isnan(result.iloc[1, 0])
        assert np.isnan(result.iloc[2, 0])
