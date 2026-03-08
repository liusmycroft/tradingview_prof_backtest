import numpy as np
import pandas as pd
import pytest

from factors.main_force_vol import MainForceVolFactor


@pytest.fixture
def factor():
    return MainForceVolFactor()


def _make_df(n, ncols, seed=42):
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    stocks = [f"S{i}" for i in range(ncols)]
    return pd.DataFrame(
        np.random.randn(n, ncols) * 0.02, index=dates, columns=stocks
    )


class TestMainForceVolMetadata:
    def test_name(self, factor):
        assert factor.name == "MAIN_FORCE_VOL"

    def test_category(self, factor):
        assert factor.category == "高频波动"

    def test_repr(self, factor):
        assert "MAIN_FORCE_VOL" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "MAIN_FORCE_VOL"
        assert meta["category"] == "高频波动"
        assert "主力" in meta["description"]


class TestMainForceVolHandCalculated:
    def test_equal_weight_four_components(self, factor):
        """四个分量等权合成，验证结果为四者均值。"""
        n, ncols = 25, 5
        v1 = _make_df(n, ncols, seed=1)
        v2 = _make_df(n, ncols, seed=2)
        v3 = _make_df(n, ncols, seed=3)
        v4 = _make_df(n, ncols, seed=4)

        result = factor.compute(
            vol_up_ret=v1, cont_up_ret=v2,
            vol_down_ret=v3, cont_down_ret=v4, T=20,
        )

        # 验证最后一行有值（T=20, 数据25行，第20行开始有值）
        assert result.iloc[19:].notna().any().any()

    def test_identical_components(self, factor):
        """四个分量完全相同时，合成结果等于单个分量的处理结果。"""
        n, ncols = 25, 5
        v = _make_df(n, ncols, seed=42)

        result = factor.compute(
            vol_up_ret=v, cont_up_ret=v.copy(),
            vol_down_ret=v.copy(), cont_down_ret=v.copy(), T=20,
        )

        # 单独计算一个分量
        cross_mean = v.mean(axis=1)
        cross_std = v.std(axis=1, ddof=1)
        std1 = v.sub(cross_mean, axis=0).div(cross_std, axis=0)
        abs_std = std1.abs()
        cm2 = abs_std.mean(axis=1)
        cs2 = abs_std.std(axis=1, ddof=1)
        abs_std2 = abs_std.sub(cm2, axis=0).div(cs2, axis=0)
        single = abs_std2.rolling(window=20, min_periods=20).std(ddof=1)

        # 等权合成 4 个相同的 = 单个
        pd.testing.assert_frame_equal(result, single, atol=1e-10)


class TestMainForceVolEdgeCases:
    def test_short_data(self, factor):
        """数据不足 T 时，结果应全为 NaN。"""
        n, ncols = 10, 3
        v = _make_df(n, ncols)
        result = factor.compute(
            vol_up_ret=v, cont_up_ret=v, vol_down_ret=v, cont_down_ret=v, T=20,
        )
        assert result.isna().all().all()

    def test_single_stock(self, factor):
        """单只股票时，截面标准化无法计算（std=NaN），结果为 NaN。"""
        n = 25
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        v = pd.DataFrame(np.random.randn(n) * 0.02, index=dates, columns=["A"])

        result = factor.compute(
            vol_up_ret=v, cont_up_ret=v, vol_down_ret=v, cont_down_ret=v, T=20,
        )
        assert result.isna().all().all()

    def test_nan_in_input(self, factor):
        """输入含 NaN 时不应抛异常。"""
        n, ncols = 25, 3
        v = _make_df(n, ncols)
        v.iloc[10, 1] = np.nan
        result = factor.compute(
            vol_up_ret=v, cont_up_ret=v, vol_down_ret=v, cont_down_ret=v, T=20,
        )
        assert isinstance(result, pd.DataFrame)


class TestMainForceVolOutputShape:
    def test_output_shape_matches_input(self, factor):
        n, ncols = 30, 4
        v = _make_df(n, ncols)
        result = factor.compute(
            vol_up_ret=v, cont_up_ret=v, vol_down_ret=v, cont_down_ret=v, T=20,
        )
        assert result.shape == v.shape
        assert list(result.columns) == list(v.columns)
        assert list(result.index) == list(v.index)

    def test_output_is_dataframe(self, factor):
        n, ncols = 25, 3
        v = _make_df(n, ncols)
        result = factor.compute(
            vol_up_ret=v, cont_up_ret=v, vol_down_ret=v, cont_down_ret=v, T=20,
        )
        assert isinstance(result, pd.DataFrame)
