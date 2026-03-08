import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_factors_dir = Path(__file__).resolve().parent.parent / "factors"

_pkg_spec = importlib.util.spec_from_file_location(
    "factors", _factors_dir / "__init__.py", submodule_search_locations=[str(_factors_dir)]
)
_pkg_mod = importlib.util.module_from_spec(_pkg_spec)
sys.modules["factors"] = _pkg_mod

_base_spec = importlib.util.spec_from_file_location("factors.base", _factors_dir / "base.py")
_base_mod = importlib.util.module_from_spec(_base_spec)
sys.modules["factors.base"] = _base_mod
_base_spec.loader.exec_module(_base_mod)

_mod_spec = importlib.util.spec_from_file_location(
    "factors.network_centrality", _factors_dir / "network_centrality.py"
)
_mod = importlib.util.module_from_spec(_mod_spec)
sys.modules["factors.network_centrality"] = _mod
_mod_spec.loader.exec_module(_mod)

NetworkCentralityFactor = _mod.NetworkCentralityFactor


@pytest.fixture
def factor():
    return NetworkCentralityFactor()


class TestNetworkCentralityMetadata:
    def test_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "NETWORK_CENTRALITY"
        assert meta["category"] == "网络结构"
        assert meta["description"] != ""

    def test_repr(self, factor):
        r = repr(factor)
        assert "NetworkCentralityFactor" in r


class TestNetworkCentralityCompute:
    def test_known_values(self, factor):
        """用已知数据验证等权组合。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        stocks = ["A", "B"]
        scc = pd.DataFrame([[0.8, 0.6], [0.4, 0.2], [1.0, 0.5]], index=dates, columns=stocks)
        tcc = pd.DataFrame([[0.2, 0.4], [0.6, 0.8], [0.0, 0.5]], index=dates, columns=stocks)

        result = factor.compute(scc=scc, tcc=tcc)

        # CC = 0.5 * SCC + 0.5 * TCC
        expected = pd.DataFrame(
            [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
            index=dates, columns=stocks,
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_equal_inputs(self, factor):
        """SCC == TCC 时，CC 应等于 SCC。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        df = pd.DataFrame({"A": [0.1, 0.2, 0.3, 0.4, 0.5]}, index=dates)

        result = factor.compute(scc=df, tcc=df)
        pd.testing.assert_frame_equal(result, df)

    def test_zero_tcc(self, factor):
        """TCC 全零时，CC = 0.5 * SCC。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        scc = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=dates)
        tcc = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=dates)

        result = factor.compute(scc=scc, tcc=tcc)
        expected = pd.DataFrame({"A": [0.5, 1.0, 1.5]}, index=dates)
        pd.testing.assert_frame_equal(result, expected)

    def test_nan_propagation(self, factor):
        """输入含 NaN 时，输出对应位置也应为 NaN。"""
        dates = pd.bdate_range("2025-01-01", periods=3)
        scc = pd.DataFrame({"A": [1.0, np.nan, 3.0]}, index=dates)
        tcc = pd.DataFrame({"A": [2.0, 4.0, np.nan]}, index=dates)

        result = factor.compute(scc=scc, tcc=tcc)
        assert pytest.approx(result.iloc[0, 0]) == 1.5
        assert pd.isna(result.iloc[1, 0])
        assert pd.isna(result.iloc[2, 0])

    def test_output_shape(self, factor):
        """输出形状应与输入一致。"""
        np.random.seed(42)
        dates = pd.bdate_range("2025-01-01", periods=30)
        stocks = ["A", "B", "C"]
        scc = pd.DataFrame(np.random.rand(30, 3), index=dates, columns=stocks)
        tcc = pd.DataFrame(np.random.rand(30, 3), index=dates, columns=stocks)

        result = factor.compute(scc=scc, tcc=tcc)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (30, 3)
        assert list(result.columns) == stocks

    def test_symmetry(self, factor):
        """交换 SCC 和 TCC 应得到相同结果（等权）。"""
        dates = pd.bdate_range("2025-01-01", periods=5)
        scc = pd.DataFrame({"A": np.random.rand(5)}, index=dates)
        tcc = pd.DataFrame({"A": np.random.rand(5)}, index=dates)

        result1 = factor.compute(scc=scc, tcc=tcc)
        result2 = factor.compute(scc=tcc, tcc=scc)
        pd.testing.assert_frame_equal(result1, result2)
