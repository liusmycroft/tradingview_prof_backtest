import numpy as np
import pandas as pd
import pytest

import importlib
import importlib.util
import pathlib
import sys
import types

_factors_dir = pathlib.Path(__file__).resolve().parent.parent / "factors"

_pkg = types.ModuleType("factors")
_pkg.__path__ = [str(_factors_dir)]
sys.modules["factors"] = _pkg

_base_spec = importlib.util.spec_from_file_location("factors.base", _factors_dir / "base.py")
_base_mod = importlib.util.module_from_spec(_base_spec)
sys.modules["factors.base"] = _base_mod
_base_spec.loader.exec_module(_base_mod)

_mod_spec = importlib.util.spec_from_file_location(
    "factors.jump_degree_centrality", _factors_dir / "jump_degree_centrality.py"
)
_mod = importlib.util.module_from_spec(_mod_spec)
sys.modules["factors.jump_degree_centrality"] = _mod
_mod_spec.loader.exec_module(_mod)

JumpDegreeCentralityFactor = _mod.JumpDegreeCentralityFactor


@pytest.fixture
def factor():
    return JumpDegreeCentralityFactor()


class TestJumpDegreeCentralityMetadata:
    def test_name(self, factor):
        assert factor.name == "JUMP_DEGREE_CENTRALITY"

    def test_category(self, factor):
        assert factor.category == "图谱网络-动量溢出"

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "JUMP_DEGREE_CENTRALITY"
        assert meta["category"] == "图谱网络-动量溢出"

    def test_repr(self, factor):
        assert "JUMP_DEGREE_CENTRALITY" in repr(factor)


class TestJumpDegreeCentralityCompute:
    def test_constant_input(self, factor):
        """常数输入时, EMA 应等于该常数。"""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        data = pd.DataFrame(0.5, index=dates, columns=["A"])

        result = factor.compute(daily_degree_centrality=data, T=20)
        np.testing.assert_allclose(result["A"].values, 0.5, atol=1e-10)

    def test_ema_manual_T3(self, factor):
        """T=3, 手动验证 EMA 值 (ewm span=3, adjust=True)。

        alpha = 2/(3+1) = 0.5
        data = [10, 20, 30, 40]
          ema_0 = 10.0
          ema_1 = (0.5*10 + 1.0*20) / (0.5+1.0) = 16.6667
          ema_2 = (0.25*10 + 0.5*20 + 1.0*30) / (0.25+0.5+1.0) = 24.2857
          ema_3 = (0.125*10 + 0.25*20 + 0.5*30 + 1.0*40) / (0.125+0.25+0.5+1.0) = 32.6667
        """
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        data = pd.DataFrame([10.0, 20.0, 30.0, 40.0], index=dates, columns=["A"])

        result = factor.compute(daily_degree_centrality=data, T=3)

        assert result.iloc[0, 0] == pytest.approx(10.0, rel=1e-6)
        assert result.iloc[1, 0] == pytest.approx(50 / 3, rel=1e-6)
        assert result.iloc[2, 0] == pytest.approx(170 / 7, rel=1e-6)
        assert result.iloc[3, 0] == pytest.approx(490 / 15, rel=1e-6)

    def test_ema_recent_weight(self, factor):
        """EMA 应赋予近期数据更高权重。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        vals = [0.1] * 5 + [0.9] * 5
        data = pd.DataFrame(vals, index=dates, columns=["A"])

        result = factor.compute(daily_degree_centrality=data, T=5)
        assert result.iloc[-1, 0] > 0.5

    def test_two_stocks_independent(self, factor):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame({"A": [0.3] * 10, "B": [0.7] * 10}, index=dates)

        result = factor.compute(daily_degree_centrality=data, T=5)
        np.testing.assert_allclose(result["A"].values, 0.3, atol=1e-10)
        np.testing.assert_allclose(result["B"].values, 0.7, atol=1e-10)


class TestJumpDegreeCentralityEdgeCases:
    def test_single_value(self, factor):
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        data = pd.DataFrame([0.42], index=dates, columns=["A"])

        result = factor.compute(daily_degree_centrality=data, T=20)
        assert result.iloc[0, 0] == pytest.approx(0.42, rel=1e-10)

    def test_nan_in_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame([0.1, np.nan, 0.3, 0.4, 0.5], index=dates, columns=["A"])

        result = factor.compute(daily_degree_centrality=data, T=3)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (5, 1)

    def test_all_nan(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame(np.nan, index=dates, columns=["A"])

        result = factor.compute(daily_degree_centrality=data, T=3)
        assert result.isna().all().all()

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        data = pd.DataFrame(np.random.uniform(0, 1, (30, 3)), index=dates, columns=stocks)

        result = factor.compute(daily_degree_centrality=data, T=20)
        assert result.shape == data.shape
        assert isinstance(result, pd.DataFrame)

    def test_no_leading_nan(self, factor):
        """min_periods=1, 第一行就有值。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(np.ones(10) * 0.5, index=dates, columns=["A"])

        result = factor.compute(daily_degree_centrality=data, T=20)
        assert result.iloc[0].notna().all()
