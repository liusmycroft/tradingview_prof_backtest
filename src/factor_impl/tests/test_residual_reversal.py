"""Tests for ResidualReversalFactor."""

import importlib
import importlib.util
import sys

import numpy as np
import pandas as pd
import pytest

# Direct import from the module file to avoid factors/__init__.py
# which may reference sibling modules that don't exist yet.
_base_spec = importlib.util.spec_from_file_location(
    "factors.base",
    "/home/lius/Desktop/code/quant/factor_impl/factors/base.py",
)
_base_mod = importlib.util.module_from_spec(_base_spec)
sys.modules["factors.base"] = _base_mod
_base_spec.loader.exec_module(_base_mod)

_spec = importlib.util.spec_from_file_location(
    "factors.residual_reversal",
    "/home/lius/Desktop/code/quant/factor_impl/factors/residual_reversal.py",
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["factors.residual_reversal"] = _mod
_spec.loader.exec_module(_mod)

ResidualReversalFactor = _mod.ResidualReversalFactor


@pytest.fixture
def factor():
    return ResidualReversalFactor()


def _make_dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-01", periods=n)


class TestMetadata:
    def test_name_and_category(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "residual_reversal"
        assert meta["category"] == "reversal"

    def test_repr(self, factor):
        assert "ResidualReversalFactor" in repr(factor)


class TestOutputShape:
    """Output should match input dimensions and types."""

    def test_shape_matches_input(self, factor):
        dates = _make_dates(40)
        stocks = ["A", "B", "C", "D", "E"]
        rng = np.random.default_rng(42)

        close = pd.DataFrame(
            100 + rng.standard_normal((40, 5)).cumsum(axis=0),
            index=dates,
            columns=stocks,
        )
        buy = pd.DataFrame(rng.uniform(1, 10, (40, 5)), index=dates, columns=stocks)
        sell = pd.DataFrame(rng.uniform(1, 10, (40, 5)), index=dates, columns=stocks)

        result = factor.compute(close=close, buy_amount=buy, sell_amount=sell, T=20)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == close.shape
        assert list(result.columns) == stocks
        assert (result.index == dates).all()

    def test_first_T_rows_are_nan(self, factor):
        """The first T rows lack enough history, so they should be NaN."""
        dates = _make_dates(40)
        stocks = ["A", "B", "C"]
        rng = np.random.default_rng(0)

        close = pd.DataFrame(
            100 + rng.standard_normal((40, 3)).cumsum(axis=0),
            index=dates,
            columns=stocks,
        )
        buy = pd.DataFrame(rng.uniform(1, 10, (40, 3)), index=dates, columns=stocks)
        sell = pd.DataFrame(rng.uniform(1, 10, (40, 3)), index=dates, columns=stocks)

        result = factor.compute(close=close, buy_amount=buy, sell_amount=sell, T=20)
        # First 20 rows should be all NaN (not enough lookback)
        assert result.iloc[:20].isna().all().all()


class TestKnownRegression:
    """Construct data where the regression outcome is deterministic."""

    def test_residual_is_zero_when_ret_is_linear_in_strength(self, factor):
        """If Ret20 = a + b*S exactly, residuals should be ~0."""
        T = 20
        n_days = T + 5
        stocks = ["S1", "S2", "S3", "S4", "S5"]
        dates = _make_dates(n_days)
        rng = np.random.default_rng(99)

        # Use random buy/sell to get varying S across stocks
        buy = pd.DataFrame(rng.uniform(1, 10, (n_days, 5)), index=dates, columns=stocks)
        sell = pd.DataFrame(rng.uniform(1, 10, (n_days, 5)), index=dates, columns=stocks)

        # Compute expected S
        net = buy - sell
        rolling_net = net.rolling(T, min_periods=T).sum()
        rolling_abs = net.abs().rolling(T, min_periods=T).sum()
        S = rolling_net / rolling_abs

        # Build close prices so that Ret20 = 0.5 + 2.0 * S exactly
        # Ret20_t = close_t / close_{t-T} - 1  =>  close_t = close_{t-T} * (1 + Ret20_t)
        close = pd.DataFrame(np.nan, index=dates, columns=stocks)
        close.iloc[0] = 100.0
        # Fill first T rows with arbitrary prices
        for i in range(1, T):
            close.iloc[i] = close.iloc[i - 1] * (1 + rng.uniform(-0.01, 0.01, 5))

        # From row T onward, set close so Ret20 = 0.5 + 2*S
        for i in range(T, n_days):
            s_row = S.iloc[i]
            ret_target = 0.5 + 2.0 * s_row
            close.iloc[i] = close.iloc[i - T] * (1 + ret_target)

        result = factor.compute(close=close, buy_amount=buy, sell_amount=sell, T=T)

        # Residuals for rows T onward should be near zero
        valid = result.iloc[T:]
        assert valid.notna().any().any(), "Should have non-NaN values"
        np.testing.assert_allclose(
            valid.dropna(how="all").values,
            0.0,
            atol=1e-8,
        )

    def test_residual_captures_orthogonal_component(self, factor):
        """If Ret20 = a + b*S + known_epsilon, residuals should equal epsilon."""
        T = 20
        n_days = T + 10
        stocks = [f"S{i}" for i in range(20)]  # need enough stocks for stable OLS
        dates = _make_dates(n_days)
        rng = np.random.default_rng(7)

        buy = pd.DataFrame(
            rng.uniform(1, 10, (n_days, len(stocks))), index=dates, columns=stocks
        )
        sell = pd.DataFrame(
            rng.uniform(1, 10, (n_days, len(stocks))), index=dates, columns=stocks
        )

        net = buy - sell
        rolling_net = net.rolling(T, min_periods=T).sum()
        rolling_abs = net.abs().rolling(T, min_periods=T).sum()
        S = rolling_net / rolling_abs

        # Construct epsilon that is orthogonal to S in each cross-section
        # For each date, generate random eps, then orthogonalize against [1, S]
        epsilon = pd.DataFrame(0.0, index=dates, columns=stocks)
        for i in range(T, n_days):
            s = S.iloc[i].values
            mask = np.isfinite(s)
            e = rng.standard_normal(mask.sum())
            sv = s[mask]
            # Orthogonalize: remove projection onto [1, S]
            X = np.column_stack([np.ones(len(sv)), sv])
            proj = X @ np.linalg.lstsq(X, e, rcond=None)[0]
            e_orth = e - proj
            epsilon.iloc[i, np.where(mask)[0]] = e_orth

        # Build close so Ret20 = 1.0 + 3.0*S + epsilon
        close = pd.DataFrame(np.nan, index=dates, columns=stocks)
        close.iloc[0] = 100.0
        for i in range(1, T):
            close.iloc[i] = close.iloc[i - 1] * (1 + rng.uniform(-0.01, 0.01, len(stocks)))
        for i in range(T, n_days):
            ret_target = 1.0 + 3.0 * S.iloc[i] + epsilon.iloc[i]
            close.iloc[i] = close.iloc[i - T] * (1 + ret_target)

        result = factor.compute(close=close, buy_amount=buy, sell_amount=sell, T=T)

        for i in range(T, n_days):
            res_row = result.iloc[i].dropna().values
            eps_row = epsilon.iloc[i].dropna().values
            if len(res_row) == 0:
                continue
            np.testing.assert_allclose(res_row, eps_row, atol=1e-6)


class TestEdgeCases:
    def test_constant_fund_flow(self, factor):
        """When buy == sell everywhere, S is 0/0 = NaN. Factor should handle gracefully."""
        dates = _make_dates(40)
        stocks = ["A", "B", "C"]
        rng = np.random.default_rng(1)

        close = pd.DataFrame(
            100 + rng.standard_normal((40, 3)).cumsum(axis=0),
            index=dates,
            columns=stocks,
        )
        constant = pd.DataFrame(5.0, index=dates, columns=stocks)

        result = factor.compute(
            close=close, buy_amount=constant, sell_amount=constant, T=20
        )
        assert isinstance(result, pd.DataFrame)
        # S is NaN everywhere (0/0), so all residuals should be NaN
        assert result.isna().all().all()

    def test_nan_in_close(self, factor):
        """NaN in close prices should propagate gracefully, not raise."""
        dates = _make_dates(40)
        stocks = ["A", "B", "C", "D"]
        rng = np.random.default_rng(3)

        close = pd.DataFrame(
            100 + rng.standard_normal((40, 4)).cumsum(axis=0),
            index=dates,
            columns=stocks,
        )
        close.iloc[25, 1] = np.nan  # inject NaN

        buy = pd.DataFrame(rng.uniform(1, 10, (40, 4)), index=dates, columns=stocks)
        sell = pd.DataFrame(rng.uniform(1, 10, (40, 4)), index=dates, columns=stocks)

        result = factor.compute(close=close, buy_amount=buy, sell_amount=sell, T=20)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == close.shape

    def test_nan_in_buy_sell(self, factor):
        """NaN in buy/sell should not crash."""
        dates = _make_dates(40)
        stocks = ["X", "Y", "Z"]
        rng = np.random.default_rng(5)

        close = pd.DataFrame(
            100 + rng.standard_normal((40, 3)).cumsum(axis=0),
            index=dates,
            columns=stocks,
        )
        buy = pd.DataFrame(rng.uniform(1, 10, (40, 3)), index=dates, columns=stocks)
        sell = pd.DataFrame(rng.uniform(1, 10, (40, 3)), index=dates, columns=stocks)
        buy.iloc[22, 0] = np.nan
        sell.iloc[23, 2] = np.nan

        result = factor.compute(close=close, buy_amount=buy, sell_amount=sell, T=20)
        assert isinstance(result, pd.DataFrame)

    def test_fewer_than_3_valid_stocks(self, factor):
        """With fewer than 3 valid stocks on a date, regression is skipped."""
        dates = _make_dates(25)
        stocks = ["A", "B"]
        rng = np.random.default_rng(10)

        close = pd.DataFrame(
            100 + rng.standard_normal((25, 2)).cumsum(axis=0),
            index=dates,
            columns=stocks,
        )
        buy = pd.DataFrame(rng.uniform(1, 10, (25, 2)), index=dates, columns=stocks)
        sell = pd.DataFrame(rng.uniform(1, 10, (25, 2)), index=dates, columns=stocks)

        result = factor.compute(close=close, buy_amount=buy, sell_amount=sell, T=20)
        # Only 2 stocks => mask.sum() < 3 => all NaN
        assert result.iloc[20:].isna().all().all()

    def test_constant_strength_across_stocks(self, factor):
        """If S is the same for all stocks on a date, std(x)==0 branch is hit."""
        T = 20
        dates = _make_dates(T + 5)
        stocks = ["A", "B", "C", "D"]
        rng = np.random.default_rng(12)

        # Make buy - sell identical across stocks so S is the same for all
        base_net = rng.uniform(-2, 2, (T + 5, 1))
        net = np.tile(base_net, (1, 4))
        buy = pd.DataFrame(net + 10, index=dates, columns=stocks)
        sell = pd.DataFrame(np.full((T + 5, 4), 10.0), index=dates, columns=stocks)

        close = pd.DataFrame(
            100 + rng.standard_normal((T + 5, 4)).cumsum(axis=0),
            index=dates,
            columns=stocks,
        )

        result = factor.compute(close=close, buy_amount=buy, sell_amount=sell, T=T)

        # When S is constant across stocks, residual = y - mean(y), should sum to ~0
        for i in range(T, T + 5):
            row = result.iloc[i].dropna()
            if len(row) > 0:
                np.testing.assert_allclose(row.sum(), 0.0, atol=1e-10)
