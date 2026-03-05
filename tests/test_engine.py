"""Tests for the portfolio simulation engine."""

from datetime import datetime

from portfolio_backtest.engine import Engine, EngineConfig


def make_ts(day: int) -> datetime:
    return datetime(2024, 1, day)


class TestEngineBasicLongTrade:
    """Buy low, sell high — verify PnL and equity."""

    def setup_method(self):
        self.engine = Engine(EngineConfig(initial_cash=10000))

    def test_buy_and_sell(self):
        self.engine.process_bar("AAPL", 100.0, make_ts(1))
        self.engine.process_order("AAPL", "buy", 10, 100.0, make_ts(1))
        self.engine.process_bar("AAPL", 110.0, make_ts(2))
        self.engine.process_order("AAPL", "sell", 10, 110.0, make_ts(2))
        self.engine.process_bar("AAPL", 112.0, make_ts(3))

        metrics = self.engine.get_metrics()
        assert metrics["total_return"] == 100.0
        assert metrics["total_return_pct"] == 1.0
        assert metrics["total_trades"] == 1
        assert metrics["winning_trades"] == 1
        assert metrics["losing_trades"] == 0

    def test_equity_after_sell(self):
        """After selling, equity should not change with price."""
        self.engine.process_bar("AAPL", 100.0, make_ts(1))
        self.engine.process_order("AAPL", "buy", 10, 100.0, make_ts(1))
        self.engine.process_bar("AAPL", 110.0, make_ts(2))
        self.engine.process_order("AAPL", "sell", 10, 110.0, make_ts(2))
        self.engine.process_bar("AAPL", 200.0, make_ts(3))

        df = self.engine.get_equity_df()
        # last equity should be 10100 (cash only, no position)
        assert df["equity"].iloc[-1] == 10100.0


class TestEngineShortTrade:
    """Short sell — verify PnL for short positions."""

    def setup_method(self):
        self.engine = Engine(EngineConfig(initial_cash=10000))

    def test_short_and_cover_profit(self):
        self.engine.process_bar("QQQ", 400.0, make_ts(1))
        self.engine.process_order("QQQ", "short", 5, 400.0, make_ts(1))
        self.engine.process_bar("QQQ", 380.0, make_ts(2))
        self.engine.process_order("QQQ", "cover", 5, 380.0, make_ts(2))

        metrics = self.engine.get_metrics()
        # profit = (400 - 380) * 5 = 100
        assert metrics["total_return"] == 100.0
        assert metrics["winning_trades"] == 1

    def test_short_and_cover_loss(self):
        self.engine.process_bar("QQQ", 400.0, make_ts(1))
        self.engine.process_order("QQQ", "short", 5, 400.0, make_ts(1))
        self.engine.process_bar("QQQ", 420.0, make_ts(2))
        self.engine.process_order("QQQ", "cover", 5, 420.0, make_ts(2))

        metrics = self.engine.get_metrics()
        # loss = (400 - 420) * 5 = -100
        assert metrics["total_return"] == -100.0
        assert metrics["losing_trades"] == 1


class TestEngineCommission:
    """Verify commission and slippage are applied."""

    def test_commission_pct(self):
        engine = Engine(EngineConfig(initial_cash=10000, commission_pct=0.1))
        engine.process_bar("AAPL", 100.0, make_ts(1))
        engine.process_order("AAPL", "buy", 10, 100.0, make_ts(1))
        # commission = 10 * 100 * 0.1% = 1.0
        # cash = 10000 - 1000 - 1 = 8999
        assert engine.cash == 8999.0

    def test_commission_min(self):
        engine = Engine(EngineConfig(initial_cash=10000, commission_min=5.0))
        engine.process_bar("AAPL", 100.0, make_ts(1))
        engine.process_order("AAPL", "buy", 1, 100.0, make_ts(1))
        # commission = max(0, 5.0) = 5.0
        # cash = 10000 - 100 - 5 = 9895
        assert engine.cash == 9895.0

    def test_slippage(self):
        engine = Engine(EngineConfig(initial_cash=10000, slippage_pct=1.0))
        engine.process_bar("AAPL", 100.0, make_ts(1))
        engine.process_order("AAPL", "buy", 10, 100.0, make_ts(1))
        # slippage: price = 100 * 1.01 = 101
        # cash = 10000 - 101 * 10 = 8990
        assert engine.cash == 8990.0


class TestEngineMultiTicker:
    """Multiple tickers in one portfolio."""

    def test_two_tickers(self):
        engine = Engine(EngineConfig(initial_cash=20000))
        engine.process_bar("AAPL", 100.0, make_ts(1))
        engine.process_bar("NVDA", 200.0, make_ts(1))
        engine.process_order("AAPL", "buy", 10, 100.0, make_ts(1))
        engine.process_order("NVDA", "buy", 5, 200.0, make_ts(1))

        engine.process_bar("AAPL", 110.0, make_ts(2))
        engine.process_bar("NVDA", 220.0, make_ts(2))

        # cash = 20000 - 1000 - 1000 = 18000
        # AAPL value = 10 * 110 = 1100
        # NVDA value = 5 * 220 = 1100
        # total = 18000 + 1100 + 1100 = 20200
        df = engine.get_equity_df()
        last_equity = df["equity"].iloc[-1]
        assert last_equity == 20200.0


class TestEngineForceClose:
    """Force close all positions."""

    def test_force_close_long(self):
        engine = Engine(EngineConfig(initial_cash=10000))
        engine.process_bar("AAPL", 100.0, make_ts(1))
        engine.process_order("AAPL", "buy", 10, 100.0, make_ts(1))
        engine.process_bar("AAPL", 110.0, make_ts(2))

        engine.force_close_all(make_ts(2))
        assert engine.positions["AAPL"].quantity == 0.0
        assert engine.cash == 10100.0

    def test_force_close_short(self):
        engine = Engine(EngineConfig(initial_cash=10000))
        engine.process_bar("QQQ", 400.0, make_ts(1))
        engine.process_order("QQQ", "short", 5, 400.0, make_ts(1))
        engine.process_bar("QQQ", 380.0, make_ts(2))

        engine.force_close_all(make_ts(2))
        assert engine.positions["QQQ"].quantity == 0.0
        # cash = 10000 + 2000 (short proceeds) - 1900 (cover cost) = 10100
        assert engine.cash == 10100.0


class TestEngineReturns:
    """Verify returns series calculation."""

    def test_returns_length(self):
        engine = Engine(EngineConfig(initial_cash=10000))
        engine.process_bar("AAPL", 100.0, make_ts(1))
        engine.process_bar("AAPL", 105.0, make_ts(2))
        engine.process_bar("AAPL", 103.0, make_ts(3))

        returns = engine.get_returns_series()
        # 3 bars → 2 returns (first is dropped by pct_change)
        assert len(returns) == 2

    def test_returns_no_position_are_zero(self):
        engine = Engine(EngineConfig(initial_cash=10000))
        engine.process_bar("AAPL", 100.0, make_ts(1))
        engine.process_bar("AAPL", 200.0, make_ts(2))

        returns = engine.get_returns_series()
        # no position, equity doesn't change
        assert returns.iloc[0] == 0.0
