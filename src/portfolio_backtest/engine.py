"""Portfolio simulation engine.

Replays bars and trades chronologically to build a daily equity curve.
"""

from dataclasses import dataclass, replace
from datetime import datetime

import pandas as pd

from portfolio_backtest.models import Action


@dataclass(frozen=True)
class Position:
    ticker: str
    quantity: float = 0.0  # positive = long, negative = short
    cost_basis: float = 0.0  # average entry price


@dataclass(frozen=True)
class EngineConfig:
    initial_cash: float = 100_000.0
    slippage_pct: float = 0.0
    commission_pct: float = 0.0
    commission_min: float = 0.0
    commission_per_share: float = 0.0
    short_enabled: bool = True
    max_position_pct: float = 1.0  # 0..1, fraction of portfolio


@dataclass
class TradeRecord:
    """A completed round-trip trade for win/loss stats."""
    ticker: str
    entry_price: float
    exit_price: float
    quantity: float
    side: str  # "long" or "short"
    pnl: float


class Engine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.cash: float = config.initial_cash
        self.positions: dict[str, Position] = {}
        self.equity_series: list[tuple[datetime, float]] = []
        self.latest_prices: dict[str, float] = {}
        self.completed_trades: list[TradeRecord] = []

    def _calc_commission(self, quantity: float, price: float) -> float:
        pct_comm = abs(quantity) * price * self.config.commission_pct / 100
        per_share_comm = abs(quantity) * self.config.commission_per_share
        return max(pct_comm + per_share_comm, self.config.commission_min)

    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        factor = 1 + self.config.slippage_pct / 100 if is_buy else 1 - self.config.slippage_pct / 100
        return price * factor

    def _portfolio_value(self) -> float:
        value = self.cash
        for ticker, pos in self.positions.items():
            price = self.latest_prices.get(ticker, 0.0)
            value += pos.quantity * price
        return value

    def process_bar(self, ticker: str, close: float, timestamp: datetime):
        self.latest_prices[ticker] = close
        self.equity_series.append((timestamp, self._portfolio_value()))

    def process_order(
        self, ticker: str, action: str, quantity: float, price: float, timestamp: datetime
    ):
        action_enum = Action(action.lower())
        is_buy = action_enum in (Action.BUY, Action.COVER)
        actual_price = self._apply_slippage(price, is_buy)

        # enforce max_position_pct for new buys / shorts
        if action_enum in (Action.BUY, Action.SHORT) and self.config.max_position_pct < 1.0:
            portfolio_value = self._portfolio_value()
            max_notional = portfolio_value * self.config.max_position_pct
            current_notional = abs(self.positions.get(ticker, Position(ticker=ticker)).quantity) * actual_price
            allowed = max(0.0, max_notional - current_notional)
            max_qty = allowed / actual_price if actual_price > 0 else 0.0
            if quantity > max_qty:
                quantity = max_qty
            if quantity <= 0:
                return

        commission = self._calc_commission(quantity, actual_price)
        pos = self.positions.get(ticker, Position(ticker=ticker))

        if action_enum == Action.BUY:
            pos = self._apply_buy(pos, quantity, actual_price, commission)
        elif action_enum == Action.SELL:
            pos = self._apply_sell(pos, ticker, quantity, actual_price, commission)
        elif action_enum == Action.SHORT:
            if not self.config.short_enabled:
                return
            pos = self._apply_short(pos, quantity, actual_price, commission)
        elif action_enum == Action.COVER:
            pos = self._apply_cover(pos, ticker, quantity, actual_price, commission)

        self.positions[ticker] = pos
        self.latest_prices[ticker] = actual_price
        self.equity_series.append((timestamp, self._portfolio_value()))

    def _apply_buy(self, pos: Position, qty: float, price: float, comm: float) -> Position:
        self.cash -= price * qty + comm
        old_qty = pos.quantity
        new_qty = old_qty + qty
        new_basis = (
            (pos.cost_basis * old_qty + price * qty) / new_qty
            if new_qty != 0 else 0.0
        )
        return replace(pos, quantity=new_qty, cost_basis=new_basis)

    def _apply_sell(self, pos: Position, ticker: str, qty: float, price: float, comm: float) -> Position:
        self.cash += price * qty - comm
        if pos.quantity > 0:
            pnl = (price - pos.cost_basis) * qty - comm
            self.completed_trades.append(
                TradeRecord(ticker=ticker, entry_price=pos.cost_basis,
                            exit_price=price, quantity=qty, side="long", pnl=pnl)
            )
        new_qty = pos.quantity - qty
        if abs(new_qty) < 1e-9:
            return replace(pos, quantity=0.0, cost_basis=0.0)
        return replace(pos, quantity=new_qty)

    def _apply_short(self, pos: Position, qty: float, price: float, comm: float) -> Position:
        self.cash += price * qty - comm
        old_qty = pos.quantity
        if old_qty >= 0:
            new_basis = price
        else:
            new_basis = (pos.cost_basis * abs(old_qty) + price * qty) / (abs(old_qty) + qty)
        return replace(pos, quantity=old_qty - qty, cost_basis=new_basis)

    def _apply_cover(self, pos: Position, ticker: str, qty: float, price: float, comm: float) -> Position:
        self.cash -= price * qty + comm
        if pos.quantity < 0:
            pnl = (pos.cost_basis - price) * qty - comm
            self.completed_trades.append(
                TradeRecord(ticker=ticker, entry_price=pos.cost_basis,
                            exit_price=price, quantity=qty, side="short", pnl=pnl)
            )
        new_qty = pos.quantity + qty
        if abs(new_qty) < 1e-9:
            return replace(pos, quantity=0.0, cost_basis=0.0)
        return replace(pos, quantity=new_qty)

    def force_close_all(self, timestamp: datetime):
        for ticker, pos in list(self.positions.items()):
            if pos.quantity > 0:
                price = self.latest_prices.get(ticker, 0.0)
                self.process_order(ticker, "sell", pos.quantity, price, timestamp)
            elif pos.quantity < 0:
                price = self.latest_prices.get(ticker, 0.0)
                self.process_order(ticker, "cover", abs(pos.quantity), price, timestamp)

    def get_equity_df(self) -> pd.DataFrame:
        if not self.equity_series:
            return pd.DataFrame(columns=["timestamp", "equity"])
        df = pd.DataFrame(self.equity_series, columns=["timestamp", "equity"])
        df = df.groupby("timestamp", as_index=False).last()
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def get_returns_series(self) -> pd.Series:
        df = self.get_equity_df()
        if df.empty:
            return pd.Series(dtype=float)
        df = df.set_index("timestamp")
        returns = df["equity"].pct_change().dropna()
        returns.index = pd.to_datetime(returns.index)
        return returns

    def get_metrics(self) -> dict:
        df = self.get_equity_df()
        if df.empty:
            return self._empty_metrics()

        initial = self.config.initial_cash
        final = df["equity"].iloc[-1]
        total_return = final - initial
        total_return_pct = (total_return / initial) * 100
        returns = self.get_returns_series()

        annualized = self._calc_annualized(initial, final, df)
        sharpe = self._calc_sharpe(returns)
        sortino = self._calc_sortino(returns)
        max_dd, dd_duration = self._calc_drawdown(df["equity"])
        trade_stats = self._calc_trade_stats()

        return {
            "total_return": round(total_return, 2),
            "total_return_pct": round(total_return_pct, 2),
            "annualized_return_pct": round(annualized, 2) if annualized is not None else None,
            "sharpe_ratio": round(sharpe, 4) if sharpe is not None else None,
            "sortino_ratio": round(sortino, 4) if sortino is not None else None,
            "max_drawdown_pct": round(max_dd, 2),
            "max_drawdown_duration_days": dd_duration,
            **trade_stats,
        }

    def _calc_annualized(self, initial: float, final: float, df: pd.DataFrame) -> float | None:
        n_days = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days
        if n_days <= 0:
            return None
        return ((final / initial) ** (365 / n_days) - 1) * 100

    def _calc_sharpe(self, returns: pd.Series) -> float | None:
        if len(returns) > 1 and returns.std() > 0:
            return float(returns.mean() / returns.std() * (252 ** 0.5))
        return None

    def _calc_sortino(self, returns: pd.Series) -> float | None:
        downside = returns[returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            return float(returns.mean() / downside.std() * (252 ** 0.5))
        return None

    def _calc_drawdown(self, equity: pd.Series) -> tuple[float, int | None]:
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_dd = float(drawdown.min() * 100)
        dd_duration = None
        in_dd = drawdown < 0
        if in_dd.any():
            groups = (~in_dd).cumsum()
            dd_lengths = in_dd.groupby(groups).sum()
            dd_duration = int(dd_lengths.max()) if not dd_lengths.empty else None
        return max_dd, dd_duration

    def _calc_trade_stats(self) -> dict:
        wins = [t for t in self.completed_trades if t.pnl > 0]
        losses = [t for t in self.completed_trades if t.pnl <= 0]
        total = len(self.completed_trades)
        win_rate = len(wins) / total * 100 if total > 0 else None
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
        return {
            "win_rate": round(win_rate, 2) if win_rate is not None else None,
            "profit_factor": round(profit_factor, 4) if profit_factor is not None else None,
            "total_trades": total,
            "winning_trades": len(wins),
            "losing_trades": len(losses),
        }

    def _empty_metrics(self) -> dict:
        return {
            "total_return": 0.0,
            "total_return_pct": 0.0,
            "annualized_return_pct": None,
            "sharpe_ratio": None,
            "sortino_ratio": None,
            "max_drawdown_pct": 0.0,
            "max_drawdown_duration_days": None,
            "win_rate": None,
            "profit_factor": None,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
        }


def build_engine_from_db(backtest) -> Engine:
    """Build an Engine and replay all bars/trades from a Backtest ORM object."""
    config = EngineConfig(
        initial_cash=backtest.initial_cash,
        slippage_pct=backtest.slippage_pct,
        commission_pct=backtest.commission_pct,
        commission_min=backtest.commission_min,
        commission_per_share=backtest.commission_per_share,
        short_enabled=backtest.short_enabled,
        max_position_pct=backtest.max_position_pct,
    )
    engine = Engine(config)

    events: list[tuple[datetime, str, object]] = []
    for bar in backtest.bars:
        events.append((bar.timestamp, "bar", bar))
    for trade in backtest.trades:
        events.append((trade.timestamp, "order", trade))

    # stable sort: bars before orders at same timestamp
    events.sort(key=lambda e: (e[0], 0 if e[1] == "bar" else 1))

    for ts, kind, obj in events:
        if kind == "bar":
            engine.process_bar(obj.ticker, obj.close, obj.timestamp)
        else:
            engine.process_order(
                obj.ticker, obj.action.value, obj.quantity, obj.price, obj.timestamp
            )

    return engine
