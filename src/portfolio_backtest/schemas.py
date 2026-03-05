from datetime import datetime
from typing import Literal

from pydantic import BaseModel


# --- Webhook payloads ---


class BarPayload(BaseModel):
    type: Literal["bar"]
    ticker: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    timestamp: str  # ISO 8601


class OrderPayload(BaseModel):
    type: Literal["order"]
    strategy: str
    ticker: str
    action: str  # buy | sell | short | cover
    quantity: float
    price: float
    timestamp: str  # ISO 8601


# --- API request/response ---


class BacktestCreate(BaseModel):
    name: str
    initial_cash: float = 100_000.0
    slippage_pct: float = 0.0
    commission_pct: float = 0.0
    commission_min: float = 0.0
    commission_per_share: float = 0.0
    short_enabled: bool = True
    short_margin_rate: float = 1.5
    short_borrow_rate_annual: float = 0.0
    max_position_pct: float = 1.0
    benchmark_ticker: str | None = None


class BacktestFinish(BaseModel):
    force_close_positions: bool = True


class BacktestOut(BaseModel):
    id: str
    name: str
    status: str
    initial_cash: float
    slippage_pct: float
    commission_pct: float
    commission_min: float
    commission_per_share: float
    short_enabled: bool
    short_margin_rate: float
    short_borrow_rate_annual: float
    max_position_pct: float
    benchmark_ticker: str | None
    created_at: datetime
    bar_count: int = 0
    trade_count: int = 0
    tickers: list[str] = []
    strategies: list[str] = []

    model_config = {"from_attributes": True}


class TradeOut(BaseModel):
    id: int
    strategy: str
    ticker: str
    action: str
    quantity: float
    price: float
    timestamp: datetime

    model_config = {"from_attributes": True}


class MetricsOut(BaseModel):
    total_return: float
    total_return_pct: float
    annualized_return_pct: float | None
    sharpe_ratio: float | None
    sortino_ratio: float | None
    max_drawdown_pct: float
    max_drawdown_duration_days: int | None
    win_rate: float | None
    profit_factor: float | None
    total_trades: int
    winning_trades: int
    losing_trades: int
