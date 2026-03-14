import datetime
import enum
import uuid

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from portfolio_backtest.database import Base


def _uuid() -> str:
    return uuid.uuid4().hex[:12]


class BacktestStatus(str, enum.Enum):
    RUNNING = "running"
    FINISHED = "finished"


class Action(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"
    SHORT = "short"
    COVER = "cover"


class Backtest(Base):
    __tablename__ = "backtests"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[BacktestStatus] = mapped_column(
        Enum(BacktestStatus), default=BacktestStatus.RUNNING
    )
    initial_cash: Mapped[float] = mapped_column(Float, default=100_000.0)

    # cost config
    slippage_pct: Mapped[float] = mapped_column(Float, default=0.0)
    commission_pct: Mapped[float] = mapped_column(Float, default=0.0)
    commission_min: Mapped[float] = mapped_column(Float, default=0.0)
    commission_per_share: Mapped[float] = mapped_column(Float, default=0.0)

    # short config
    short_enabled: Mapped[bool] = mapped_column(default=True)
    short_margin_rate: Mapped[float] = mapped_column(Float, default=1.5)
    short_borrow_rate_annual: Mapped[float] = mapped_column(Float, default=0.0)

    # risk
    max_position_pct: Mapped[float] = mapped_column(Float, default=1.0)

    # benchmark
    benchmark_ticker: Mapped[str | None] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=lambda: datetime.datetime.now(datetime.UTC)
    )

    trades: Mapped[list["Trade"]] = relationship(back_populates="backtest", cascade="all, delete-orphan")
    bars: Mapped[list["Bar"]] = relationship(back_populates="backtest", cascade="all, delete-orphan")


class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    backtest_id: Mapped[str] = mapped_column(ForeignKey("backtests.id"))
    strategy: Mapped[str] = mapped_column(String, nullable=False)
    ticker: Mapped[str] = mapped_column(String, nullable=False)
    action: Mapped[Action] = mapped_column(Enum(Action), nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime, nullable=False)

    backtest: Mapped["Backtest"] = relationship(back_populates="trades")


class Bar(Base):
    __tablename__ = "bars"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    backtest_id: Mapped[str] = mapped_column(ForeignKey("backtests.id"))
    ticker: Mapped[str] = mapped_column(String, nullable=False)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, default=0.0)
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime, nullable=False)

    backtest: Mapped["Backtest"] = relationship(back_populates="bars")
