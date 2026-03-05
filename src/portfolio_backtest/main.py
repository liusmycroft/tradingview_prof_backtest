"""FastAPI application — webhook receiver + management API + static UI."""

from contextlib import asynccontextmanager
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func
from sqlalchemy.orm import Session

from portfolio_backtest.database import get_db, init_db
from portfolio_backtest.engine import build_engine_from_db
from portfolio_backtest.models import Action, Backtest, BacktestStatus, Bar, Trade
from portfolio_backtest.report import generate_html_report
from portfolio_backtest.schemas import (
    BacktestCreate,
    BacktestFinish,
    BacktestOut,
    MetricsOut,
    TradeOut,
)

STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Portfolio Backtest Server", lifespan=lifespan)


# --- Static files ---
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --- UI routes ---


@app.get("/", response_class=HTMLResponse)
def ui_index():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        return HTMLResponse("<h1>Static files not found</h1>", status_code=404)
    return HTMLResponse(index.read_text())


@app.get("/backtest/{backtest_id}/view", response_class=HTMLResponse)
def ui_backtest_detail(backtest_id: str):
    page = STATIC_DIR / "backtest.html"
    if not page.exists():
        return HTMLResponse("<h1>Page not found</h1>", status_code=404)
    return HTMLResponse(page.read_text())


@app.get("/backtest/{backtest_id}/report/view", response_class=HTMLResponse)
def ui_report(backtest_id: str):
    page = STATIC_DIR / "report.html"
    if not page.exists():
        return HTMLResponse("<h1>Page not found</h1>", status_code=404)
    return HTMLResponse(page.read_text())


# --- API: Backtest management ---


def _backtest_to_out(bt: Backtest, db: Session) -> BacktestOut:
    bar_count = db.query(func.count(Bar.id)).filter(Bar.backtest_id == bt.id).scalar()
    trade_count = db.query(func.count(Trade.id)).filter(Trade.backtest_id == bt.id).scalar()
    tickers = [
        r[0]
        for r in db.query(Bar.ticker).filter(Bar.backtest_id == bt.id).distinct().all()
    ]
    strategies = [
        r[0]
        for r in db.query(Trade.strategy).filter(Trade.backtest_id == bt.id).distinct().all()
    ]
    return BacktestOut(
        id=bt.id,
        name=bt.name,
        status=bt.status.value,
        initial_cash=bt.initial_cash,
        slippage_pct=bt.slippage_pct,
        commission_pct=bt.commission_pct,
        commission_min=bt.commission_min,
        commission_per_share=bt.commission_per_share,
        short_enabled=bt.short_enabled,
        short_margin_rate=bt.short_margin_rate,
        short_borrow_rate_annual=bt.short_borrow_rate_annual,
        max_position_pct=bt.max_position_pct,
        benchmark_ticker=bt.benchmark_ticker,
        created_at=bt.created_at,
        bar_count=bar_count,
        trade_count=trade_count,
        tickers=tickers,
        strategies=strategies,
    )


@app.post("/api/backtest", response_model=BacktestOut)
def create_backtest(payload: BacktestCreate, db: Session = Depends(get_db)):
    bt = Backtest(
        name=payload.name,
        initial_cash=payload.initial_cash,
        slippage_pct=payload.slippage_pct,
        commission_pct=payload.commission_pct,
        commission_min=payload.commission_min,
        commission_per_share=payload.commission_per_share,
        short_enabled=payload.short_enabled,
        short_margin_rate=payload.short_margin_rate,
        short_borrow_rate_annual=payload.short_borrow_rate_annual,
        max_position_pct=payload.max_position_pct,
        benchmark_ticker=payload.benchmark_ticker,
    )
    db.add(bt)
    db.commit()
    db.refresh(bt)
    return _backtest_to_out(bt, db)


@app.get("/api/backtest", response_model=list[BacktestOut])
def list_backtests(db: Session = Depends(get_db)):
    backtests = db.query(Backtest).order_by(Backtest.created_at.desc()).all()
    return [_backtest_to_out(bt, db) for bt in backtests]


@app.get("/api/backtest/{backtest_id}", response_model=BacktestOut)
def get_backtest(backtest_id: str, db: Session = Depends(get_db)):
    bt = db.query(Backtest).filter(Backtest.id == backtest_id).first()
    if not bt:
        raise HTTPException(404, "Backtest not found")
    return _backtest_to_out(bt, db)


@app.post("/api/backtest/{backtest_id}/finish")
def finish_backtest(
    backtest_id: str,
    payload: BacktestFinish = BacktestFinish(),
    db: Session = Depends(get_db),
):
    bt = db.query(Backtest).filter(Backtest.id == backtest_id).first()
    if not bt:
        raise HTTPException(404, "Backtest not found")
    if bt.status == BacktestStatus.FINISHED:
        raise HTTPException(400, "Backtest already finished")

    if payload.force_close_positions:
        # build engine to force close, then record the closing trades
        engine = build_engine_from_db(bt)
        last_ts = max(
            (b.timestamp for b in bt.bars),
            default=datetime.now(UTC),
        )
        for ticker, pos in list(engine.positions.items()):
            price = engine.latest_prices.get(ticker, 0.0)
            if pos.quantity > 0:
                trade = Trade(
                    backtest_id=bt.id,
                    strategy="__force_close__",
                    ticker=ticker,
                    action=Action.SELL,
                    quantity=pos.quantity,
                    price=price,
                    timestamp=last_ts,
                )
                db.add(trade)
            elif pos.quantity < 0:
                trade = Trade(
                    backtest_id=bt.id,
                    strategy="__force_close__",
                    ticker=ticker,
                    action=Action.COVER,
                    quantity=abs(pos.quantity),
                    price=price,
                    timestamp=last_ts,
                )
                db.add(trade)

    bt.status = BacktestStatus.FINISHED
    db.commit()
    return {"status": "finished", "id": bt.id}


@app.delete("/api/backtest/{backtest_id}")
def delete_backtest(backtest_id: str, db: Session = Depends(get_db)):
    bt = db.query(Backtest).filter(Backtest.id == backtest_id).first()
    if not bt:
        raise HTTPException(404, "Backtest not found")
    db.query(Bar).filter(Bar.backtest_id == backtest_id).delete()
    db.query(Trade).filter(Trade.backtest_id == backtest_id).delete()
    db.delete(bt)
    db.commit()
    return {"status": "deleted"}


# --- API: Trades ---


@app.get("/api/backtest/{backtest_id}/trades", response_model=list[TradeOut])
def list_trades(backtest_id: str, db: Session = Depends(get_db)):
    bt = db.query(Backtest).filter(Backtest.id == backtest_id).first()
    if not bt:
        raise HTTPException(404, "Backtest not found")
    trades = (
        db.query(Trade)
        .filter(Trade.backtest_id == backtest_id)
        .order_by(Trade.timestamp.desc())
        .all()
    )
    return [TradeOut.model_validate(t) for t in trades]


# --- API: Metrics + Report ---


@app.get("/api/backtest/{backtest_id}/metrics", response_model=MetricsOut)
def get_metrics(backtest_id: str, db: Session = Depends(get_db)):
    bt = db.query(Backtest).filter(Backtest.id == backtest_id).first()
    if not bt:
        raise HTTPException(404, "Backtest not found")
    engine = build_engine_from_db(bt)
    return engine.get_metrics()


@app.get("/api/backtest/{backtest_id}/report", response_class=HTMLResponse)
def get_report(backtest_id: str, db: Session = Depends(get_db)):
    bt = db.query(Backtest).filter(Backtest.id == backtest_id).first()
    if not bt:
        raise HTTPException(404, "Backtest not found")
    engine = build_engine_from_db(bt)
    returns = engine.get_returns_series()

    # benchmark returns if configured and bars exist
    benchmark_returns = None
    if bt.benchmark_ticker:
        benchmark_bars = (
            db.query(Bar)
            .filter(Bar.backtest_id == backtest_id, Bar.ticker == bt.benchmark_ticker)
            .order_by(Bar.timestamp)
            .all()
        )
        if benchmark_bars:
            prices = pd.Series(
                [b.close for b in benchmark_bars],
                index=pd.to_datetime([b.timestamp for b in benchmark_bars]),
            )
            benchmark_returns = prices.pct_change().dropna()

    html = generate_html_report(returns, title=f"Backtest: {bt.name}", benchmark_returns=benchmark_returns)
    return HTMLResponse(html)


# --- Webhook endpoint ---


@app.post("/webhook/{backtest_id}")
async def receive_webhook(backtest_id: str, request: Request, db: Session = Depends(get_db)):
    bt = db.query(Backtest).filter(Backtest.id == backtest_id).first()
    if not bt:
        raise HTTPException(404, "Backtest not found")
    if bt.status != BacktestStatus.RUNNING:
        raise HTTPException(400, "Backtest is not running")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    msg_type = body.get("type")

    if msg_type == "bar":
        try:
            bar = Bar(
                backtest_id=bt.id,
                ticker=body["ticker"],
                open=float(body["open"]),
                high=float(body["high"]),
                low=float(body["low"]),
                close=float(body["close"]),
                volume=float(body.get("volume", 0)),
                timestamp=datetime.fromisoformat(body["timestamp"]),
            )
        except (KeyError, ValueError, TypeError) as e:
            raise HTTPException(400, f"Invalid bar payload: {e}")
        db.add(bar)
        db.commit()
        return {"status": "ok", "type": "bar", "ticker": bar.ticker}

    elif msg_type == "order":
        try:
            action_str = body["action"].lower()
            if action_str not in ("buy", "sell", "short", "cover"):
                raise ValueError(f"Invalid action: {action_str}")
            trade = Trade(
                backtest_id=bt.id,
                strategy=body["strategy"],
                ticker=body["ticker"],
                action=Action(action_str),
                quantity=float(body["quantity"]),
                price=float(body["price"]),
                timestamp=datetime.fromisoformat(body["timestamp"]),
            )
        except (KeyError, ValueError, TypeError) as e:
            raise HTTPException(400, f"Invalid order payload: {e}")
        db.add(trade)
        db.commit()
        return {"status": "ok", "type": "order", "ticker": trade.ticker, "action": action_str}

    else:
        raise HTTPException(400, f"Unknown message type: {msg_type}")


def run():
    import uvicorn

    uvicorn.run("portfolio_backtest.main:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    run()
