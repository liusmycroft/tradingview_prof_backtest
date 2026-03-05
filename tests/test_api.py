"""Tests for the webhook API and backtest management."""

import pytest
from fastapi.testclient import TestClient

from portfolio_backtest.database import Base, engine as db_engine
from portfolio_backtest.main import app


@pytest.fixture(autouse=True)
def setup_db():
    """Create fresh tables for each test."""
    Base.metadata.create_all(bind=db_engine)
    yield
    Base.metadata.drop_all(bind=db_engine)


@pytest.fixture
def client():
    return TestClient(app)


class TestBacktestCRUD:
    def test_create_backtest(self, client):
        resp = client.post("/api/backtest", json={"name": "test"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "test"
        assert data["status"] == "running"
        assert data["initial_cash"] == 100000.0

    def test_list_backtests(self, client):
        client.post("/api/backtest", json={"name": "bt1"})
        client.post("/api/backtest", json={"name": "bt2"})
        resp = client.get("/api/backtest")
        assert len(resp.json()) == 2

    def test_get_backtest(self, client):
        bt = client.post("/api/backtest", json={"name": "test"}).json()
        resp = client.get(f"/api/backtest/{bt['id']}")
        assert resp.json()["name"] == "test"

    def test_delete_backtest(self, client):
        bt = client.post("/api/backtest", json={"name": "test"}).json()
        resp = client.delete(f"/api/backtest/{bt['id']}")
        assert resp.status_code == 200
        resp = client.get(f"/api/backtest/{bt['id']}")
        assert resp.status_code == 404

    def test_finish_backtest(self, client):
        bt = client.post("/api/backtest", json={"name": "test"}).json()
        resp = client.post(f"/api/backtest/{bt['id']}/finish")
        assert resp.json()["status"] == "finished"

    def test_finish_already_finished(self, client):
        bt = client.post("/api/backtest", json={"name": "test"}).json()
        client.post(f"/api/backtest/{bt['id']}/finish")
        resp = client.post(f"/api/backtest/{bt['id']}/finish")
        assert resp.status_code == 400


class TestWebhook:
    def _create_running(self, client):
        return client.post("/api/backtest", json={"name": "test"}).json()

    def test_receive_bar(self, client):
        bt = self._create_running(client)
        resp = client.post(f"/webhook/{bt['id']}", json={
            "type": "bar",
            "ticker": "AAPL",
            "open": 100, "high": 105, "low": 99, "close": 103,
            "volume": 1000,
            "timestamp": "2024-01-15T10:00:00",
        })
        assert resp.status_code == 200
        assert resp.json()["type"] == "bar"

    def test_receive_order(self, client):
        bt = self._create_running(client)
        resp = client.post(f"/webhook/{bt['id']}", json={
            "type": "order",
            "strategy": "momentum",
            "ticker": "AAPL",
            "action": "buy",
            "quantity": 10,
            "price": 185.5,
            "timestamp": "2024-01-15T10:30:00",
        })
        assert resp.status_code == 200
        assert resp.json()["action"] == "buy"

    def test_reject_webhook_for_finished(self, client):
        bt = self._create_running(client)
        client.post(f"/api/backtest/{bt['id']}/finish")
        resp = client.post(f"/webhook/{bt['id']}", json={
            "type": "bar",
            "ticker": "AAPL",
            "open": 100, "high": 105, "low": 99, "close": 103,
            "timestamp": "2024-01-15T10:00:00",
        })
        assert resp.status_code == 400

    def test_invalid_type(self, client):
        bt = self._create_running(client)
        resp = client.post(f"/webhook/{bt['id']}", json={"type": "unknown"})
        assert resp.status_code == 400

    def test_bar_count_updates(self, client):
        bt = self._create_running(client)
        for i in range(3):
            client.post(f"/webhook/{bt['id']}", json={
                "type": "bar",
                "ticker": "AAPL",
                "open": 100, "high": 105, "low": 99, "close": 103,
                "timestamp": f"2024-01-{15+i}T10:00:00",
            })
        detail = client.get(f"/api/backtest/{bt['id']}").json()
        assert detail["bar_count"] == 3

    def test_trades_list(self, client):
        bt = self._create_running(client)
        client.post(f"/webhook/{bt['id']}", json={
            "type": "order",
            "strategy": "strat_a",
            "ticker": "AAPL",
            "action": "buy",
            "quantity": 10,
            "price": 100,
            "timestamp": "2024-01-15T10:00:00",
        })
        trades = client.get(f"/api/backtest/{bt['id']}/trades").json()
        assert len(trades) == 1
        assert trades[0]["strategy"] == "strat_a"


class TestMetrics:
    def test_metrics_with_trades(self, client):
        bt = client.post("/api/backtest", json={"name": "test"}).json()
        bid = bt["id"]

        # send bars and orders
        for day, close in [(15, 100), (16, 105), (17, 110), (18, 108), (19, 115)]:
            client.post(f"/webhook/{bid}", json={
                "type": "bar", "ticker": "AAPL",
                "open": close, "high": close+2, "low": close-2, "close": close,
                "timestamp": f"2024-01-{day}T16:00:00",
            })

        client.post(f"/webhook/{bid}", json={
            "type": "order", "strategy": "test", "ticker": "AAPL",
            "action": "buy", "quantity": 10, "price": 100,
            "timestamp": "2024-01-15T16:00:00",
        })
        client.post(f"/webhook/{bid}", json={
            "type": "order", "strategy": "test", "ticker": "AAPL",
            "action": "sell", "quantity": 10, "price": 115,
            "timestamp": "2024-01-19T16:00:00",
        })

        resp = client.get(f"/api/backtest/{bid}/metrics")
        m = resp.json()
        assert m["total_return"] > 0
        assert m["total_trades"] == 1
        assert m["winning_trades"] == 1
