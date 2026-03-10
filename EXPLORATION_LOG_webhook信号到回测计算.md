# 探索日志：一个 TradingView webhook 信号从被接收到完成回测计算，经历了哪些步骤？

## 调用链

[1] src/portfolio_backtest/main.py:262 → `receive_webhook()`
-> 接收 webhook 请求，根据 type 字段分发：bar 存为 Bar，order 存为 Trade，写入 DB
-> 入参: `{"type":"order","strategy":"ma_cross","ticker":"AAPL","action":"buy","quantity":10,"price":150.0,"timestamp":"..."}`
-> 输出: `{"status":"ok","type":"order","ticker":"AAPL","action":"buy"}`

[2] src/portfolio_backtest/engine.py:256 → `build_engine_from_db()`
-> 从 DB 读取 Backtest 的所有 Bar 和 Trade，合并为事件列表按时间排序（同一时刻 bar 在 order 前），逐条回放到 Engine
-> 入参: Backtest ORM 对象（含 bars 和 trades 关系）
-> 输出: 回放完毕的 Engine 实例

[3] src/portfolio_backtest/engine.py:67 → `Engine.process_bar()`
-> 更新 ticker 最新价格，计算当前组合总价值并追加到 equity_series
-> 入参: process_bar("AAPL", 155, t3)，当前 cash=90000, 持有 AAPL 100 股
-> 输出: equity_series 追加 (t3, 90000+100×155) = (t3, 105500)

[4] src/portfolio_backtest/engine.py:71 → `Engine.process_order()`
-> 应用滑点计算实际成交价，计算手续费，根据 action 分发到 _apply_buy/_apply_sell/_apply_short/_apply_cover 更新持仓和现金
-> 入参: process_order("AAPL", "buy", 100, 150.0, t2)，slippage=0.1%, commission=0.1%
-> 输出: 滑点后价格 150.15，手续费 15.015，cash 减少至 84969.985，持仓 AAPL quantity=100 cost_basis=150.15

[5] src/portfolio_backtest/engine.py:166 → `Engine.get_metrics()`
-> 基于 equity_series 计算年化收益、夏普率、Sortino、最大回撤、胜率、盈亏比等指标
-> 入参: 回放完毕的 Engine 实例
-> 输出: `{"total_return":1500,"total_return_pct":1.5,"sharpe_ratio":...,"max_drawdown_pct":-0.015,"win_rate":100.0,...}`

## 时序图

```plantuml
@startuml
actor TradingView
participant "main.py\nreceive_webhook()" as Webhook
database "SQLite DB" as DB
participant "main.py\nget_metrics()" as Metrics
participant "engine.py\nbuild_engine_from_db()" as Builder
participant "engine.py\nEngine" as Engine

== 数据入库阶段 ==
TradingView -> Webhook: POST /webhook/{id}\n{"type":"bar", ...}
Webhook -> DB: INSERT Bar
Webhook --> TradingView: {"status":"ok","type":"bar"}

TradingView -> Webhook: POST /webhook/{id}\n{"type":"order", ...}
Webhook -> DB: INSERT Trade
Webhook --> TradingView: {"status":"ok","type":"order"}

== 回测计算阶段 ==
Metrics -> Builder: build_engine_from_db(backtest)
Builder -> DB: 读取所有 Bar + Trade
Builder -> Builder: 按时间排序\n(同时刻 bar 优先)
loop 遍历事件
    Builder -> Engine: process_bar(ticker, close, ts)
    Engine -> Engine: 更新价格 + 记录净值快照
    Builder -> Engine: process_order(ticker, action, qty, price, ts)
    Engine -> Engine: 滑点 → 手续费 → 更新持仓/现金 → 净值快照
end
Builder --> Metrics: 返回 Engine 实例

Metrics -> Engine: get_metrics()
Engine -> Engine: equity → returns → sharpe/drawdown/win_rate
Engine --> Metrics: 指标字典
@enduml
```

## 类依赖关系

```plantuml
@startuml
class Engine {
    +config: EngineConfig
    +cash: float
    +positions: dict[str, Position]
    +equity_series: list[tuple]
    +latest_prices: dict[str, float]
    +completed_trades: list[TradeRecord]
    +process_bar()
    +process_order()
    +get_metrics()
    +get_returns_series()
    +get_equity_df()
}

class EngineConfig <<frozen>> {
    +initial_cash: float
    +slippage_pct: float
    +commission_pct: float
    +commission_min: float
    +commission_per_share: float
    +short_enabled: bool
}

class Position <<frozen>> {
    +ticker: str
    +quantity: float
    +cost_basis: float
}

class TradeRecord {
    +ticker: str
    +entry_price: float
    +exit_price: float
    +quantity: float
    +side: str
    +pnl: float
}

class Backtest <<ORM>> {
    +id: str
    +name: str
    +status: BacktestStatus
    +initial_cash: float
    +bars: list[Bar]
    +trades: list[Trade]
}

class Bar <<ORM>> {
    +ticker: str
    +open/high/low/close: float
    +volume: float
    +timestamp: datetime
}

class Trade <<ORM>> {
    +strategy: str
    +ticker: str
    +action: Action
    +quantity: float
    +price: float
    +timestamp: datetime
}

Engine *-- EngineConfig
Engine *-- "0..*" Position
Engine *-- "0..*" TradeRecord
Backtest o-- "0..*" Bar
Backtest o-- "0..*" Trade
Engine ..> Backtest : build_engine_from_db()
@enduml
```
