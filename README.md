# Portfolio Backtest Server

接收 TradingView 策略回测的 webhook 信号，计算投资组合表现指标并生成可视化报告。

## 功能

- 接收 TradingView webhook（行情 bar + 订单信号）
- 支持多策略汇聚到同一个投资组合
- 支持做多（buy/sell）和做空（short/cover）
- 可配置滑点、手续费、做空参数
- 计算夏普率、最大回撤、胜率等指标
- 生成 quantstats HTML 报告
- WebUI 管理回测

## 安装

```bash
cd portfolio-backtest
pip install -e ".[dev]"
```

## 启动

```bash
portfolio-backtest
# 或
python -m portfolio_backtest.main
```

浏览器打开 http://localhost:8000

## TradingView 配置

在 TradingView 中创建 alert，webhook URL 设为：

```
http://<your-ip>:8000/webhook/<backtest_id>
```

### 行情消息（每个 bar 发一次）

在 Pine Script 中，每个 bar 发送价格数据：

```json
{
  "type": "bar",
  "ticker": "{{ticker}}",
  "open": {{open}},
  "high": {{high}},
  "low": {{low}},
  "close": {{close}},
  "volume": {{volume}},
  "timestamp": "{{time}}"
}
```

### 订单消息（有交易时发）

```json
{
  "type": "order",
  "strategy": "my_strategy",
  "ticker": "{{ticker}}",
  "action": "buy",
  "quantity": 10,
  "price": {{close}},
  "timestamp": "{{time}}"
}
```

`action` 支持: `buy`, `sell`, `short`, `cover`

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/backtest` | 创建回测 |
| GET | `/api/backtest` | 列出所有回测 |
| GET | `/api/backtest/{id}` | 回测详情 |
| POST | `/api/backtest/{id}/finish` | 结束回测 |
| DELETE | `/api/backtest/{id}` | 删除回测 |
| POST | `/webhook/{id}` | 接收 webhook |
| GET | `/api/backtest/{id}/trades` | 订单列表 |
| GET | `/api/backtest/{id}/metrics` | JSON 指标 |
| GET | `/api/backtest/{id}/report` | HTML 报告 |

## 测试

```bash
pytest tests/ -v
```
