---
name: OneBacktest
description: OneBacktest 框架的架构说明、模块文档和使用指南。涵盖目录结构、核心类职责、数据流、策略编写方法和运行示例。
---

# OneBacktest

Event-driven 回测框架，面向美股日线/分钟线策略研发。

## 目录结构

```
OneBacktest/
├── event.py                    # 事件定义 (Bar/Order/Fill/Signal)
├── run_example.py              # 运行示例
│
├── data/                       # 数据层
│   ├── types.py                #   Bar, Frequency, AdjustType
│   ├── feed.py                 #   DataFeed 抽象基类 (Iterator)
│   ├── history.py              #   HistoryManager (1d 缓存 + 1min 按需加载)
│   ├── fundamentals.py         #   Point-in-Time 基本面加载
│   ├── prices.py               #   价格面板加载工具
│   ├── sources/
│   │   └── historic.py         #   HistoricFeed (Parquet → heapq 时间排序)
│   └── storage/
│       └── parquet.py          #   ParquetStorage (按年分文件读写)
│
├── backtest/                   # 回测引擎
│   ├── engine.py               #   BacktestEngine (主循环 + 聚合回调)
│   └── analytics.py            #   calc_metrics, print_report
│
├── strategy/                   # 策略层
│   ├── base.py                 #   Strategy 抽象基类 (回调 + 下单 API)
│   ├── portfolio.py            #   Portfolio (持仓/市值/P&L 记账)
│   ├── factors.py              #   46 因子库 (技术/反转/基本面/1min 微观结构...)
│   └── examples/               #   示例策略
│       ├── buy_and_hold.py
│       ├── rsi_reversal.py
│       ├── momentum_vol_sized.py
│       ├── hht_timing.py
│       ├── qrs_timing.py
│       └── weekly_factor.py
│
└── execution/                  # 执行层
    └── handler.py              #   SimulatedExecutionHandler (close 价成交)
```

## 数据流

```
ParquetStorage
      │
      ▼
  HistoricFeed ──(Bar)──▶ BacktestEngine 主循环
                              │
                              ├─▶ HistoryManager._on_bar()   ← 累积 1d 滚动窗口
                              ├─▶ ExecutionHandler            ← 执行 pending orders (T+1)
                              ├─▶ Portfolio.update_market()   ← 更新持仓市值
                              └─▶ Strategy.on_bar()           ← 策略决策 → 产出 OrderEvent
                                    │
                                    ├── self.history.panel()         ← 1d 面板查询
                                    ├── self.history.panel_1min()    ← 1min 面板查询
                                    ├── self.buy() / self.sell()     ← 下单
                                    └── self.rebalance_to()          ← 一键调仓
```

## 核心类

### BacktestEngine

主循环逻辑：逐 Bar 驱动，T 日信号 → T+1 日 close 价成交。

日期边界自动触发聚合回调：

| 回调 | 触发时机 |
|------|----------|
| `on_market_close(dt)` | 每个交易日所有 symbol bar 处理完 |
| `on_week_end(dt)` | 每周最后一个交易日 |
| `on_month_end(dt)` | 每月最后一个交易日 |

初始化参数：

```python
BacktestEngine(
    data_feed,          # DataFeed 实例
    strategy,           # Strategy 子类实例
    portfolio,          # Portfolio 实例
    execution_handler,  # ExecutionHandler 实例
    latest_prices={},   # 共享最新价格字典
    history_lookback=504,   # HistoryManager 滚动窗口大小
    storage_1min=None,      # ParquetStorage(bars_1min)，传入则开启 1min 查询
)
```

### Strategy

抽象基类，Engine 启动时自动注入以下属性：

| 属性 | 类型 | 说明 |
|------|------|------|
| `self.history` | `HistoryManager` | 历史数据查询 |
| `self.positions` | `dict` | 当前持仓 `{symbol: qty}` |
| `self.holdings` | `dict` | 当前市值 `{symbol: mktval, 'cash': ..., 'total': ...}` |
| `self.latest_prices` | `dict` | 最新 Bar `{symbol: Bar}` |
| `self.events` | `Queue` | 事件队列 (内部使用) |

下单 API：

```python
self.buy(symbol, qty)
self.sell(symbol, qty)
self.rebalance_to(target_symbols, weights=None)  # weights=None 时等权
```

查询 API：

```python
self.get_position(symbol) -> int
self.get_portfolio_value() -> float
self.get_cash() -> float
```

### HistoryManager

数据层 handler，由 Engine 创建并注入到 `strategy.history`。

**1d 数据** — Engine 每根 bar 自动累积到 deque 滚动窗口：

```python
# 截面面板: DataFrame (dates × symbols)
close = self.history.panel('close', periods=252)
high  = self.history.panel('high', periods=60)

# 单 symbol 序列: 1-D numpy array
arr = self.history.get('AAPL', 'close', periods=20)
```

**1min 数据** — 按需从 ParquetStorage 加载（需初始化时传入 `storage_1min`）：

```python
# 宽表面板，单字段返回 DataFrame，多字段返回 tuple
close_1m = self.history.panel_1min('close', start='2025-01-01', end='2025-06-01')
close_1m, vol_1m = self.history.panel_1min('close', 'volume', start=..., end=...)

# OHLCV dict
panels = self.history.ohlcv_1min(start=..., end=...)  # {'open': df, 'high': df, ...}

# 长表 (含 symbol 列)，用于 groupby
raw = self.history.raw_1min(start=..., end=...)
```

1min 数据不缓存，每次调用直接从 Parquet 读取。RTH 时段自动过滤 (09:35-15:57)。

### ParquetStorage

按年分文件的 Parquet 读写后端。文件格式 `{year}_{frequency}.parquet`。

```python
storage = ParquetStorage('./data/processed/bars_1d')
df = storage.load(['AAPL', 'MSFT'], start, end, '1d')
storage.save(df, '1d')  # 增量合并，按 (timestamp, symbol) 去重
```

### Portfolio

纯记账组件，跟踪持仓数量、市值、现金和交易日志。不生成订单。

```python
portfolio.get_equity_curve() -> pd.DataFrame  # index=timestamp, columns 含 total/cash
portfolio.trade_log  # List[FillEvent]
```

## 因子库

`strategy/factors.py` 提供 46 个因子，按数据源分类：

| 分类 | 因子数 | 数据源 | 函数 |
|------|--------|--------|------|
| 技术 | 5 | 1d | `compute_technical_factors(close, high, low)` |
| 反转 | 4 | 1d | `compute_reversal_factors(close, open_price)` |
| 基本面 | 8 | quarterly | `compute_fundamental_factors(symbols, close)` |
| 微观结构 | 3 | 1min+1d | `compute_microstructure_factors(close_1m, vol_1m, close, open_p)` |
| 博弈 | 5 | 1min | `compute_battle_factors(raw_1m)` |
| 激增 | 3 | 1min | `compute_surge_factors(close_1m, vol_1m)` |
| 回归 | 4 | 1min | `compute_regression_factors(close_1m, vol_1m)` |
| 模糊 | 3 | 1min | `compute_fuzzy_factors(close_1m, vol_1m)` |
| 灾后重建 | 2 | 1min | `compute_rebuild_factors(panels_1m)` |
| 潮汐 | 4 | 1min | `compute_tidal_factors(close_1m, vol_1m)` |
| 跳跃 | 5 | 1min+1d | `compute_jump_factors(close_1m, close, high, low)` |

一键计算：

```python
from strategy.factors import compute_all_factors

all_f = compute_all_factors(
    self.history,
    start_1min='2025-01-01',   # 可选，不传则跳过 1min 因子
    end_1min='2025-06-01',
    periods=504,
)
# all_f: {factor_name: DataFrame(dates × symbols)}
```

工具函数：

```python
cross_sectional_zscore(factor_df)        # 横截面 MAD winsorize → z-score
rolling_composite(daily_panel, window)    # rolling mean+std → z-score 等权
```

## 运行示例

```python
import pandas as pd
from data import ParquetStorage, HistoricFeed
from backtest.engine import BacktestEngine
from backtest.analytics import calc_metrics, print_report
from strategy.portfolio import Portfolio
from execution.handler import SimulatedExecutionHandler

# 自定义策略
from strategy.base import Strategy
from data.types import Bar

class MyStrategy(Strategy):
    def __init__(self, symbols):
        self.symbols = symbols

    def on_bar(self, bar: Bar):
        if self.get_position(bar.symbol) == 0:
            self.buy(bar.symbol, 100)

# 组装并运行
storage = ParquetStorage(r'D:\04_Project\quant-lab\data\processed\bars_1d')
feed = HistoricFeed(storage, frequency='1d')
feed.subscribe(['AAPL'], pd.Timestamp('2020-01-01'), pd.Timestamp('2025-12-31'))

latest_prices = {}
portfolio = Portfolio(['AAPL'], latest_prices, initial_capital=100000)
execution = SimulatedExecutionHandler(latest_prices)
strategy = MyStrategy(['AAPL'])

engine = BacktestEngine(feed, strategy, portfolio, execution, latest_prices)
engine.run_backtest()

metrics = calc_metrics(portfolio.get_equity_curve(), portfolio.trade_log)
print_report(metrics)
```

内置示例策略直接运行：

```bash
cd OneBacktest
python run_example.py
```

## 事件类型

定义在 `event.py`，均为 frozen dataclass：

| 事件 | 用途 |
|------|------|
| `OrderEvent` | 策略产出，含 symbol/side/quantity/order_type |
| `FillEvent` | 执行器产出，含 fill_price/commission/slippage |
| `MarketEvent` | 预留，未来实盘推送用 |
| `SignalEvent` | 预留，含 signal_type + strength |
| `CancelEvent` | 订单取消 |

## 数据存储

框架使用的 Parquet 数据位于项目根目录 `data/` 下：

```
quant-lab/data/
├── processed/bars_1d/      # 日线 OHLCV (年份文件: 2020_1d.parquet, ...)
├── processed/bars_1min/    # 分钟线 OHLCV
├── fundamentals/massive/   # 财报数据 (symbol.parquet)
└── _index_symbols.json     # S&P500 + NASDAQ100 成分股缓存
```

数据由独立的 ETL 脚本维护 (`data/etl/`)，不属于回测框架本身。
