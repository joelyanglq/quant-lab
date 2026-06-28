# trading_platform

Carver-style multi-strategy stock trading framework — backtest 与 IBKR 实盘共用同一份策略代码。

> 目标：在一个统一框架下跑 4 类股票策略（择时 / 多因子 / 配对 / 轮动），按 Carver《Systematic Trading》三层乘性架构组合，从回测到实盘小资金渐进上线。

---

## 30 秒概览

```
策略 (Alpha)        →  组合 (Combiner)    →  仓位 (RiskSizer)    →  执行 (ExecutionHandler)
 forecast(dt)           handcrafted/layered     vol target              backtest / shadow
 ∈ [-20, +20]           weighted average         half-Kelly              paper / live (IBKR)
                        forecast cap             drawdown scaling
                        NaN-aware
```

每一层是**独立纯函数**，可单独测试，可单独替换。这是 Carver Ch5 的"模块化三层乘性"模型。

四类策略共用同一个 `Alpha` 接口（`forecast(dt) -> dict[symbol → float]`），引擎对策略类型完全透明。

---

## 1. 目录结构

```
trading_platform/
├── core/
│   ├── events.py            # Bar / OrderEvent / FillEvent / RiskEvent (frozen dataclass)
│   ├── clock.py             # BacktestClock / LiveClock — 策略读 self.clock.now()
│   ├── context.py           # DataContext (PIT-safe), BacktestDataContext (Parquet)
│   └── engine.py            # 多频率事件循环 (heapq 合并 EOD + 1min + ...)
│
├── data/
│   ├── feed.py              # DataFeed ABC + BacktestFeed (Parquet)
│   ├── live_feed.py         # LiveDataFeed (IBKR 实时 bar)
│   └── storage/parquet.py   # ParquetStorage (年份分文件，沿用 OneBacktest)
│
├── strategy/
│   ├── alpha.py             # Alpha ABC, ScalingMixin (Carver forecast 协议)
│   ├── combiner.py          # WeightedCombiner / LayeredCombiner / HandcraftedCombiner
│   ├── sizer.py             # RiskSizer: vol target + half-Kelly + drawdown
│   ├── composite.py         # CompositeStrategy: alphas → combiner → sizer → orders
│   └── archetypes/
│       ├── single_name_timing.py   # HHT / QRS / MA / RSI / momentum
│       ├── cross_section.py        # 多因子横截面 (sector-neutral, expanding-window)
│       ├── pairs.py                # 协整对统计套利
│       └── rotation.py             # 板块/风格轮动 (sector ETF + ERC)
│
├── execution/
│   ├── base.py              # ExecutionHandler ABC
│   ├── simulated.py         # 回测 (T+1 close + 滑点 + 佣金)
│   ├── shadow.py            # 实时数据 + 模拟下单 (logs only)
│   └── live_ibkr.py         # 真实下单 IBKR (ib_insync)
│
├── risk/
│   ├── portfolio.py         # 持仓 / 现金 / equity 曲线
│   ├── kill_switch.py       # 6 个触发条件，state 持久化
│   ├── reconciliation.py    # 60s 对账 IBKR
│   └── monitoring.py        # 滑点 / 订单审计
│
├── analytics/
│   └── metrics.py           # SR + CI + 偏度 + 滚动SR + Carver 阈值标记
│
├── runtime/
│   ├── backtest.py          # python -m trading_platform.runtime.backtest ...
│   ├── live.py              # python -m trading_platform.runtime.live ...
│   └── pairs_scanner.py     # 离线协整扫描 (PairsAlpha 输入)
│
├── tests/                   # pytest 单元测试
└── templates/               # 4 类策略模板 (参考下方"添加新策略")
```

---

## 2. 三层架构详解（Carver 形式）

### 2.1 Forecast 协议（Carver Ch7）

每个 `Alpha` 子类实现：

```python
def forecast(self, dt: pd.Timestamp, ctx) -> dict[str, float]:
    # 返回 {symbol: forecast in [-20, +20] or NaN}
```

**契约**：

| 属性 | 要求 |
|---|---|
| 量纲 | 无单位（不是 z-score、不是收益率、不是概率） |
| 期望 | `E[\|forecast\|] ≈ 10` |
| Cap | 硬截断到 `[-20, +20]` |
| NaN | 表示"无信号"，组合层视为不参与 |
| 符号 | 正 = 看多，负 = 看空 |
| 解耦 | 与价格、波动率、资金量、当前持仓**完全无关** |

继承 `ScalingMixin` 自动处理 scaling：

```python
class MyAlpha(Alpha, ScalingMixin):
    def __init__(self):
        super().__init__()
        self._init_scaling(window=252)  # rolling 1y |raw|

    def forecast(self, dt, ctx):
        raw = {sym: self._compute_raw(sym, dt, ctx) for sym in self.universe(dt, ctx)}
        return self._scale_and_cap(raw)   # → [-20, +20]
```

### 2.2 Combiner（Carver Ch8）

三种合并方式：

| 类 | 用途 | 输入 |
|---|---|---|
| `WeightedCombiner` | 简单加权平均 | 显式权重或等权 |
| `HandcraftedCombiner` | Carver Ch4 查表权重 | 策略相关性矩阵 |
| `LayeredCombiner` | 乘性正交分层 | 分组 (regime × picking × timing) |

所有 combiner 都做 **±2σ winsorize** 再合并；NaN-aware（renormalize 权重）。

### 2.3 RiskSizer（Carver Ch9-10）

```
position[i] = (forecast[i] / 10)                 # 标准化
            × (target_vol_per_inst / σ[i])       # vol target
            × (capital × inst_weight[i] / price[i])  # 转手数
            × half_kelly_factor                  # 0.5 × SR / target_vol, clipped to [0.3, 1.0]
            × drawdown_scaler                    # 1.0 → 0.5 → 0 在 5% / 10% / 20% DD
```

最后做 max_leverage 约束（默认 1.0，实盘初期建议 0.5）。

---

## 3. 多频率引擎

引擎用 `heapq` 合并多个频率的 bar 流；策略通过 `trigger_freq` 声明：

```python
class CrossSectionAlpha(Alpha):
    trigger_freq = Frequency.EOD       # 只在日 bar close 触发

class FastTimingAlpha(Alpha):
    trigger_freq = Frequency.MIN_5     # 每个 5 分钟 bar 触发

class PairsAlpha(Alpha):
    trigger_freq = Frequency.MIN_30    # 每 30 分钟检查 spread
```

EOD 和 bar-close **可以同时存在**——它们只是不同 frequency 的 bar 事件，引擎 heap 合并按时间顺序分发，互不冲突。

---

## 4. 数据上下文（DataContext）

所有策略**只通过** `ctx.as_of(dt, key)` 读数据，框架保证 PIT 安全。

支持 keys：

```python
ctx.as_of(dt, "price_1d", lookback=504, symbols=[...])  # DataFrame: dates × symbols
ctx.as_of(dt, "price_1min", lookback=...)
ctx.as_of(dt, "fundamentals")                            # quarterly, by acceptedDate
ctx.as_of(dt, "gics_sector")                             # Series: symbol → sector
ctx.as_of(dt, "ewma_vol", symbols=[...])                 # Series: symbol → annual vol
ctx.as_of(dt, "cointegration_pairs")                     # DataFrame: a, b, β, valid_from/until
```

实盘版 `LiveDataContext`（`runtime/live.py`）混合 IBKR 实时 bar + 历史 Parquet，策略代码不变。

---

## 5. 数据约定

| 数据 | 路径 | 格式 |
|---|---|---|
| 1d bars | `data/processed/bars_1d/<year>_1d.parquet` | symbol, open, high, low, close, volume, idx=timestamp |
| 1min bars | `data/processed/bars_1min/<year>_1min.parquet` | 同上 |
| 基本面 | `data/fundamentals/massive/<symbol>.parquet` | 含 `acceptedDate` 列 |
| GICS sector | `data/reference/gics_sector_map.parquet` | symbol, sector, [effective_date] |
| Cointegration pairs | `data/processed/cointegration_pairs/valid_pairs.parquet` | symbol_a, symbol_b, hedge_ratio, valid_from, valid_until |
| Symbol universe | `data/_index_symbols.json` | `{"sp500": [...], "ndx100": [...]}` |

数据本身由原 `OneBacktest/data/etl/` 维护（不在本框架范围内）。

---

## 6. 命令行用法

### 回测

```bash
# 单票择时 (HHT)
python -m trading_platform.runtime.backtest \
    --start 2020-01-01 --end 2024-12-31 \
    --strategy timing --symbols AAPL,MSFT,NVDA --rule HHT

# 多因子横截面 (周频, sector-neutral)
python -m trading_platform.runtime.backtest \
    --start 2020-01-01 --end 2024-12-31 \
    --strategy cross_section --rebalance W-FRI

# 板块轮动 (月频)
python -m trading_platform.runtime.backtest \
    --start 2018-01-01 --end 2024-12-31 \
    --strategy rotation --rebalance M

# 协整对 (先扫描，再回测)
python -m trading_platform.runtime.pairs_scanner \
    --start 2020-01-01 --end 2024-01-01 --p 0.05
python -m trading_platform.runtime.backtest \
    --start 2024-01-01 --end 2025-12-31 \
    --strategy pairs
```

### 实盘渐进

```bash
# 1) Shadow: 连 IBKR 实时数据，但不发单（≥1 周）
python -m trading_platform.runtime.live --mode shadow \
    --strategy timing --symbols AAPL --rule HHT --bar-size '5 mins'

# 2) Paper: 端口 7497 真实下单到 paper 账户（≥2 周）
python -m trading_platform.runtime.live --mode paper --port 7497 \
    --strategy timing --symbols AAPL --rule HHT --bar-size '5 mins'

# 3) Live small: 端口 7496 真实资金，必须显式 confirm
python -m trading_platform.runtime.live --mode live --port 7496 \
    --strategy timing --symbols AAPL --rule HHT \
    --initial-capital 5000 --max-leverage 0.5 --max-daily-loss 0.01 \
    --i-understand-this-uses-real-money
```

---

## 7. 实盘保护：Kill-Switch

`runtime/state/kill_switch.json` 持久化状态。**任何**触发都会立即：取消所有挂单 + 拒绝新订单 + 报警。

| 触发条件 | 默认阈值 |
|---|---|
| 总杠杆 > max_leverage | 1.0 (实盘初期 0.5) |
| 日内 P&L < -max_daily_loss × NetLiq | 2% (初期 1%) |
| 单策略 7 日累计 < -5% × allocation | 暂停该策略 |
| IBKR 断线 > 5 分钟 | 自动 trip |
| Bar 价格相对前一根偏离 > 50% | 暂停该 symbol（不直接 trip） |
| 持仓对账 mismatch > 1 股或 $100 | 立即 trip |

**重启不会自动重置**——必须 `KillSwitch().reset(operator_confirm=True)` 显式解除。

---

## 8. 添加新策略（4 类模板）

`templates/` 目录下有 4 个可复制改造的骨架：

| 模板 | 适用 |
|---|---|
| `templates/timing_template.py` | 单票择时（HHT/QRS/MA/RSI 之外的新规则） |
| `templates/cross_section_template.py` | 加新 alpha 因子到多因子库 |
| `templates/pairs_template.py` | 自定义 pairs 信号（非 EG） |
| `templates/rotation_template.py` | 自定义轮动规则 |
| `templates/multi_strategy_composite.py` | 组合多个策略到一个 CompositeStrategy |

复制 → 改类名 / 业务逻辑 → 在 `runtime/backtest.py` 或自己写脚本里 import 即可。

---

## 9. 测试

```bash
cd D:\04_Project\quant-lab
pytest trading_platform/tests/ -v
```

关键测试：

- `test_smoke_buy_and_hold.py` — 端到端 engine 跑通
- `test_forecast_protocol.py` — 验证 Carver `E[|f|] ≈ 10` 契约

后续推荐补充：
- DataContext lookahead 检测（合成数据，验证 `as_of(dt)` 不返未来）
- RiskSizer 的 vol/Kelly/drawdown 单测
- 每个策略原型单跑 ≥1 年回测，flags 不应出现 SR 超阈值警告

---

## 10. 与 OneBacktest 的关系

`OneBacktest/` 仍然完整保留，作为：
- 数据 ETL 维护（`data/etl/`）
- 已有策略的运行参考（HHT advisor、cross-section pick_stocks、alpha101 pipeline）
- 历史回测对照基准

`trading_platform/` 是 Carver 化的目标架构。两者**不互相依赖**；策略可以从 OneBacktest 逐步迁移过来，迁移完成前两者并存。

---

## 11. Carver 关键决策对照（如果你想验证我的实现）

| Carver 章节 | 实现位置 |
|---|---|
| Ch5 三层架构 | `strategy/composite.py::CompositeStrategy` |
| Ch7 forecast 标准化 | `strategy/alpha.py::ScalingMixin` |
| Ch8 forecast 合并 + cap | `strategy/combiner.py` |
| Ch9 vol target | `strategy/sizer.py::RiskSizer.size` |
| Ch9 Half-Kelly | `strategy/sizer.py::RiskSizer.half_kelly_factor` |
| Ch10 头寸缩放 | `strategy/sizer.py::RiskSizer.size` 的 capital × weight / price |
| Ch11 instrument weight | `strategy/sizer.py::RiskSizer.size(instrument_weights=...)` |
| Ch4 handcrafting | `strategy/combiner.py::handcraft_weights` |
| Ch3 expanding-window 因子筛选 | `strategy/archetypes/cross_section.py::_handcraft_factor_weights` |
| Ch2 SR 上限 plausibility flag | `analytics/metrics.py::compute_metrics` |
| Ch3 SR 置信区间 | `analytics/metrics.py::sharpe_ci_t / sharpe_ci_bootstrap` |

---

## 12. 限制 / 已知问题

- 当前 `BacktestDataContext._price_panel` 在大 universe + 长 lookback 时会一次性加载全部年份；优化空间大。
- `LiveExecutionHandler` 的对账（reconciliation）只在 paper/live 启动时启线程，断开后没有重启逻辑——后续要加 watchdog。
- `LayeredCombiner` 的乘性结果重缩到 ±20 是简化做法，对极小 forecast 的乘积可能丢精度——首期可接受。
- Pairs 协整重验证（90 天 rerun EG test）目前在 scanner 里通过 `valid_until` 字段表达；运行期不会自动 re-scan，需要定期手动跑 scanner。

下一步推进顺序见 `openspec/changes/carver-multi-strategy-platform/tasks.md` 的 Phase 3 后续任务。
