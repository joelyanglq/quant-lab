# OneBacktest 架构演进方案 (v1 → v2)

> 本文档记录 OneBacktest 从 bar-only event-driven 架构向 **多源数据 + 多信号合成** 架构的演进设计。

---

## 目录

1. [现有架构 (v1) 回顾](#1-现有架构-v1-回顾)
2. [v1 遇到的问题](#2-v1-遇到的问题)
3. [v2 架构总览](#3-v2-架构总览)
4. [新增组件详解](#4-新增组件详解)
5. [组件交互流程](#5-组件交互流程)
6. [v2 的已知局限](#6-v2-的已知局限)
7. [未来演进方向](#7-未来演进方向)
8. [附录：术语表](#8-附录术语表)

---

## 1. 现有架构 (v1) 回顾

### 1.1 核心元素

| 组件 | 职责 | 关键接口 |
|------|------|----------|
| **DataFeed** | 按时间顺序逐根发出 Bar (OHLCV) | `next() → Bar`, `has_next()` |
| **HistoryManager** | 为策略缓存滚动 1d 窗口；按需读取 1min | `panel()`, `get()`, `panel_1min()` |
| **Strategy** | 接收 Bar，产出 OrderEvent | `on_bar()`, `buy()`, `sell()`, `rebalance_to()` |
| **Portfolio** | 纯记账：持仓、市值、P&L | `update_market()`, `update_fill()` |
| **ExecutionHandler** | 将 OrderEvent 转为 FillEvent | `execute_order()` |
| **BacktestEngine** | 主循环编排以上所有组件 | `run_backtest()` |

### 1.2 数据流

```
ParquetStorage
      │
      ▼
  HistoricFeed ──(Bar)──▶ BacktestEngine
                              │
                              ├─ HistoryManager._on_bar()     累积 1d 滚动窗口
                              ├─ ExecutionHandler              执行 pending orders (T+1)
                              ├─ Portfolio.update_market()     mark-to-market
                              └─ Strategy.on_bar()             决策 → OrderEvent
                                    │
                                    └─ buy() / sell()          直接下单
```

### 1.3 v1 的设计优势

- **简洁**：一条数据流（Bar stream），一个决策点（Strategy），一个输出（Order）
- **无前视偏差**：T 日信号 → T+1 close 价成交，由引擎保证
- **向量化历史查询**：`history.panel()` 返回 dates × symbols 的 DataFrame，策略可以用 pandas/numpy 做截面运算
- **T+1 执行模型**：订单延迟一根 bar 执行，模拟真实下单延迟

---

## 2. v1 遇到的问题

### 问题 A：数据源不止 Bar

| 策略方向 | 需要的数据 | v1 怎么获取 | 问题 |
|----------|-----------|-------------|------|
| 截面因子 | ROE, EV/EBITDA 等财报数据 | 策略自己调 `build_fundamental_panel()` | 引擎不知道，时间对齐靠策略自觉 |
| Regime 配置 | 3M T-Bill rate | 策略自己读 CSV | 同上 |
| 波动率目标 | 已实现波动率 (realized vol) | 策略自己算 `panel('close').pct_change().std()` | 重复计算，没有统一口径 |

**根本矛盾**：Bar 是逐笔流式的（iterator），而财报/利率等数据天然是 **低频 panel/snapshot** 形式。DataFeed 的 `next() → Bar` 接口无法容纳这些异构数据。

**后果**：
- 策略代码里混杂了数据加载逻辑，违反关注点分离
- 没有统一的 point-in-time 保护，前视偏差风险转嫁给策略作者
- 不同策略对同一基础数据（如 ROE）各自加载，无法复用

### 问题 B：多策略信号无法合成

当前架构是 **一个 Strategy 实例直接下单**。但实际需求是：

```
同一个 AAPL：
  - 择时策略说：看多，forecast = +0.8
  - Regime 模型说：当前是 risk-off，应缩仓 40%
  - 截面因子说：AAPL 排名靠前，forecast = +0.6
```

v1 下只能写一个巨型策略把所有逻辑塞在一起，或者跑多个独立回测再手动合并。

**根本矛盾**：Strategy 的输出是 OrderEvent（离散的买卖指令），不是连续的 forecast 信号。没有中间层做信号合成和仓位计算。

**后果**：
- 择时 × regime 的组合需要硬编码在一个策略里，复用性差
- 无法对单个 alpha 做独立归因
- 波动率目标定仓（vol targeting）没有统一入口

---

## 3. v2 架构总览

### 3.1 设计原则

1. **向后兼容**：v1 的 Strategy（直接 `buy()/sell()`）继续工作，不受影响
2. **增量引入**：新组件可以单独使用，不要求全部一起上线
3. **关注点分离**：数据加载、信号生成、信号合成、仓位计算四个阶段清晰分离
4. **统一 PIT 保护**：所有数据源的 point-in-time 对齐由框架保证，不依赖策略作者

### 3.2 新增元素一览

| 组件 | 角色 | 解决什么问题 |
|------|------|-------------|
| **DataContext** | 异构数据注册中心 | 问题 A：财报/利率等非 bar 数据的统一接入与 PIT 查询 |
| **Alpha** | 信号生成器抽象基类 | 问题 B：将"策略"拆分为独立的 forecast 生产单元 |
| **ForecastCombiner** | 信号合成器 | 问题 B：多个 Alpha 的 forecast 加权合并 |
| **VolTargetSizer** | 仓位计算器 | 问题 B：forecast → 考虑波动率目标后的目标股数 |
| **CompositeStrategy** | 组合策略编排器 | 问题 B：串联 Alpha → Combiner → Sizer → Order 的完整流程 |

### 3.3 全局架构图

```
                    ┌─────────────────────────────────┐
                    │         DataContext              │
                    │                                  │
                    │  bars  → HistoryManager (已有)    │
                    │  panels → {name: DataFrame}      │
                    │    ├── "roe"  (dates × symbols)  │
                    │    ├── "fcf_yield"               │
                    │    ├── "tbill_3m" (dates × 1)    │
                    │    └── ...                       │
                    │                                  │
                    │  as_of(dt, name) → 截面/标量     │
                    │  panel_between(name, start, end) │
                    └───────────┬─────────────────────┘
                                │ 注入
                    ┌───────────▼─────────────────────┐
                    │       BacktestEngine             │
                    │  (主循环不变，新增 DataContext 注入) │
                    └───────────┬─────────────────────┘
                                │
              ┌─────────────────┼─────────────────────┐
              │                 │                      │
              ▼                 ▼                      ▼
     ┌────────────┐   ┌────────────────┐    ┌──────────────────┐
     │ v1 Strategy │   │ CompositeStrategy│  │ (未来) 其他编排器 │
     │             │   │                  │  │                  │
     │ on_bar()    │   │ ┌──────────┐    │  └──────────────────┘
     │ buy()/sell()│   │ │ Alpha ×N │    │
     │ (照旧)      │   │ │ 各自产出  │    │
     │             │   │ │ forecast │    │
     └────────────┘   │ └────┬─────┘    │
                       │      │          │
                       │ ┌────▼────────┐ │
                       │ │ Forecast    │ │
                       │ │ Combiner    │ │
                       │ └────┬────────┘ │
                       │      │          │
                       │ ┌────▼────────┐ │
                       │ │ VolTarget   │ │
                       │ │ Sizer       │ │
                       │ └────┬────────┘ │
                       │      │          │
                       │   rebalance()   │
                       └──────────────────┘
```

---

## 4. 新增组件详解

### 4.1 DataContext — 异构数据注册中心

#### 应对的场景

策略需要 bar 之外的数据（财报、利率、自定义因子面板等），且需要框架层面保证 point-in-time 安全。

#### 设计

```python
class DataContext:
    """
    引擎级数据注册中心。
    在回测启动时一次性加载所有注册的 panel，运行时提供 PIT 查询。
    """

    def __init__(self):
        self._panels: Dict[str, pd.DataFrame] = {}
        # 每个 panel: index=DatetimeIndex (数据可用日期), columns=symbols 或单列

    # ── 注册 ──
    def register(self, name: str, panel: pd.DataFrame):
        """
        注册一个数据面板。
        panel 的 index 是数据可用日期 (对于财报，是 filing_date 而非 report_date)。
        """

    # ── 查询 ──
    def as_of(self, dt: pd.Timestamp, name: str) -> pd.Series | float:
        """
        返回 name 面板在 dt 日 (含) 之前最近一行数据。
        - 多 symbol 面板 → pd.Series (symbol → value)
        - 单列面板 (如利率) → float 标量
        防止前视偏差：只返回 index <= dt 的数据。
        """

    def panel_between(self, name: str, start, end) -> pd.DataFrame:
        """
        返回 [start, end] 之间的完整面板切片。
        用于策略需要一段历史（如计算 IC、回归）时。
        """

    def has(self, name: str) -> bool:
        """检查某个 panel 是否已注册。"""

    @property
    def names(self) -> List[str]:
        """列出所有已注册的 panel 名称。"""
```

#### 与现有组件的关系

- **HistoryManager** 不变，继续负责 bar 的滚动缓存。DataContext 不替代它
- **Engine** 新增一个 `data_context` 参数，启动时注入到 Strategy 的 `self.data` 属性
- **fundamentals.py** 的 `build_fundamental_panel()` 变成 DataContext 的数据源之一，而不是策略直接调用

#### 典型用法

```python
# ── 引擎组装时 ──
from data.fundamentals import build_fundamental_panel

ctx = DataContext()

# 注册财报 panel (PIT aligned)
fund_panels = build_fundamental_panel(symbols, ['totalRevenue', 'netIncome'], trading_dates)
for name, panel in fund_panels.items():
    ctx.register(name, panel)

# 注册利率
tbill = pd.read_csv('data/processed/rates/DTB3.csv', index_col=0, parse_dates=True)
ctx.register('tbill_3m', tbill)

engine = BacktestEngine(feed, strategy, portfolio, execution,
                        latest_prices, data_context=ctx)

# ── 策略里 ──
class MyStrategy(Strategy):
    def on_market_close(self, dt):
        roe = self.data.as_of(dt, 'roe')          # Series(symbol → float)
        rf  = self.data.as_of(dt, 'tbill_3m')     # float
        close_252 = self.history.panel('close', 252)
        # ... 用 roe, rf, close 做决策
```

#### 为什么不把所有数据都塞进 HistoryManager

| | HistoryManager | DataContext |
|---|---|---|
| **数据频率** | 每根 bar 更新 (1d / 1min) | 低频、不规则 (季报/月度) |
| **更新方式** | 引擎逐 bar push | 预加载，查询时 PIT 截断 |
| **数据形状** | 固定字段 (OHLCV) | 任意命名的 panel |
| **内存模型** | 滚动窗口 deque(maxlen) | 全量加载 (数据量小) |

硬把财报塞进 HistoryManager 会导致：
1. HistoryManager 需要处理非 OHLCV 字段名，接口膨胀
2. 低频数据大部分交易日没有更新，逐 bar push 模型不适用
3. 季报需要 TTM 等预处理，不是简单的 append

---

### 4.2 Alpha — 信号生成器

#### 应对的场景

将"策略"拆分为可独立开发、独立测试、独立归因的 forecast 生产单元。

#### 设计

```python
class Alpha(ABC):
    """
    Alpha 产出 forecast (连续信号)，不直接下单。
    forecast 范围: [-1, +1]
      +1 = 最大看多
       0 = 中性 (无观点)
      -1 = 最大看空 (若允许做空)

    Alpha 通过 self.data (DataContext) 和 self.history (HistoryManager)
    访问所有数据，由引擎注入。
    """

    # ── 引擎注入 ──
    data: DataContext
    history: HistoryManager
    latest_prices: Dict[str, Bar]

    # ── 生命周期 ──
    def on_init(self):
        """初始化，可用于预热指标。"""

    def on_bar(self, bar: Bar):
        """
        逐 bar 回调。
        用于更新内部状态 (如均线、HHT 相位)。
        不返回值——forecast 在 on_market_close 中一次性产出。
        """

    @abstractmethod
    def on_market_close(self, dt: pd.Timestamp) -> Dict[str, float]:
        """
        日终回调，返回该 Alpha 对每个 symbol 的 forecast。
        返回: {symbol: forecast}
        只包含有观点的 symbol，无观点的 symbol 不出现在 dict 中。
        """

    def on_week_end(self, dt: pd.Timestamp) -> Optional[Dict[str, float]]:
        """周频 Alpha 可以 override 此方法。默认不产出。"""
        return None

    def on_month_end(self, dt: pd.Timestamp) -> Optional[Dict[str, float]]:
        """月频 Alpha 可以 override 此方法。默认不产出。"""
        return None
```

#### Alpha vs Strategy 的区别

| | Alpha | Strategy (v1) |
|---|---|---|
| **输出** | `Dict[str, float]` (连续 forecast) | `OrderEvent` (离散买卖指令) |
| **是否下单** | 否 | 是 |
| **是否持有仓位信息** | 否 (无状态) | 是 (`self.positions`) |
| **可组合性** | 可多个并行 → Combiner 合成 | 单一策略独占 |
| **可归因性** | 每个 Alpha 的 forecast 可独立追踪 | 只有最终 P&L |

#### 示例：把现有 HHT 择时改写为 Alpha

```python
class HHTTimingAlpha(Alpha):
    """
    Hilbert-Huang Transform 择时信号。
    相位在 [-π/2, π/2] → forecast = +1 (看多)
    其他相位 → forecast = 0 (中性)
    """
    def __init__(self, symbols, ma_period=60, ht_period=30):
        self.symbols = symbols
        self.ma_period = ma_period
        self.ht_period = ht_period

    def on_market_close(self, dt):
        forecasts = {}
        for sym in self.symbols:
            closes = self.history.get(sym, 'close', self.ma_period + self.ht_period)
            if len(closes) < self.ma_period + self.ht_period:
                continue
            # ... 计算 HHT 相位 ...
            phase = compute_hht_phase(closes, self.ma_period, self.ht_period)
            forecasts[sym] = 1.0 if -np.pi/2 <= phase <= np.pi/2 else 0.0
        return forecasts
```

#### 示例：Regime Alpha

```python
class RegimeAlpha(Alpha):
    """
    HMM regime 模型。
    输出 regime 权重 ∈ [0, 1]，作为其他 Alpha forecast 的缩放因子。
    """
    def on_market_close(self, dt):
        # 用 SPY 判断大盘 regime
        spy_closes = self.history.get('SPY', 'close', 252)
        regime_prob = self._hmm_predict(spy_closes)  # P(risk-on)

        # 对所有 symbol 返回统一的 regime 权重
        return {sym: regime_prob for sym in self.symbols}
```

---

### 4.3 ForecastCombiner — 信号合成器

#### 应对的场景

同一资产有多个 Alpha 产出 forecast，需要合并成一个综合 forecast 再决定仓位。

#### 设计

```python
class ForecastCombiner(ABC):
    """将多个 Alpha 的 forecast 合并为单一综合 forecast。"""

    @abstractmethod
    def combine(self,
                forecasts: List[Dict[str, float]],
                weights: List[float]
                ) -> Dict[str, float]:
        """
        参数:
          forecasts: 每个 Alpha 的输出 [{symbol: forecast}, ...]
          weights:   对应的权重 [w1, w2, ...]

        返回: {symbol: combined_forecast}
        combined_forecast ∈ [-1, +1]
        """
```

#### 内置实现

```python
class WeightedAvgCombiner(ForecastCombiner):
    """
    加权平均合成。最常用的方式。

    combined[sym] = clip(Σ(w_i × f_i[sym]) / Σw_i, -1, +1)

    适用场景: 多个同类 Alpha (如多个择时信号) 的等权/不等权混合。
    """

class MultiplicativeCombiner(ForecastCombiner):
    """
    乘性合成。

    combined[sym] = clip(f_1[sym] × f_2[sym] × ... , -1, +1)

    适用场景: Alpha 之间是"门控"关系，如:
      择时 forecast × regime 权重 = 最终 forecast
    当 regime = 0 (risk-off) 时，无论择时信号多强，combined = 0。
    """

class LayeredCombiner(ForecastCombiner):
    """
    分层合成: 先组内加权平均，再组间乘法。

    适用场景: 最典型的实际用法
      Layer 1 (加法): [HHT, QRS, MA] → timing_forecast (同类信号平均)
      Layer 2 (乘法): timing_forecast × regime_weight → final_forecast
    """
    def __init__(self, groups: List[List[int]], group_mode: List[str]):
        """
        groups: [[0,1,2], [3]] — Alpha 索引分组
        group_mode: ['avg', 'multiply'] — 组间合成方式
        """
```

#### 为什么不在 Alpha 内部做合成

- 合成策略是**正交于信号生成**的独立决策。同一个 HHT Alpha 可以和不同的 Regime Alpha 组合
- 合成权重可能需要动态调整 (如根据 IC 表现加权)，这不应该是 Alpha 的职责
- 独立的 Combiner 让用户可以方便地实验不同合成方式，而不改动任何 Alpha 代码

---

### 4.4 VolTargetSizer — 波动率目标仓位计算器

#### 应对的场景

forecast 是 [-1, +1] 的信号强度，需要转化为具体的目标持仓股数。仓位大小应与波动率成反比，以维持组合层面的风险恒定。

#### 设计

```python
class VolTargetSizer:
    """
    Rob Carver 式波动率目标仓位计算。

    target_position[sym] = (forecast × target_vol / realized_vol[sym])
                           × (capital / price[sym])

    其中:
      forecast:      ∈ [-1, +1]，由 Combiner 产出
      target_vol:    组合年化目标波动率 (如 15%)
      realized_vol:  个股已实现年化波动率 (如 rolling 20d std × √252)
      capital:       分配给该 symbol 的资金
    """

    def __init__(self,
                 target_vol: float = 0.15,
                 vol_lookback: int = 20,
                 max_leverage: float = 1.0):
        """
        参数:
          target_vol:    年化目标波动率 (默认 15%)
          vol_lookback:  已实现波动率回看窗口 (交易日)
          max_leverage:  最大杠杆倍数 (默认 1.0 = 不加杠杆)
        """

    def size(self,
             combined_forecasts: Dict[str, float],
             history: HistoryManager,
             capital: float,
             latest_prices: Dict[str, Bar]
             ) -> Dict[str, int]:
        """
        参数:
          combined_forecasts:  {symbol: forecast}
          history:             用于计算 realized vol
          capital:             总可用资金
          latest_prices:       当前价格 (用于换算股数)

        返回:
          {symbol: target_shares}  (正=多头, 负=空头, 0=平仓)
        """
```

#### 计算步骤

```
对于每个 symbol:
  1. returns = close[-vol_lookback:] 的日收益率
  2. realized_vol = std(returns) × √252          # 年化
  3. vol_scalar = target_vol / realized_vol       # 波动率缩放因子
  4. vol_scalar = min(vol_scalar, max_leverage)   # 杠杆上限
  5. raw_position = forecast × vol_scalar         # 风险调整后的仓位比例
  6. capital_per_sym = capital × abs(raw_position) / Σ|raw_position|  # 按比例分配资金
  7. target_shares = int(capital_per_sym / price × sign(raw_position))
```

#### 为什么不让策略自己算仓位

- 波动率目标是**组合层面**的决策，需要看到所有 symbol 的 forecast 才能做资金分配
- 不同策略对 realized vol 的计算口径不统一，会导致组合风险不可控
- 统一的 Sizer 可以方便地切换仓位算法（如加入风险平价、最大化 Sharpe 等）

---

### 4.5 CompositeStrategy — 组合策略编排器

#### 应对的场景

串联 Alpha → Combiner → Sizer → Order 的完整流程，同时保持与现有 Engine 的兼容。

#### 设计

```python
class CompositeStrategy(Strategy):
    """
    组合策略: 管理多个 Alpha，通过 Combiner 和 Sizer 产出最终订单。
    继承自 Strategy，对 Engine 而言和普通策略没有区别。
    """

    def __init__(self,
                 alphas: List[Alpha],
                 combiner: ForecastCombiner,
                 sizer: VolTargetSizer,
                 alpha_weights: Optional[List[float]] = None,
                 rebalance_freq: str = 'daily'):
        """
        参数:
          alphas:          Alpha 实例列表
          combiner:        ForecastCombiner 实例
          sizer:           VolTargetSizer 实例
          alpha_weights:   各 Alpha 权重 (None → 等权)
          rebalance_freq:  调仓频率 'daily' | 'weekly' | 'monthly'
        """

    # ── 引擎回调 ──

    def on_init(self):
        # 将 self.data, self.history 等注入到每个 Alpha
        for alpha in self.alphas:
            alpha.data = self.data
            alpha.history = self.history
            alpha.latest_prices = self.latest_prices
            alpha.on_init()

    def on_bar(self, bar: Bar):
        # 转发 bar 到每个 Alpha (更新内部状态)
        for alpha in self.alphas:
            alpha.on_bar(bar)

    def on_market_close(self, dt):
        if self.rebalance_freq != 'daily':
            return

        self._rebalance(dt)

    def on_week_end(self, dt):
        if self.rebalance_freq == 'weekly':
            self._rebalance(dt)

    def on_month_end(self, dt):
        if self.rebalance_freq == 'monthly':
            self._rebalance(dt)

    # ── 内部 ──

    def _rebalance(self, dt):
        # 1. 收集所有 Alpha 的 forecast
        forecasts = [alpha.on_market_close(dt) for alpha in self.alphas]

        # 2. 合成
        combined = self.combiner.combine(forecasts, self.alpha_weights)

        # 3. 计算目标仓位
        targets = self.sizer.size(
            combined, self.history,
            self.get_portfolio_value(), self.latest_prices
        )

        # 4. 差量下单
        self._rebalance_to_targets(targets)

    def _rebalance_to_targets(self, targets: Dict[str, int]):
        """根据目标股数与当前持仓的差异，生成 buy/sell 订单。"""
        all_symbols = set(list(targets.keys()) + list(self.positions.keys()))
        for sym in all_symbols:
            target = targets.get(sym, 0)
            current = self.get_position(sym)
            diff = target - current
            if diff > 0:
                self.buy(sym, diff)
            elif diff < 0:
                self.sell(sym, abs(diff))
```

#### CompositeStrategy 与 v1 Strategy 的关系

CompositeStrategy **继承自** Strategy，是 Strategy 的一个特化版本。Engine 对它没有特殊处理——它就是一个普通 Strategy，只是内部编排了 Alpha/Combiner/Sizer 的流程。

```
Strategy (v1 抽象基类)
   ├── BuyAndHoldStrategy     (直接 buy/sell，v1 用法)
   ├── HHTTimingStrategy      (直接 buy/sell，v1 用法)
   └── CompositeStrategy      (v2 用法: Alpha × N → Combiner → Sizer → buy/sell)
```

这意味着：
- **v1 策略完全不受影响**，不需要任何修改
- 新策略可以选择用 v1 方式 (简单直接) 或 v2 方式 (多信号合成)
- Engine 层改动最小：只是多注入一个 `self.data = DataContext`

---

## 5. 组件交互流程

### 5.1 场景 A：单资产 + 择时 + Regime (乘性合成)

```
用户意图: "AAPL 择时做多/平仓，但 risk-off 时缩仓"

Alpha 配置:
  [0] HHTTimingAlpha(symbols=['AAPL'])   → {AAPL: 1.0} 或 {AAPL: 0.0}
  [1] RegimeAlpha(symbols=['AAPL'])      → {AAPL: 0.8} (risk-on) 或 {AAPL: 0.2} (risk-off)

Combiner: MultiplicativeCombiner
  combined = HHT × Regime
  risk-on  且看多: 1.0 × 0.8 = 0.8
  risk-off 且看多: 1.0 × 0.2 = 0.2  ← 缩仓
  risk-on  但看空: 0.0 × 0.8 = 0.0  ← 空仓
```

### 5.2 场景 B：多资产截面 + 波动率目标 (加权合成)

```
用户意图: "50 只股票，截面因子选股 + 动量择时，年化波动率控制在 15%"

Alpha 配置:
  [0] FactorAlpha(symbols=sp500)         → {AAPL: +0.7, MSFT: -0.3, ...}
  [1] MomentumAlpha(symbols=sp500)       → {AAPL: +0.5, MSFT: +0.1, ...}

Combiner: WeightedAvgCombiner(weights=[0.6, 0.4])
  combined[AAPL] = 0.6×0.7 + 0.4×0.5 = 0.62
  combined[MSFT] = 0.6×(-0.3) + 0.4×0.1 = -0.14

Sizer: VolTargetSizer(target_vol=0.15)
  AAPL realized_vol = 25% → vol_scalar = 0.15/0.25 = 0.6
  AAPL raw_pos = 0.62 × 0.6 = 0.372  → 分配 37.2% 资金
  MSFT: forecast 太小，可能 skip (设 min_forecast 阈值)
```

### 5.3 场景 C：分层组合 (组内平均 + 组间相乘)

```
用户意图: "多个择时 Alpha 取平均，再乘以 Regime 缩放"

Alpha 配置:
  [0] HHTTimingAlpha   → {AAPL: 1.0}     ┐
  [1] QRSTimingAlpha   → {AAPL: 0.6}     ├─ 择时组 (加权平均)
  [2] MATimingAlpha    → {AAPL: 0.8}     ┘
  [3] RegimeAlpha      → {AAPL: 0.5}       ← Regime 组

Combiner: LayeredCombiner(
    groups=[[0,1,2], [3]],
    group_mode=['avg', 'multiply']
)

计算:
  timing_avg = (1.0 + 0.6 + 0.8) / 3 = 0.8
  final = timing_avg × 0.5 = 0.4
```

### 5.4 时序图：一个完整交易日

```
时间线 ──────────────────────────────────────────────────►

Engine.run_backtest():
│
├── bar(AAPL, 2025-06-02)
│   ├── execute pending orders (from 2025-05-30)
│   ├── Portfolio.update_market()
│   ├── HistoryManager._on_bar()
│   └── Strategy.on_bar()
│       └── CompositeStrategy → 转发给每个 Alpha.on_bar()
│
├── bar(MSFT, 2025-06-02)
│   └── (同上)
│
├── bar(GOOGL, 2025-06-02)
│   └── (同上)
│
├── [检测到日期边界: 2025-06-02 所有 bar 处理完毕]
│
├── Strategy.on_market_close(2025-06-02)
│   └── CompositeStrategy._rebalance():
│       ├── Alpha[0].on_market_close() → {AAPL: 0.8, MSFT: 0.3, GOOGL: -0.1}
│       ├── Alpha[1].on_market_close() → {AAPL: 0.5, MSFT: 0.5, GOOGL: 0.5}
│       ├── Combiner.combine()         → {AAPL: 0.4, MSFT: 0.15, GOOGL: -0.05}
│       ├── Sizer.size()               → {AAPL: 150, MSFT: 45, GOOGL: 0}
│       └── _rebalance_to_targets()    → buy(AAPL, 30), sell(GOOGL, 10), ...
│
└── [订单进入 pending queue，2025-06-03 执行]
```

---

## 6. v2 的已知局限

### 6.1 仅支持日频调仓

当前 Engine 的聚合回调粒度是 daily/weekly/monthly。若策略需要**盘中信号**（如 1min 级别的择时），Alpha 的 `on_market_close()` 接口不够。

**缓解**：Alpha 可以在 `on_bar()` 中更新内部状态，但最终 forecast 仍然在日终产出。对于大多数策略这足够了——高频交易不在此框架的目标范围内。

### 6.2 Forecast 是标量，不包含置信度

`forecast ∈ [-1, +1]` 丢失了不确定性信息。Alpha 可能对某些 symbol 有高置信度、对另一些低置信度，但 Combiner 无法区分。

**缓解**：可以在 Alpha 返回值中加入可选的置信度字段。但初期保持简单——标量 forecast 在实践中已经被 Man AHL、Winton 等量化机构验证过有效。

### 6.3 无跨资产约束

VolTargetSizer 按单个 symbol 独立计算仓位，没有考虑：
- 相关性（持有 5 只高相关科技股 ≈ 集中持仓）
- 行业/因子暴露限制
- 总杠杆约束

**缓解**：初期通过 `max_leverage` 做粗粒度控制。后续可以引入组合优化器（见未来方向）。

### 6.4 Alpha 之间无信息共享

每个 Alpha 独立运行，不能读取其他 Alpha 的 forecast。这是有意为之（保持独立性），但某些场景下 Alpha 之间有自然的依赖（如 factor timing: 根据 regime 动态调整因子权重）。

**缓解**：这类逻辑应放在 Combiner 层（动态权重），而非 Alpha 之间互通。

### 6.5 不支持多资产类别混合

当前的 Bar/price 体系假设所有 symbol 是同质的（股票）。债券、期货、期权有完全不同的定价模型和保证金规则。

**缓解**：跨资产类别的组合（如股债配置）应在更高层用独立的 Engine 实例跑，然后在外部合并权重。

### 6.6 回测级别的因果保证，不等于实盘保证

DataContext 的 `as_of(dt)` 防止了数据层面的前视偏差，但策略逻辑层面的偏差（如用未来信息做过滤/选股）仍需策略作者自觉。

---

## 7. 未来演进方向

### 7.1 组合优化器 (PortfolioOptimizer)

**应对的新问题**：当 symbol 数量增多（如 500 只），独立定仓会导致高相关性集中、因子暴露不可控。

**方向**：在 VolTargetSizer 之后加一层 PortfolioOptimizer：

```
Sizer 产出 raw_targets → Optimizer 加入约束 → final_targets
```

可选算法：
- 风险平价 (Risk Parity)
- 最小方差 (Min Variance)
- 最大分散化 (Max Diversification)
- 因子暴露约束的均值-方差优化

**何时需要**：当策略覆盖 50+ symbol 且对组合风险特征有明确要求时。

### 7.2 动态合成权重 (Adaptive Combiner)

**应对的新问题**：固定权重的 Combiner 无法适应市场状态变化。某些 Alpha 在趋势市表现好，另一些在震荡市表现好。

**方向**：

```python
class AdaptiveCombiner(ForecastCombiner):
    """根据滚动 IC/Sharpe 动态调整 Alpha 权重。"""
    def combine(self, forecasts, weights, alpha_history):
        rolling_ic = [calc_rolling_ic(f, returns, 60) for f in alpha_history]
        dynamic_weights = softmax(rolling_ic)
        return weighted_avg(forecasts, dynamic_weights)
```

**何时需要**：当有 3+ 个 Alpha 且表现轮动明显时。

### 7.3 交易成本模型 (TransactionCostModel)

**应对的新问题**：当前的 SimulatedExecutionHandler 假设零成本、零滑点、无限流动性。频繁调仓的策略会高估收益。

**方向**：

```python
class TransactionCostModel:
    """估算交易成本，供 Sizer 做 cost-aware 定仓。"""
    def estimate_cost(self, symbol, shares, adv) -> float:
        # 固定佣金 + 冲击成本 (与 ADV 的比例相关)
        commission = shares * per_share_cost
        impact = sigma * sqrt(shares / adv)
        return commission + impact
```

同时改进 Sizer：如果调仓的边际 alpha 不足以覆盖交易成本，则不调仓（引入换手惩罚）。

**何时需要**：当策略日/周频调仓且持仓标的数较多时。

### 7.4 Alpha 归因与诊断 (AlphaDiagnostics)

**应对的新问题**：组合 P&L 是多个 Alpha 的混合结果，需要分解每个 Alpha 的贡献。

**方向**：

```python
class AlphaDiagnostics:
    """记录每个 Alpha 的 forecast 历史，回测后分析。"""
    # 产出:
    # - 每个 Alpha 的 IC 时间序列
    # - Alpha 之间的相关性矩阵
    # - 每个 Alpha 对组合收益的边际贡献 (Shapley value / 子集分析)
    # - forecast 分布统计 (是否用满 [-1,+1] 范围)
```

**何时需要**：当组合有 3+ 个 Alpha 且需要理解哪些 Alpha 在贡献、哪些在拖累时。

### 7.5 实盘桥接 (LiveBridge)

**应对的新问题**：回测与实盘的数据源和执行路径不同，但信号逻辑应该复用。

**方向**：

```
回测模式:
  DataFeed=HistoricFeed + ExecutionHandler=Simulated

实盘模式:
  DataFeed=LiveFeed (IBKR/WebSocket) + ExecutionHandler=IBKRExecution
  DataContext: 同一套 panel，实时追加新数据

Strategy/Alpha/Combiner/Sizer: 完全不变
```

Engine 的主循环逻辑不变，只替换两端（数据源和执行器）。DataContext 的 panel 在实盘模式下通过定时任务追加最新数据。

**何时需要**：当策略通过回测验证后准备上线实盘时。

### 7.6 多周期引擎 (Multi-Frequency Engine)

**应对的新问题**：当前 Engine 是单一 bar stream (1d)。有些策略需要混合 1d + 1min：日线级别判断方向，分钟线级别择时入场。

**方向**：Engine 支持多个 DataFeed（不同频率），Alpha 可以声明自己需要哪个频率的 bar。引擎按时间戳合并不同频率的 bar stream。

**何时需要**：当策略明确需要多频率数据驱动决策（而非只是查询历史 1min 数据）时。目前 `history.panel_1min()` 已经覆盖了大部分"查一段 1min 历史做计算"的需求，Multi-Frequency Engine 面向的是"盘中实时响应 1min bar"的场景。

---

## 8. 附录：术语表

| 术语 | 定义 |
|------|------|
| **Bar** | 一根 OHLCV K 线，包含 timestamp, symbol, open, high, low, close, volume |
| **Forecast** | Alpha 对某资产的方向性观点，标准化到 [-1, +1] |
| **Alpha** | 产出 forecast 的独立信号单元，不直接下单 |
| **Combiner** | 将多个 Alpha 的 forecast 合成为单一综合 forecast |
| **Sizer** | 将综合 forecast 转化为目标持仓股数，考虑波动率目标 |
| **PIT (Point-in-Time)** | 在某时间点实际可获得的数据，避免使用未来数据 |
| **Vol Targeting** | 通过调整仓位大小使组合波动率维持在目标水平 |
| **Regime** | 市场状态 (如 risk-on/risk-off)，通常由 HMM 等模型判断 |
| **IC (Information Coefficient)** | forecast 与未来收益的 Spearman 秩相关系数，衡量 Alpha 的预测能力 |
| **TTM (Trailing Twelve Months)** | 最近四个季度的累计值，用于年化财务指标 |
| **ADV (Average Daily Volume)** | 日均成交额，衡量流动性 |
| **T+1 执行** | 今天产生的信号，明天（下一根 bar）才执行 |

---

## 变更日志

| 日期 | 版本 | 内容 |
|------|------|------|
| 2026-03-06 | v2 draft | 初稿：DataContext + Alpha + Combiner + Sizer + CompositeStrategy |
