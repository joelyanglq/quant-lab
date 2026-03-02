---
name: factor_research
description: factor_research 包的架构说明和模块文档。涵盖目录结构、因子分类、回测框架、周频 ML 管线和 CLI 入口。
---

# factor_research

多因子横截面研究框架。独立于 OneBacktest 回测引擎，专注于因子挖掘、筛选和组合构建。

## 目录结构

```
factor_research/
├── __init__.py
├── data_loader.py          # 数据加载 (价格面板 + Point-in-Time 基本面)
├── factors.py              # 46 因子计算 (11 大类, 1d + 1min)
├── screening.py            # 因子筛选 (IC/相关性过滤) + 正交化
├── ranking.py              # 横截面 z-score + 分位组分配
├── composite.py            # 等权/加权合成因子
├── backtest.py             # 单因子月频回测 (五分位 + IC)
├── analytics.py            # 绩效指标 (Sharpe/IC/ICIR/回撤/换手)
├── plotting.py             # 四面板可视化
├── run_factor_backtest.py  # CLI: 月频因子回测
└── weekly_pipeline.py      # CLI: 周频因子筛选 + walk-forward ML
```

## 数据加载

`data_loader.py` — 独立的数据加载层，从磁盘读取 Parquet 数据。

```python
from factor_research.data_loader import load_index_symbols, load_price_panel

symbols = load_index_symbols()  # S&P500 + NASDAQ100 成分股
panels = load_price_panel(symbols, start='2019-01-01', end='2026-12-31')
# panels: {'open': df, 'close': df, 'high': df, 'low': df, 'volume': df}
#   每个 df: index=DatetimeIndex, columns=symbols
```

基本面加载 (Point-in-Time，避免前视偏差)：

```python
from factor_research.data_loader import build_fundamental_panel

panels = build_fundamental_panel(symbols, fields, trading_dates)
# flow 项 (利润表/现金流) → TTM 滚动4季求和
# stock 项 (资产负债表) → 取最新值
# 按 filing_date 前向填充到交易日
```

## 因子库

`factors.py` — 46 个因子，分 11 大类。

### 1d 因子 (入参为日线面板)

| 分类 | 因子数 | 函数 | 输出 |
|------|--------|------|------|
| 技术 | 5 | `compute_technical_factors(close, high, low)` | RS_12M, RS_6M, Range_52W, RSI_28W, RSI_16W |
| 反转 | 4 | `compute_reversal_factors(close, open_price)` | interday/intraday/overnight_rev_volflip, team_coin |
| 基本面 | 8 | `compute_fundamental_factors(symbols, close)` | ROE, ROIC, EV_EBITDA, FCF_Yield, PS, FCF_Growth, EPS_Score, Growth_Stability |

### 1min 因子 (内部从 ParquetStorage 加载分钟线)

| 分类 | 因子数 | 函数 | 输出 |
|------|--------|------|------|
| 微观结构 | 3 | `compute_microstructure_factors(symbols, start, end)` | go_with_flow, lone_goose, sailing |
| 博弈 | 5 | `compute_battle_factors(symbols, start, end)` | vol_battle_ret/pos, amp_battle, vol_battle, bull_bear_battle |
| 激增 | 3 | `compute_surge_factors(symbols, start, end)` | weekly_dazzling_vol/ret, moderate_risk |
| 回归 | 4 | `compute_regression_factors(symbols, start, end)` | morning_mist, noon_shade, night_frost, flower_hidden |
| 模糊 | 3 | `compute_fuzzy_factors(symbols, start, end)` | fuzzy_corr, fuzzy_amount_ratio, fuzzy_volume_ratio |
| 灾后重建 | 2 | `compute_rebuild_factors(symbols, start, end)` | disaster_rebuild, peak_climbing |
| 潮汐 | 4 | `compute_tidal_factors(symbols, start, end)` | full_tidal, strong_half_tidal, aggressive_weak_half, stable_weak_half |
| 跳跃 | 5 | `compute_jump_factors(symbols, start, end)` | weekly_jump, modified_amplitude_1/2, modified_amplitude, moth_to_flame |

> 注: `factor_research/factors.py` 中的 1min 因子函数自行加载数据（传入 symbols + 时间范围），与 `strategy/factors.py` 的纯面板入参设计不同。

## 因子筛选与正交化

`screening.py` — 独立的因子筛选模块，被 `weekly_pipeline.py` 调用，也可单独使用。

### IC 分析

```python
from factor_research.screening import compute_rank_ic, compute_ic_summary

# 单因子 IC 序列
ic_series = compute_rank_ic(factor_w, fwd_ret_w, min_obs=20)
# Series(index=dates, values=spearman_ic)

# 批量 IC 统计
ic_df = compute_ic_summary(factors_w, fwd_ret_w)
# DataFrame(index=factor_name, columns=[mean_ic, ic_std, icir, n_periods])
```

### 筛选

```python
from factor_research.screening import ic_filter, correlation_dedup, screen_factors

# 分步调用
passed = ic_filter(ic_df, min_abs_ic=0.005)           # |IC| 过滤
selected = correlation_dedup(factors_w, ic_df, passed, max_corr=0.7)  # 相关性去冗余

# 一步到位 (等价于上面两步)
selected, ic_df = screen_factors(factors_w, fwd_ret_w,
                                  min_abs_ic=0.005, max_corr=0.7)
```

### 正交化

```python
from factor_research.screening import orthogonalize, orthogonalize_sequential

# 单因子正交化: 逐截面 OLS 回归取残差
pure_factor = orthogonalize(new_factor, existing_factors)
# new_factor[t] = β₀ + Σβᵢ·existing_i[t] + ε[t], 返回 ε

# 顺序正交化 (Gram-Schmidt): 按优先级逐个正交化
# order 通常按 |ICIR| 降序 — 高 ICIR 因子保持原样, 低 ICIR 因子去除冗余
ortho_factors = orthogonalize_sequential(factors, order=['RS_12M', 'ROE', ...])
```

## 横截面处理

`ranking.py`:

```python
from factor_research.ranking import cross_sectional_zscore, assign_quantiles

z = cross_sectional_zscore(factor_df, winsorize_sigma=3.0)  # MAD winsorize → z-score
q = assign_quantiles(factor_df, n_quantiles=5)              # 1=worst, 5=best
```

`composite.py` — 加权合成：

```python
from factor_research.composite import build_composite_factor

composite = build_composite_factor(factor_dict, weights=DEFAULT_WEIGHTS)
# DEFAULT_WEIGHTS: ROE 12.5%, ROIC 12.5%, EV_EBITDA 10%, FCF_Yield 10%, ...
```

## 单因子回测

`backtest.py` — 提供周频/月频/自定义周期的因子回测接口。

```python
from factor_research.backtest import (
    run_factor_backtest,           # 核心回测引擎 (频率无关)
    build_monthly_rebalance,       # 月末调仓
    build_weekly_rebalance,        # 周五调仓
    build_periodic_rebalance,      # 通用周期调仓
)

# ── 方式 1: 核心引擎 (手动准备数据) ──
result = run_factor_backtest(factor_df, forward_returns, n_quantiles=5)

# ── 方式 2: 月频回测 (月末调仓 → T+1 执行 → 持有 21 天) ──
result = build_monthly_rebalance(factor, close, holding_period=21)

# ── 方式 3: 周频回测 (周五调仓 → T+1 执行 → 持有 5 天) ──
result = build_weekly_rebalance(factor, close, holding_period=5)

# ── 方式 4: 通用周期回测 (自定义频率) ──
result = build_periodic_rebalance(
    factor, close,
    rebalance_freq='W-FRI',  # 'W-FRI', 'M', 'Q', etc.
    holding_period=None,     # None = 自动推断 (W:5天, M:21天, Q:63天)
)

# result 包含:
#   'quantile_returns': DataFrame(index=dates, columns=[1..5])
#   'long_short': Series (Q5 - Q1)
#   'ic_series': Series (每期 Spearman Rank IC)
#   'quantiles': DataFrame (分组结果)
```

`analytics.py`:

```python
from factor_research.analytics import compute_factor_metrics, format_metrics

metrics = compute_factor_metrics(result)
# metrics: {ls_annual_return, ls_sharpe, rank_ic_mean, rank_icir,
#           monotonicity, max_drawdown, turnover, ...}
print(format_metrics(metrics, 'ROE'))
```

`plotting.py`:

```python
from factor_research.plotting import plot_factor_report

plot_factor_report(result, metrics, factor_name='ROE', save_path='output/ROE.png')
# 四面板: (1) 五分位累计收益 (2) 多空净值+指标 (3) IC 时序 (4) 分位年化收益柱状图
```

## CLI: 月频因子回测

```bash
cd OneBacktest
python -m factor_research.run_factor_backtest --start 2019-01-01 --end 2025-12-31
python -m factor_research.run_factor_backtest --factor ROE --no-plot
python -m factor_research.run_factor_backtest --n-symbols 100 --save-dir output/
```

流程: 加载数据 → 计算 13 因子 (8 基本面 + 5 技术) → 逐因子月频回测 → 合成因子 → 绘图。

## CLI: 周频 ML 管线

`weekly_pipeline.py` — 因子筛选 + walk-forward 机器学习预测。

```bash
cd OneBacktest
python -m factor_research.weekly_pipeline
python -m factor_research.weekly_pipeline --n-pca 5 --min-train 8 --top-n 30
python -m factor_research.weekly_pipeline --no-backtest  # 只输出 live signal
```

流程:

```
compute_all_factors    46 因子 (11 大类)
        │
        ▼
prepare_weekly         对齐到 W-FRI, 计算周前瞻收益
        │
        ▼
screen_factors         IC 过滤 + 相关性过滤 (来自 screening.py)
        │
        ▼
build_panel_data       (week, symbol) 长面板, z-score 标准化
        │
        ▼
walk_forward_predict   Ridge / RandomForest / XGBoost / LightGBM, expanding window
        │
        ▼
backtest_predictions   五分位回测 + long-only top_n
        │
        ▼
generate_live_signal   最新一周预测 → top_n 持仓建议
```

模型:

| 模型 | 库 | 默认 |
|------|------|------|
| Ridge | sklearn | 始终使用 |
| RandomForest | sklearn | 始终使用 |
| XGBoost | xgboost | 可选 (未安装则跳过) |
| LightGBM | lightgbm | 可选 (未安装则跳过) |
| average | — | 所有模型等权平均 |

## 与 OneBacktest 的关系

两个框架独立运作，不存在 import 依赖。

- `factor_research` 专注**因子研究**: 横截面排名、IC 分析、walk-forward ML。数据通过自己的 `data_loader.py` 加载。
- `OneBacktest` 专注**策略回测**: event-driven 引擎、持仓管理、T+1 执行。`strategy/factors.py` 中有一份适配回测框架的因子库副本（纯面板入参，不自行加载数据）。
