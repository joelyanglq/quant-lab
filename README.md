# Quant Lab

A modular quantitative finance research platform covering data infrastructure, factor research, event-driven backtesting, trading advisory, and fixed-income analytics.

## Project Structure

```
quant-lab/
├── OneBacktest/                   # Equity research & trading platform
│   ├── data/                      #   Data layer (ETL + storage + feed)
│   ├── backtest/                  #   Event-driven backtesting engine
│   ├── strategy/                  #   Strategy framework + examples
│   ├── execution/                 #   Order execution simulation
│   ├── factor_research/           #   Multi-factor alpha pipeline
│   ├── advisor/                   #   Trading signal advisor (HHT/QRS + IBKR)
│   └── event.py                   #   Core event definitions
├── yield-curve-construction/      # Fixed-income: Treasury yield curve
├── option_basics/                 # Option analytics (IV surface)
├── data/                          # Shared data directory
│   ├── processed/bars_1d/         #   Daily OHLCV (Parquet, year-based)
│   ├── processed/bars_1min/       #   1-min OHLCV (Parquet, per-symbol)
│   └── fundamentals/              #   Income / Balance / CashFlow
└── output/                        # Reports, charts, signals
```

---

## 1. Data Layer (`OneBacktest/data/`)

Year-based Parquet storage with multi-source ETL pipelines for US equities.

| Module | Description |
|--------|-------------|
| `storage/parquet.py` | `ParquetStorage` — read/write year-partitioned Parquet files |
| `types.py` | `Bar`, `Frequency`, `AdjustType` data structures |
| `feed.py` | `DataFeed` abstract interface (iterator pattern) |
| `sources/historic.py` | `HistoricFeed` — heapq-based multi-symbol chronological feed |

### ETL Pipelines

| Script | Source | Data | Rate Limit |
|--------|--------|------|------------|
| `etl/daily_update.py` | Massive API | 1d bars, ~12K symbols | 1 call/day (all symbols) |
| `etl/twelvedata_1min.py` | Twelve Data | 1min bars, S&P500+NDX100 | 800 calls/day, 8/min |
| `etl/alphavantage_fundamentals.py` | Alpha Vantage | Income/Balance/CashFlow | 25 calls/day (~8 symbols) |

```bash
cd OneBacktest
python -m data.etl.daily_update              # Daily 1d + 1min update
python -m data.etl.daily_update --status     # Check progress
```

---

## 2. Event-Driven Backtesting Engine (`OneBacktest/backtest/`)

Bar-by-bar backtesting engine with T+1 execution delay, event queue architecture, and immutable event objects.

### Event Flow

```
DataFeed (heapq)          Strategy               Portfolio
     │                       │                       │
     ├─ Bar(T) ─────────────►│                       │
     │                       ├─ on_bar() ────► buy/sell OrderEvent
     │                       │                       │
     ├─ Bar(T+1) ──► execute pending orders          │
     │               └─► FillEvent ─────────────────►│ update_fill()
     │                       │◄── on_fill() ─────────┤
     │                       │               update_market() → snapshot
```

### Key Classes

| Class | File | Role |
|-------|------|------|
| `BacktestEngine` | `backtest/engine.py` | Orchestrates event loop, T+1 execution |
| `Strategy` (ABC) | `strategy/base.py` | `on_init()`, `on_bar()`, `on_fill()`, `buy()`, `sell()` |
| `Portfolio` | `strategy/portfolio.py` | Position tracking, equity curve, trade log |
| `SimulatedExecutionHandler` | `execution/handler.py` | Order → Fill at close price |
| `calc_metrics()` | `backtest/analytics.py` | Sharpe, Sortino, MaxDD, Calmar, win rate, etc. |

### Example Strategies

| Strategy | File | Description |
|----------|------|-------------|
| Buy & Hold | `strategy/examples/buy_and_hold.py` | Baseline benchmark |
| Simple MA | `strategy/examples/simple_ma.py` | 50/200 MA crossover |
| Momentum Vol-Sized | `strategy/examples/momentum_vol_sized.py` | Momentum with volatility sizing |
| RSI Reversal | `strategy/examples/rsi_reversal.py` | RSI mean-reversion |
| HHT Timing | `strategy/examples/hht_timing.py` | Hilbert-Huang Transform phase |
| QRS Timing | `strategy/examples/qrs_timing.py` | Quantitative Resistance-Support |

```bash
cd OneBacktest
python run_example.py    # Compare all strategies on sample symbols
```

---

## 3. Multi-Factor Alpha Pipeline (`OneBacktest/factor_research/`)

End-to-end weekly alpha pipeline: 46 factors across 11 categories, automated screening, walk-forward ML, and live signal generation for 500+ US equities.

### Pipeline Overview

```
46 factors (1d + 1min) → weekly alignment → IC screening → correlation filter
→ ~26 selected factors → cross-sectional z-score → walk-forward ML → live signal
```

### Factor Library (`factors.py`)

| Category | Factors | Data | Source |
|----------|---------|------|--------|
| Technical | RS_6M, RS_12M, RSI_16W, RSI_28W, Range_52W | 1d | Price momentum & mean-reversion |
| Reversal | interday/intraday/overnight_rev_volflip, team_coin | 1d | Volatility-flip reversal |
| Fundamental | ROE, ROIC, EPS_Score, PS, EV_EBITDA, FCF_Yield, FCF_Growth, Growth_Stability | 1d + financials | Value & quality |
| Microstructure | go_with_flow, lone_goose, sailing | 1min | Intraday price-volume patterns |
| Battle | vol_battle_ret/pos, amp_battle, vol_battle, bull_bear_battle | 1min | Bull-bear volume dynamics |
| Surge | weekly_dazzling_vol/ret, moderate_risk | 1min | Intraday surge detection |
| Regression | morning_mist, noon_shade, night_frost, flower_hidden | 1min | Intraday regression coefficients |
| Fuzzy | fuzzy_corr, fuzzy_amount_ratio, fuzzy_volume_ratio | 1min | Price-volume correlation |
| Rebuild | disaster_rebuild, peak_climbing | 1min | OHLCV structure rebuild |
| Tidal | full_tidal, strong/weak_half_tidal, stable_weak_half | 1min | Tidal force decomposition |
| Jump | weekly_jump, modified_amplitude_1/2, modified_amplitude, moth_to_flame | 1min | Taylor expansion jump degree |

### Screening & ML

| Step | Method |
|------|--------|
| IC filter | Keep factors with \|Rank IC\| >= 0.005 |
| Correlation filter | Drop if pairwise corr > 0.7 (keep higher ICIR) |
| Normalization | Per-date cross-sectional MAD-winsorize + z-score |
| Models | equal_weight, Ridge, Random Forest, PCA+Ridge |
| Validation | Walk-forward (expanding window, 1-week step) |

### Current Performance (517 symbols, 2025-10 ~ 2026-02)

| Model | L/S Sharpe | WinRate | MaxDD | LO Annual | LO Sharpe |
|-------|-----------|---------|-------|-----------|-----------|
| equal_weight | 1.68 | 64.0% | -4.31% | 52.1% | 2.33 |
| Random Forest | 1.54 | 58.0% | -4.78% | 37.9% | 1.84 |

### Supporting Modules

| Module | Role |
|--------|------|
| `data_loader.py` | Load index symbols, daily price panels, fundamentals |
| `ranking.py` | `cross_sectional_zscore`, `assign_quantiles` |
| `backtest.py` | Quantile L/S backtest, IC/ICIR/monotonicity |
| `analytics.py` | Factor-level performance metrics |
| `plotting.py` | Quantile return charts, IC bar plots |
| `weekly_pipeline.py` | All-in-one: compute → screen → ML → backtest → live signal |

```bash
cd OneBacktest
python -m factor_research.weekly_pipeline                # Full pipeline (~5 min)
python -m factor_research.weekly_pipeline --no-backtest  # Signal only
python -m factor_research.weekly_pipeline --n-symbols 50 # Quick test
```

---

## 4. Trading Advisor (`OneBacktest/advisor/`)

Combines HHT + QRS strategy signals with IBKR live positions to generate actionable trading recommendations.

| Module | Role |
|--------|------|
| `signal_engine.py` | Compute HHT phase signals and QRS z-score signals |
| `ibkr_reader.py` | Read-only IBKR connection (TWS/Gateway, paper/live) |
| `comparator.py` | Signal vs position → BUY / SELL / HOLD / REVIEW recommendations |

```bash
cd OneBacktest
python advisor_cli.py                         # Full advisor run
python advisor_cli.py --offline               # Without IBKR connection
```

---

## 5. Yield Curve Construction (`yield-curve-construction/`)

US Treasury yield curve bootstrapping with multiple interpolation methods and bond pricing.

| Module | Description |
|--------|-------------|
| `curves/bootstrapping/` | Bootstrap zero-coupon curve from Treasury prices |
| `curves/interpolation/` | Log-linear, Cubic Spline, Nelson-Siegel-Svensson |
| `curves/instruments/` | T-Bill, T-Note, T-Bond with cashflow generation |
| `pricing/bond_pricer.py` | Bond present value calculation |
| `pricing/z_spread.py` | Z-Spread solve and pricing |

```bash
cd yield-curve-construction
python scripts/build_curve.py -i data/treasuries_2018-12-28.parquet -d 2018-12-28
python scripts/visualize_curve.py -i data/treasuries_2018-12-28.parquet -d 2018-12-28
```

---

## 6. Option Analytics (`option_basics/`)

Implied volatility surface construction and visualization (Jupyter notebook).

---

## Tech Stack

- **Language**: Python 3.12
- **Data**: Parquet (pyarrow), pandas, numpy
- **ML**: scikit-learn (Ridge, RandomForest, PCA)
- **Signal Processing**: scipy (Hilbert transform, regression)
- **Broker**: ib_insync (IBKR TWS/Gateway)
- **APIs**: Massive, Twelve Data, Alpha Vantage
