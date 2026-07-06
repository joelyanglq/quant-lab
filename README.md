# Quant Lab

A modular quantitative finance research & trading platform covering data infrastructure, systematic trading (Carver-style), ETF valuation monitoring, factor research, and fixed-income analytics.

## Project Structure

```
quant-lab/
├── trading_platform/              # ★ Active: Carver-style multi-strategy trading framework
│   ├── core/                      #   Engine, events, clock, DataContext (PIT-safe)
│   ├── data/                      #   DataFeed ABC, BacktestFeed, LiveDataFeed
│   ├── strategy/                  #   Alpha ABC, Combiner, RiskSizer, CompositeStrategy
│   │   └── archetypes/            #   4 strategy types: timing, cross_section, pairs, rotation
│   ├── execution/                 #   Simulated, Shadow, Live IBKR
│   ├── risk/                      #   Kill-switch, reconciliation, monitoring
│   ├── runtime/                   #   Backtest & live entry points
│   ├── analytics/                 #   Sharpe + CI + Carver metrics
│   └── templates/                 #   Strategy templates (copy & modify)
│
├── data_sync/                     # ★ Active: Unified data sync CLI (FMP-based)
│   ├── cli.py                     #   python -m data_sync sync / status / dry-run
│   ├── providers/                 #   prices, financials, analysts, universe, treasury
│   ├── client/fmp.py              #   FMPClient (rate limit, retry)
│   └── storage/parquet.py         #   ParquetStorage (year/month sharding)
│
├── etf_valuation/                 # ★ Active: Bottom-up ETF valuation monitoring
│   ├── run_backfill.py            #   7-step pipeline: holdings → ratios → compute → charts
│   ├── fetcher.py                 #   FMPHoldingsFetcher + RatiosBulkFetcher
│   ├── metrics.py                 #   Weighted harmonic/arithmetic aggregation
│   ├── history.py                 #   N-PORT historical reconstruction (27Q)
│   ├── report_charts.py           #   Time-series trajectory charts (per-tier PNGs)
│   └── README.html                #   Full documentation (open in browser)
│
├── OneBacktest/                   # ⚠ Deprecated: legacy backtest engine & strategies
│   ├── data/etl/                  #   Legacy ETL (use data_sync instead)
│   ├── backtest/                  #   Event-driven engine (superseded by trading_platform)
│   ├── strategies/                #   HHT/QRS timing, cross-section, multi-strategy
│   └── strategy/                  #   v2 Alpha/Combiner/Sizer (migrated to trading_platform)
│
├── yield-curve-construction/      # Fixed-income: Treasury yield curve bootstrapping
├── options/                       # SPX Iron Condor via IBKR
└── data/                          # Local data (D: drive, legacy storage)
```

---

## 1. Trading Platform (`trading_platform/`)

Carver-style (《Systematic Trading》) multi-strategy framework. Backtest and IBKR live trading share the same strategy code.

### Three-Layer Architecture

```
Alpha (forecast)     →  Combiner           →  RiskSizer           →  Execution
 forecast(dt)            weighted/layered       vol target              backtest
 ∈ [-20, +20]            handcrafted            half-Kelly              shadow / paper / live
                         NaN-aware              drawdown scaling
```

### Four Strategy Archetypes

| Archetype | File | Description |
|-----------|------|-------------|
| Single-name timing | `archetypes/single_name_timing.py` | HHT / QRS / MA / RSI / momentum |
| Cross-section | `archetypes/cross_section.py` | Multi-factor (sector-neutral, expanding-window) |
| Pairs | `archetypes/pairs.py` | Cointegration-based statistical arbitrage |
| Rotation | `archetypes/rotation.py` | Sector ETF rotation (12-1M momentum + ERC) |

### Usage

```bash
# Backtest
python -m trading_platform.runtime.backtest --strategy timing --symbols AAPL,MSFT --rule HHT

# Shadow mode (IBKR data, no orders)
python -m trading_platform.runtime.live --mode shadow --strategy timing --symbols AAPL

# Paper trading (IBKR paper account)
python -m trading_platform.runtime.live --mode paper --port 7497 --strategy timing --symbols AAPL
```

See `trading_platform/README.md` for full documentation.

---

## 2. Data Sync (`data_sync/`)

Unified data synchronization CLI built on FMP API (Ultimate plan). Stores to `E:/stocks/`.

| Provider | Data | Storage |
|----------|------|---------|
| `prices` | 1d EOD bulk + 1min per-symbol | Year-sharded / month-sharded Parquet |
| `financials` | Income / Balance / CashFlow (PIT-safe) | Per-symbol Parquet |
| `analysts` | Analyst estimates (EPS/Revenue consensus) | Per-symbol Parquet |
| `universe` | Stock list, profiles, index constituents | JSON + Parquet |
| `treasury` | US Treasury rates (all maturities) | Single Parquet |

```bash
python -m data_sync status                       # Check sync state
python -m data_sync sync                         # Incremental sync (all providers)
python -m data_sync sync --only prices_1d        # Sync daily bars only
python -m data_sync sync --only prices_1min      # Sync 1-min bars (Russell 1000)
```

---

## 3. ETF Valuation (`etf_valuation/`)

Bottom-up ETF valuation monitoring system. Aggregates ETF holdings × stock ratios → weighted metrics → historical percentiles → timing signals.

- **25 ETFs**: SPY + 11 GICS sectors + 13 themes
- **7 metrics**: P/E TTM, P/B, P/S, Div Yield, ERP, FCF Yield, EV/EBITDA
- **Holdings**: FMP `/stable/etf/holdings` (daily, real-time weights)
- **History**: 27 quarters reconstructed from SEC N-PORT (2019Q3→2026Q1)
- **Output**: CSV/JSON reports + per-tier time-series trajectory charts

```bash
python -m etf_valuation.run_backfill             # Full pipeline (~0.6 min)
python -m etf_valuation.history                  # Historical reconstruction (one-time)
python -m etf_valuation.mean_reversion           # Mean reversion tests
```

See `etf_valuation/README.html` for full documentation (open in browser).

---

## 4. OneBacktest (Deprecated)

> **Note**: `OneBacktest/` is the legacy backtest engine and strategy library. It has been superseded by `trading_platform/` which implements a proper Carver three-layer architecture with forecast protocol, risk sizing, and IBKR live trading support.
>
> **What's still useful in OneBacktest**:
> - `strategies/timing/advisor/` — HHT+QRS live advisory CLI (IBKR integration)
> - `strategies/cross_section/` — 46-factor research pipeline, pattern event signals
> - `strategies/multi/` — Multi-strategy composite backtest reference
>
> **Use `data_sync` instead of** `OneBacktest/data/etl/` for all data synchronization.

---

## 5. Other Modules

| Module | Description |
|--------|-------------|
| `yield-curve-construction/` | US Treasury yield curve bootstrapping (Nelson-Siegel-Svensson) |
| `options/` | SPX Iron Condor strategy via IBKR |

---

## Tech Stack

- **Language**: Python 3.12
- **Data**: Parquet (pyarrow), pandas, numpy
- **APIs**: FMP (primary), FRED, SEC EDGAR
- **Broker**: ib_insync (IBKR TWS/Gateway)
- **ML**: scikit-learn (Ridge, RandomForest, PCA)
- **Signal Processing**: scipy (Hilbert transform, regression)
