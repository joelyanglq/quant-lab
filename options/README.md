# Options Strategy Execution Framework

SPX Iron Condor execution framework built on Interactive Brokers (`ib_insync`).
Supports multiple concurrent strategies, each running in its own thread with
independent risk management and audit logging.

## Quick Start

```bash
# Prerequisites: TWS or IB Gateway running on 127.0.0.1:7497 (paper)

# List available strategies
python -m options list

# Run one or more strategies
python -m options run dte0_ic
python -m options run monthly_ic dte0_ic    # run both
python -m options run dte0_ic --port 7496   # override to live

# View performance
python -m options summary dte0_ic
```

Press `Ctrl+C` to gracefully stop all running strategies.

## Architecture

```
options/
├── __main__.py              # Entry point
├── cli.py                   # CLI: list / run / status / summary
├── core/
│   ├── base.py              # Strategy ABC + state machine
│   ├── broker.py            # IBKR wrapper (market data, orders)
│   ├── logger.py            # JSON Lines trade/risk logger
│   └── manager.py           # Strategy registry + thread manager
└── strategies/
    ├── dte0_ic/             # 0DTE Iron Condor
    │   ├── config.py        # Parameters (dataclass)
    │   ├── scanner.py       # Entry trigger + leg selection
    │   ├── risk.py          # Risk monitoring + P&L exits
    │   └── strategy.py      # Poll loop orchestration
    └── monthly_ic/          # Monthly Iron Condor
        ├── config.py
        ├── scanner.py
        ├── risk.py
        └── strategy.py
```

### Core Components

| Module | Role |
|--------|------|
| `base.py` | Abstract `Strategy` class with lifecycle hooks: `setup` → `poll_once` → `teardown`. Thread-safe state machine (STOPPED → STARTING → RUNNING → STOPPING). |
| `broker.py` | Wraps `ib_insync` for VIX quotes, option chain scanning, delta lookup, and 4-leg BAG order placement. |
| `logger.py` | Append-only JSON Lines files (`trades.jsonl`, `risk_events.jsonl`). Provides `summary()` with win rate, PnL, max drawdown. |
| `manager.py` | `@register` decorator for plugin discovery. `StrategyManager` spawns/stops strategy threads. |

### Execution Flow

1. CLI calls `StrategyManager.start(name)` → spawns daemon thread
2. Thread runs `strategy._run()`: connect broker → `setup()` → polling loop → `teardown()`
3. Each `poll_once()` cycle follows priority order:
   - Forced exit (EOD / teardown)
   - Risk check (proximity + delta red lines)
   - P&L exit (profit target / stop loss)
   - New entry (if conditions met)

## Strategies

### 0DTE Iron Condor (`dte0_ic`)

Sells same-day-expiry iron condors on SPX within a narrow morning window.

| Parameter | Value | Description |
|-----------|-------|-------------|
| Entry window | 9:45–10:00 ET | Scan for entry only in this window |
| VIX band | 14–30 | Skip if VIX outside range |
| Target delta | 0.06 | Very OTM short strikes |
| Wing width | 10 pts | Narrow wings for 0DTE |
| Min distance | ±30 pts from spot | Short strike floor/ceiling |
| Min credit | $0.50 | Reject low-premium setups |
| Profit target | 50% of credit | Close when debit ≤ 0.5× credit |
| Stop loss | 2× credit | Close when debit ≥ 2× credit |
| EOD close | 3:45 PM ET | Force close via market order |
| Max entries | 1/day | Single position at a time |
| Delta red line | 15 | Market order exit if breached |
| Proximity red line | 15 pts | Market order exit if spot within range |
| Poll interval | 45s | |

### Monthly Iron Condor (`monthly_ic`)

Sells longer-dated iron condors on SPX triggered by VIX spikes.

| Parameter | Value | Description |
|-----------|-------|-------------|
| Entry trigger | VIX daily change ≥ 20% | Spike-based, not band-based |
| Min DTE | 49 days | Minimum time to expiry |
| Target delta | 0.10 | Wider exposure than 0DTE |
| Wing width | 25 pts | Wider protection |
| Min distance | put ±125 / call ±100 | Short strike floor/ceiling |
| Min credit | $3.00 | Higher premium requirement |
| Time exit | DTE ≤ 30 | Close as expiry approaches |
| Max positions | 3 concurrent | |
| Delta red line | 30 | Partial close (50% qty) |
| Proximity red line | 50 pts | Partial close (50% qty) |
| Poll interval | 300s (5 min) | |
| Prefer quarter strikes | Yes | Multiples of 25 for liquidity |

### Strategy Comparison

| Aspect | 0DTE | Monthly |
|--------|------|---------|
| Holding period | Intraday | ~19 days (49→30 DTE) |
| Entry trigger | Time window + VIX band | VIX spike ≥ 20% |
| Risk exit | Full close (market order) | Partial close (50% qty) |
| P&L exit | Profit target / stop loss | Time-based only |
| Concurrency | 1 position | Up to 3 positions |
| IBKR client_id | 21 | 20 |

## Adding a New Strategy

1. Create a new directory under `strategies/`:

```
strategies/
└── my_strategy/
    ├── __init__.py
    ├── config.py
    ├── scanner.py
    ├── risk.py
    └── strategy.py
```

2. Implement the `Strategy` base class:

```python
from options.core.base import Strategy

class MyStrategy(Strategy):
    name = "my_strategy"

    def make_broker(self):  ...
    def make_logger(self):  ...
    def setup(self, broker, logger):  ...
    def poll_once(self, broker, logger):  ...
    def teardown(self, broker, logger):  ...
    def poll_interval(self):  return 60
    def in_trading_hours(self):  ...
    def get_open_position_count(self):  ...
```

3. Register in `__init__.py`:

```python
from options.core.manager import register
from .strategy import MyStrategy
register(MyStrategy)
```

4. Import in `strategies/__init__.py`:

```python
from . import my_strategy  # noqa: F401
```

## Logging

Trade events use an **Order/Fill** model in JSON Lines format under each strategy's `log_dir`:

```
logs/
├── dte0_ic/
│   ├── trades.jsonl       # ORDER and FILL events
│   └── risk_events.jsonl  # RISK events
└── monthly_ic/
    ├── trades.jsonl
    └── risk_events.jsonl
```

### Event types in `trades.jsonl`

**ORDER** — records intent (order submitted):
```json
{"event":"ORDER","order_id":"a1b2c3d4e5f6","action":"open","expiry":"20260217",
 "short_put":5700,"long_put":5690,"short_call":5900,"long_call":5910,
 "limit_price":1.50,"qty":1,"reason":"","ts":"2026-02-17T09:46:12"}
```

**FILL** — records reality (order filled):
```json
{"event":"FILL","order_id":"a1b2c3d4e5f6","fill_price":1.45,
 "qty":1,"commission":2.60,"ts":"2026-02-17T09:46:18"}
```

Derived metrics (not stored, computed from ORDER+FILL pairs):
- **Slippage** = `fill_price - limit_price`
- **True P&L** = `(open_fill - close_fill) × 100 × qty - commissions`

### Summary

```bash
python -m options summary dte0_ic
# Output: total_trades, wins, losses, win_rate, gross_pnl, total_commission,
#         net_pnl, avg_pnl, max_win, max_loss, max_drawdown
```

## Requirements

- Python 3.10+
- `ib_insync` (Interactive Brokers API wrapper)
- TWS or IB Gateway running locally
- Port 7497 (paper) or 7496 (live)
