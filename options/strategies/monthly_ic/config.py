from dataclasses import dataclass, field
from datetime import time


@dataclass
class MonthlyICConfig:
    # --- Underlying ---
    symbol: str = "SPX"
    exchange: str = "CBOE"
    currency: str = "USD"

    # --- IBKR connection ---
    host: str = "127.0.0.1"
    port: int = 7497          # 7497=paper, 7496=live
    client_id: int = 20

    # --- Entry trigger ---
    vix_spike_pct: float = 0.20   # VIX daily change >= 20%

    # --- Build rules ---
    min_credit: float = 3.0
    target_delta: float = 0.10
    max_delta: float = 0.12
    min_dte: int = 49
    short_call_min_dist: int = 100
    short_put_min_dist: int = 125
    wing_width: int = 25
    prefer_quarter_strikes: bool = True

    # --- Exit rules ---
    exit_dte: int = 30

    # --- Risk red lines ---
    delta_red_line: float = 30.0
    proximity_red_line: float = 50.0

    # --- Execution ---
    max_positions: int = 3
    default_qty: int = 1
    poll_interval_sec: int = 300

    # --- RTH window (ET) ---
    rth_start: time = field(default_factory=lambda: time(9, 35))
    rth_end: time = field(default_factory=lambda: time(15, 45))

    # --- Logging ---
    log_dir: str = "logs/monthly_ic"
