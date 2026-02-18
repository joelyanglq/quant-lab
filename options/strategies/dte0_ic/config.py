from dataclasses import dataclass, field
from datetime import time


@dataclass
class ZeroDTEConfig:
    # --- Underlying ---
    symbol: str = "SPX"
    exchange: str = "CBOE"
    currency: str = "USD"

    # --- IBKR connection ---
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 21       # different from monthly_ic (20)

    # --- Entry trigger ---
    vix_min: float = 14.0     # only enter if VIX >= 14
    vix_max: float = 30.0     # skip if VIX > 30
    entry_time: time = field(default_factory=lambda: time(9, 45))
    entry_window_min: int = 15  # entry window: 9:45 ~ 10:00

    # --- Build rules ---
    target_delta: float = 0.06
    max_delta: float = 0.08
    wing_width: int = 10          # narrower wings for 0DTE
    short_put_min_dist: int = 30  # short put <= spot - 30
    short_call_min_dist: int = 30 # short call >= spot + 30
    min_credit: float = 0.50
    prefer_quarter_strikes: bool = False  # use 5-pt strikes freely
    min_risk_reward: float = 0.10  # credit / max_loss >= 10%

    # --- Exit rules ---
    profit_target_pct: float = 0.50   # close at 50% of credit
    stop_loss_mult: float = 2.0       # close if debit >= 2x credit
    eod_exit_time: time = field(default_factory=lambda: time(15, 45))

    # --- Risk red lines ---
    delta_red_line: float = 15.0
    proximity_red_line: float = 15.0

    # --- Execution ---
    default_qty: int = 1
    max_entries_per_day: int = 1
    poll_interval_sec: int = 45

    # --- RTH window (ET) ---
    rth_start: time = field(default_factory=lambda: time(9, 35))
    rth_end: time = field(default_factory=lambda: time(15, 55))

    # --- Logging ---
    log_dir: str = "logs/dte0_ic"
