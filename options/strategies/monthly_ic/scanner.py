"""Entry trigger (VIX spike) and condor leg selection."""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Optional

from ib_insync import Option

from options.core.broker import Broker, CondorLegs
from .config import MonthlyICConfig

log = logging.getLogger(__name__)


def check_entry_trigger(broker: Broker, cfg: MonthlyICConfig) -> bool:
    """Return True if VIX spike exceeds threshold."""
    current, prev_close = broker.get_vix()
    if prev_close is None or prev_close <= 0 or math.isnan(prev_close):
        log.warning("VIX prev_close unavailable")
        return False
    if current is None or math.isnan(current):
        log.warning("VIX current price unavailable")
        return False

    change = (current - prev_close) / prev_close
    log.info("VIX  current=%.2f  prev_close=%.2f  change=%.1f%%",
             current, prev_close, change * 100)
    return change >= cfg.vix_spike_pct


def _pick_expiry(broker: Broker, cfg: MonthlyICConfig) -> Optional[str]:
    """Pick the nearest expiry with DTE >= min_dte."""
    expirations = broker.get_chain_expirations()
    today = datetime.now().date()
    for exp_str in expirations:
        exp_date = datetime.strptime(exp_str, "%Y%m%d").date()
        dte = (exp_date - today).days
        if dte >= cfg.min_dte:
            log.info("Selected expiry %s  DTE=%d", exp_str, dte)
            return exp_str
    log.warning("No expiry found with DTE >= %d", cfg.min_dte)
    return None


def _candidate_strikes(spot: float, cfg: MonthlyICConfig,
                       side: str) -> list[float]:
    """Generate candidate strikes for short leg scanning."""
    if side == "put":
        center = spot - cfg.short_put_min_dist
        low = center - 100
        high = center + 25
    else:
        center = spot + cfg.short_call_min_dist
        low = center - 25
        high = center + 100

    low = int(low // 5) * 5
    high = int(math.ceil(high / 5)) * 5

    strikes = list(range(low, high + 1, 5))
    if cfg.prefer_quarter_strikes:
        strikes = [s for s in strikes if s % 25 == 0]
    return [float(s) for s in strikes]


def find_condor_legs(broker: Broker, cfg: MonthlyICConfig) -> Optional[CondorLegs]:
    """Scan option chain and return legs satisfying all build rules."""
    spot = broker.get_underlying_price()
    if spot is None or math.isnan(spot):
        log.error("Cannot get underlying price")
        return None
    log.info("%s spot = %.2f", cfg.symbol, spot)

    expiry = _pick_expiry(broker, cfg)
    if not expiry:
        return None

    # --- find short put ---
    put_strikes = _candidate_strikes(spot, cfg, "put")
    put_contracts = [
        Option(cfg.symbol, expiry, s, "P", "SMART") for s in put_strikes
    ]
    put_deltas = broker.get_option_delta(put_contracts)

    best_put_strike = None
    best_put_diff = 999.0
    for c in put_contracts:
        d = put_deltas.get(c.conId)
        if d is None:
            continue
        ad = abs(d)
        if ad > cfg.max_delta:
            continue
        if c.strike > spot - cfg.short_put_min_dist:
            continue
        diff = abs(ad - cfg.target_delta)
        if diff < best_put_diff:
            best_put_diff = diff
            best_put_strike = c.strike

    if best_put_strike is None:
        log.warning("No valid short put found")
        return None

    # --- find short call ---
    call_strikes = _candidate_strikes(spot, cfg, "call")
    call_contracts = [
        Option(cfg.symbol, expiry, s, "C", "SMART") for s in call_strikes
    ]
    call_deltas = broker.get_option_delta(call_contracts)

    best_call_strike = None
    best_call_diff = 999.0
    for c in call_contracts:
        d = call_deltas.get(c.conId)
        if d is None:
            continue
        ad = abs(d)
        if ad > cfg.max_delta:
            continue
        if c.strike < spot + cfg.short_call_min_dist:
            continue
        diff = abs(ad - cfg.target_delta)
        if diff < best_call_diff:
            best_call_diff = diff
            best_call_strike = c.strike

    if best_call_strike is None:
        log.warning("No valid short call found")
        return None

    # --- long legs ---
    long_put_strike = best_put_strike - cfg.wing_width
    long_call_strike = best_call_strike + cfg.wing_width

    # --- build & qualify ---
    legs = CondorLegs(
        expiry=expiry,
        short_put=best_put_strike,
        long_put=long_put_strike,
        short_call=best_call_strike,
        long_call=long_call_strike,
    )

    sp = Option(cfg.symbol, expiry, best_put_strike, "P", "SMART")
    lp = Option(cfg.symbol, expiry, long_put_strike, "P", "SMART")
    sc = Option(cfg.symbol, expiry, best_call_strike, "C", "SMART")
    lc = Option(cfg.symbol, expiry, long_call_strike, "C", "SMART")
    broker.ib.qualifyContracts(sp, lp, sc, lc)

    legs.short_put_con = sp
    legs.long_put_con = lp
    legs.short_call_con = sc
    legs.long_call_con = lc

    log.info("Condor: SP=%.0f LP=%.0f SC=%.0f LC=%.0f  expiry=%s",
             best_put_strike, long_put_strike,
             best_call_strike, long_call_strike, expiry)
    return legs
