"""0DTE entry trigger and condor leg selection."""

from __future__ import annotations

import logging
import math
from datetime import datetime, time, timedelta
from typing import Optional

from ib_insync import Option

from options.core.broker import Broker, CondorLegs
from .config import ZeroDTEConfig

log = logging.getLogger(__name__)


def check_vix_condition(broker: Broker, cfg: ZeroDTEConfig) -> bool:
    """Return True if VIX is within the acceptable band."""
    current, _ = broker.get_vix()
    if current is None or math.isnan(current):
        log.warning("VIX price unavailable")
        return False

    log.info("VIX = %.2f  band=[%.0f, %.0f]",
             current, cfg.vix_min, cfg.vix_max)
    return cfg.vix_min <= current <= cfg.vix_max


def is_entry_window(cfg: ZeroDTEConfig, now_et: datetime) -> bool:
    """Return True if within the entry time window."""
    t = now_et.time()
    end_min = cfg.entry_time.minute + cfg.entry_window_min
    end_hour = cfg.entry_time.hour + end_min // 60
    end_min = end_min % 60
    entry_end = time(end_hour, end_min)
    return cfg.entry_time <= t <= entry_end


def _pick_expiry_today(broker: Broker, cfg: ZeroDTEConfig) -> Optional[str]:
    """Pick today's expiry. Try SPXW (weekly/daily), fallback to SPX chain."""
    today_str = datetime.now().strftime("%Y%m%d")

    # try SPXW first (0DTE daily expirations)
    for tc in ["SPXW", cfg.symbol]:
        expirations = broker.get_chain_expirations(trading_class=tc)
        if today_str in expirations:
            log.info("Selected 0DTE expiry %s  tradingClass=%s",
                     today_str, tc)
            return today_str

    log.warning("No 0DTE expiry found for today (%s)", today_str)
    return None


def _candidate_strikes(spot: float, cfg: ZeroDTEConfig,
                       side: str) -> list[float]:
    """Generate candidate strikes for 0DTE short leg scanning."""
    if side == "put":
        center = spot - cfg.short_put_min_dist
        low = center - 40
        high = center + 10
    else:
        center = spot + cfg.short_call_min_dist
        low = center - 10
        high = center + 40

    low = int(low // 5) * 5
    high = int(math.ceil(high / 5)) * 5

    strikes = list(range(low, high + 1, 5))
    if cfg.prefer_quarter_strikes:
        strikes = [s for s in strikes if s % 25 == 0]
    return [float(s) for s in strikes]


def check_risk_reward(credit: float, cfg: ZeroDTEConfig) -> bool:
    """Pre-entry R:R gate. Returns True if acceptable."""
    max_loss = (cfg.wing_width - credit) * 100  # per contract in $
    if max_loss <= 0:
        return False
    rr = (credit * 100) / max_loss
    log.info("R:R check  credit=%.2f  max_loss=$%.0f  ratio=%.2f  min=%.2f",
             credit, max_loss, rr, cfg.min_risk_reward)
    return rr >= cfg.min_risk_reward


def find_condor_legs(broker: Broker, cfg: ZeroDTEConfig) -> Optional[CondorLegs]:
    """Scan option chain for 0DTE condor legs."""
    spot = broker.get_underlying_price()
    if spot is None or math.isnan(spot):
        log.error("Cannot get underlying price")
        return None
    log.info("%s spot = %.2f", cfg.symbol, spot)

    expiry = _pick_expiry_today(broker, cfg)
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

    log.info("0DTE Condor: SP=%.0f LP=%.0f SC=%.0f LC=%.0f  expiry=%s",
             best_put_strike, long_put_strike,
             best_call_strike, long_call_strike, expiry)
    return legs
