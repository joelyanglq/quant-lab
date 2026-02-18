"""Risk monitoring + P&L exit for 0DTE Iron Condor."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional

from ib_insync import Option

from options.core.broker import Broker, CondorLegs
from .config import ZeroDTEConfig

log = logging.getLogger(__name__)


@dataclass
class RiskAlert:
    rule: str
    detail: str
    severity: str = "RED"


def check_risk(broker: Broker, legs: CondorLegs,
               cfg: ZeroDTEConfig) -> List[RiskAlert]:
    """Check proximity and delta red lines (tighter for 0DTE)."""
    alerts: List[RiskAlert] = []
    spot = broker.get_underlying_price()
    if spot is None or math.isnan(spot):
        alerts.append(RiskAlert("data", "Cannot get underlying price"))
        return alerts

    # --- proximity red line ---
    dist_put = spot - legs.short_put
    dist_call = legs.short_call - spot

    if dist_put < cfg.proximity_red_line:
        msg = (f"Spot {spot:.0f} within {dist_put:.0f} pts "
               f"of short put {legs.short_put:.0f}")
        log.warning("PROXIMITY ALERT (put): %s", msg)
        alerts.append(RiskAlert("proximity_put", msg))

    if dist_call < cfg.proximity_red_line:
        msg = (f"Spot {spot:.0f} within {dist_call:.0f} pts "
               f"of short call {legs.short_call:.0f}")
        log.warning("PROXIMITY ALERT (call): %s", msg)
        alerts.append(RiskAlert("proximity_call", msg))

    # --- delta red line ---
    contracts = [
        Option(cfg.symbol, legs.expiry, legs.short_put, "P", "SMART"),
        Option(cfg.symbol, legs.expiry, legs.short_call, "C", "SMART"),
    ]
    deltas = broker.get_option_delta(contracts)

    net_delta = 0.0
    for c in contracts:
        d = deltas.get(c.conId, 0.0)
        net_delta += -d

    abs_delta = abs(net_delta)
    log.info("Net delta = %.2f  (red line = %.0f)", abs_delta,
             cfg.delta_red_line)

    if abs_delta > cfg.delta_red_line:
        msg = f"Net |delta| = {abs_delta:.1f} > {cfg.delta_red_line:.0f}"
        log.warning("DELTA ALERT: %s", msg)
        alerts.append(RiskAlert("delta", msg))

    return alerts


def check_pnl_exit(broker: Broker, legs: CondorLegs,
                   entry_credit: float,
                   cfg: ZeroDTEConfig) -> Optional[str]:
    """Check profit target and stop loss.

    Returns exit reason string ("profit_target" or "stop_loss") or None.
    """
    debit = broker.get_combo_mid_price(legs)
    if debit is None:
        log.warning("Cannot get combo mid price for P&L check")
        return None

    # profit target: cost to close <= credit * (1 - target_pct)
    # e.g. credit=1.50, target=50% → close when debit <= 0.75
    profit_threshold = entry_credit * (1 - cfg.profit_target_pct)
    if debit <= profit_threshold:
        log.info("PROFIT TARGET  debit=%.2f <= %.2f  (credit=%.2f, target=%.0f%%)",
                 debit, profit_threshold, entry_credit,
                 cfg.profit_target_pct * 100)
        return "profit_target"

    # stop loss: cost to close >= credit * stop_loss_mult
    loss_threshold = entry_credit * cfg.stop_loss_mult
    if debit >= loss_threshold:
        log.warning("STOP LOSS  debit=%.2f >= %.2f  (credit=%.2f, mult=%.1fx)",
                    debit, loss_threshold, entry_credit, cfg.stop_loss_mult)
        return "stop_loss"

    log.info("P&L check  debit=%.2f  profit_at=%.2f  stop_at=%.2f",
             debit, profit_threshold, loss_threshold)
    return None
