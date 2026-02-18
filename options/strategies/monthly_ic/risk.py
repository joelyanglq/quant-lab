"""Risk monitoring — delta red line & proximity red line."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List

from ib_insync import Option

from options.core.broker import Broker, CondorLegs
from .config import MonthlyICConfig

log = logging.getLogger(__name__)


@dataclass
class RiskAlert:
    rule: str          # "delta" | "proximity_put" | "proximity_call"
    detail: str
    severity: str = "RED"


def check_risk(broker: Broker, legs: CondorLegs,
               cfg: MonthlyICConfig) -> List[RiskAlert]:
    """Check risk rules against current market state."""
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
