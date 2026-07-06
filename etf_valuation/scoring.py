"""
Scoring and signal classification for ETF valuation.

Converts percentile values into human-readable signals
(STRONG_BUY to STRONG_SELL) and computes composite scores.
"""
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class Signal:
    label: str
    emoji: str
    action: str


SIGNALS = {
    "STRONG_BUY":  Signal("极度低估", "🟢", "STRONG_BUY"),
    "BUY":         Signal("低估",     "🟢", "BUY"),
    "LEAN_BUY":    Signal("偏低",     "🟡", "LEAN_BUY"),
    "HOLD":        Signal("合理",     "⚪", "HOLD"),
    "LEAN_SELL":   Signal("偏高",     "🟡", "LEAN_SELL"),
    "SELL":        Signal("高估",     "🔴", "SELL"),
    "STRONG_SELL": Signal("极度高估", "🔴", "STRONG_SELL"),
}

# Thresholds for percentile → signal mapping
# direction: "higher_expensive" means higher percentile = more expensive
THRESHOLDS_EXPENSIVE = [
    (0.00, 0.10, "STRONG_BUY"),
    (0.10, 0.25, "BUY"),
    (0.25, 0.40, "LEAN_BUY"),
    (0.40, 0.60, "HOLD"),
    (0.60, 0.75, "LEAN_SELL"),
    (0.75, 0.90, "SELL"),
    (0.90, 1.01, "STRONG_SELL"),
]

# direction: "higher_cheap" means higher percentile = cheaper (inverted)
THRESHOLDS_CHEAP = [
    (0.00, 0.10, "STRONG_SELL"),
    (0.10, 0.25, "SELL"),
    (0.25, 0.40, "LEAN_SELL"),
    (0.40, 0.60, "HOLD"),
    (0.60, 0.75, "LEAN_BUY"),
    (0.75, 0.90, "BUY"),
    (0.90, 1.01, "STRONG_BUY"),
]


def percentile_to_signal(
    percentile: float,
    direction: str = "higher_expensive",
) -> str:
    """
    Convert a percentile value to a signal label.

    Args:
        percentile: value in [0, 1]
        direction: "higher_expensive" or "higher_cheap"

    Returns:
        Signal key (e.g. "STRONG_BUY")
    """
    if np.isnan(percentile):
        return "N/A"

    thresholds = THRESHOLDS_EXPENSIVE if direction == "higher_expensive" else THRESHOLDS_CHEAP

    for lo, hi, signal in thresholds:
        if lo <= percentile < hi:
            return signal

    return "N/A"


def composite_score(
    primary_pct: float,
    secondary_pct: Optional[float],
    primary_direction: str,
    secondary_direction: Optional[str] = None,
    primary_weight: float = 0.7,
    secondary_weight: float = 0.3,
) -> float:
    """
    Compute composite percentile score from primary and secondary metrics.

    Normalizes all percentiles to the "higher_expensive" direction before
    combining, so the composite always means: higher = more expensive.

    Returns:
        Composite percentile in [0, 1], higher = more expensive
    """
    # Normalize to "higher = expensive"
    p = primary_pct if primary_direction == "higher_expensive" else 1.0 - primary_pct

    if secondary_pct is not None and not np.isnan(secondary_pct) and secondary_direction:
        s = secondary_pct if secondary_direction == "higher_expensive" else 1.0 - secondary_pct
        return p * primary_weight + s * secondary_weight

    return p


def score_etf(
    primary_pct: float,
    secondary_pct: Optional[float],
    primary_metric: str,
    secondary_metric: Optional[str],
    primary_weight: float = 0.7,
    secondary_weight: float = 0.3,
) -> Dict:
    """
    Full scoring for an ETF: percentiles → signals → composite.

    Args:
        primary_pct: primary metric percentile [0, 1]
        secondary_pct: secondary metric percentile [0, 1]
        primary_metric: metric key (e.g. "pe_ttm")
        secondary_metric: metric key or None

    Returns:
        dict with signal info
    """
    from etf_valuation.config import METRICS

    primary_dir = METRICS.get(primary_metric, {}).get("direction", "higher_expensive")
    primary_signal = percentile_to_signal(primary_pct, primary_dir)

    result = {
        "primary_metric": primary_metric,
        "primary_pct": primary_pct,
        "primary_signal": primary_signal,
    }

    if secondary_metric and secondary_pct is not None:
        secondary_dir = METRICS.get(secondary_metric, {}).get("direction", "higher_expensive")
        secondary_signal = percentile_to_signal(secondary_pct, secondary_dir)
        comp = composite_score(
            primary_pct, secondary_pct, primary_dir, secondary_dir,
            primary_weight, secondary_weight,
        )
        comp_signal = percentile_to_signal(comp, "higher_expensive")

        result.update({
            "secondary_metric": secondary_metric,
            "secondary_pct": secondary_pct,
            "secondary_signal": secondary_signal,
            "composite_pct": comp,
            "composite_signal": comp_signal,
        })
    else:
        result["composite_pct"] = primary_pct if primary_dir == "higher_expensive" else 1.0 - primary_pct
        result["composite_signal"] = primary_signal

    return result
