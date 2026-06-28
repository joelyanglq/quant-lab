"""Alpha ABC — Carver normalized forecast contract.

All strategy archetypes implement this interface:
    forecast(dt) -> dict[symbol -> float in [-20, +20]]

The forecast is dimensionless and obeys E[|forecast|] ≈ 10. Strategies
must self-estimate their scaling factor (see ScalingMixin).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Optional

import numpy as np
import pandas as pd

from ..core.events import Frequency

FORECAST_CAP = 20.0
TARGET_ABS_FORECAST = 10.0


class Alpha(ABC):
    """Abstract Alpha — produces normalized forecasts.

    Subclasses set:
        trigger_freq: when forecast() should be invoked (e.g. Frequency.EOD)
        min_history:  minimum bars before forecast can produce a non-NaN value
        strategy_id:  human-readable identifier (used in logs / attribution)
    """

    trigger_freq: Frequency = Frequency.EOD
    min_history: int = 60
    strategy_id: str = "alpha"

    @abstractmethod
    def forecast(self, dt: pd.Timestamp, ctx) -> dict[str, float]:
        """Return {symbol: forecast in [-20,+20] or NaN}.

        The framework guarantees `ctx.as_of(dt, ...)` is PIT-safe.
        """

    def universe(self, dt: pd.Timestamp, ctx) -> list[str]:
        """Override to scope a sub-universe; default = full ctx universe."""
        return ctx.universe(dt)


class ScalingMixin:
    """Rolling self-scaling so that long-run E[|raw|] -> TARGET_ABS_FORECAST.

    Usage:
        class MyAlpha(Alpha, ScalingMixin):
            def __init__(self, ...):
                super().__init__()
                self._init_scaling(window=252)

            def forecast(self, dt, ctx):
                raw = self._compute_raw(dt, ctx)        # arbitrary scale
                scaled = self._scale_and_cap(raw)        # -> [-20, +20] or NaN
                return scaled
    """

    def _init_scaling(self, window: int = 252):
        self._scale_window = window
        self._raw_history: deque = deque(maxlen=window)

    def _scale_factor(self) -> float:
        """Estimate the divisor to map raw values to E[|f|]≈10."""
        if not self._raw_history:
            return 1.0
        arr = np.array([abs(x) for x in self._raw_history if not np.isnan(x)])
        if arr.size < 20:
            return 1.0
        avg_abs = float(arr.mean())
        if avg_abs <= 0:
            return 1.0
        return avg_abs / TARGET_ABS_FORECAST

    def _scale_and_cap(self, raw: dict[str, float]) -> dict[str, float]:
        """Apply rolling scaling then ±20 cap. Updates internal raw history."""
        # Track magnitudes from this round.
        for v in raw.values():
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                self._raw_history.append(v)
        scale = self._scale_factor()
        out = {}
        for sym, v in raw.items():
            if v is None or (isinstance(v, float) and np.isnan(v)):
                out[sym] = float("nan")
                continue
            scaled = v / scale if scale != 0 else 0.0
            scaled = max(-FORECAST_CAP, min(FORECAST_CAP, scaled))
            out[sym] = float(scaled)
        return out

    def validate_scaling(self) -> tuple[bool, float]:
        """Returns (is_valid, mean_abs_forecast). Used pre-flight before live."""
        if len(self._raw_history) < 100:
            return (False, float("nan"))
        scale = self._scale_factor()
        arr = np.array([abs(x) / scale for x in self._raw_history if not np.isnan(x)])
        mean_abs = float(arr.mean())
        return (7.0 <= mean_abs <= 13.0, mean_abs)
