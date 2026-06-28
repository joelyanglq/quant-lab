"""RiskSizer — Carver Ch9-10 risk budget.

position[i] = (forecast[i] / TARGET_ABS_FORECAST)
            * (target_vol_per_instrument / sigma[i])
            * (capital * instrument_weight[i] / price[i])
            * half_kelly_factor
            * drawdown_scaler

Plus a max-leverage clamp at the portfolio level.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import pandas as pd

from .alpha import TARGET_ABS_FORECAST


class RiskSizer:
    def __init__(
        self,
        target_vol: float = 0.15,
        max_leverage: float = 1.0,
        kelly_floor: float = 0.3,
        kelly_ceil: float = 1.0,
        kelly_default: float = 0.5,
        dd_thresholds: tuple[float, float, float] = (0.05, 0.10, 0.20),
        diversification_multiplier: float = 1.0,
        vol_floor: float = 0.05,
    ):
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.kelly_floor = kelly_floor
        self.kelly_ceil = kelly_ceil
        self.kelly_default = kelly_default
        self.dd_thresholds = dd_thresholds
        self.diversification_multiplier = diversification_multiplier
        self.vol_floor = vol_floor
        self._returns_history: deque = deque(maxlen=252)
        self._equity_peak: float = 0.0

    # ── Kelly factor ─────────────────────────────────────────────────
    def update_returns(self, daily_return: float) -> None:
        if not np.isnan(daily_return):
            self._returns_history.append(daily_return)

    def half_kelly_factor(self) -> float:
        if len(self._returns_history) < 60:
            return self.kelly_default
        arr = np.array(self._returns_history)
        mu = arr.mean() * 252
        sigma = arr.std(ddof=1) * np.sqrt(252)
        if sigma <= 0:
            return self.kelly_default
        sr = mu / sigma
        kelly = 0.5 * sr / self.target_vol
        return float(np.clip(kelly, self.kelly_floor, self.kelly_ceil))

    # ── Drawdown scaler ──────────────────────────────────────────────
    def update_equity(self, equity: float) -> None:
        if equity > self._equity_peak:
            self._equity_peak = equity

    def drawdown_scaler(self, equity: float) -> float:
        if self._equity_peak <= 0:
            return 1.0
        dd = 1.0 - equity / self._equity_peak
        t1, t2, t3 = self.dd_thresholds
        if dd < t1:
            return 1.0
        if dd < t2:
            # Linear from 1.0 to 0.5 between t1 and t2.
            return 1.0 - 0.5 * (dd - t1) / (t2 - t1)
        if dd < t3:
            # Linear from 0.5 to 0 between t2 and t3.
            return 0.5 * (1.0 - (dd - t2) / (t3 - t2))
        return 0.0

    # ── Per-instrument vol target ────────────────────────────────────
    def per_instrument_vol_target(self, n_instruments: int) -> float:
        if n_instruments <= 0:
            return 0.0
        return self.target_vol * self.diversification_multiplier / n_instruments

    # ── Main sizing ──────────────────────────────────────────────────
    def size(
        self,
        forecasts: dict[str, float],
        prices: dict[str, float],
        ewma_vol: dict[str, float],
        instrument_weights: Optional[dict[str, float]] = None,
        capital: float = 100_000.0,
        equity: Optional[float] = None,
    ) -> dict[str, float]:
        """Return target positions in shares (signed)."""
        active = {s: f for s, f in forecasts.items()
                  if f is not None and not np.isnan(f)}
        if not active:
            return {}

        n = len(active)
        per_inst_target_vol = self.per_instrument_vol_target(n)
        kelly = self.half_kelly_factor()
        dd_scaler = self.drawdown_scaler(equity) if equity is not None else 1.0
        weights = instrument_weights or {s: 1.0 / n for s in active}

        positions = {}
        total_notional = 0.0
        for sym, f in active.items():
            price = prices.get(sym)
            sigma = ewma_vol.get(sym)
            if price is None or sigma is None or price <= 0:
                continue
            sigma = max(sigma, self.vol_floor)
            w = weights.get(sym, 0.0)
            if w <= 0:
                continue
            qty = (
                (f / TARGET_ABS_FORECAST)
                * (per_inst_target_vol / sigma)
                * (capital * w / price)
                * kelly
                * dd_scaler
            )
            positions[sym] = qty
            total_notional += abs(qty) * price

        # Max leverage clamp.
        max_notional = self.max_leverage * capital
        if total_notional > max_notional and total_notional > 0:
            scale = max_notional / total_notional
            positions = {s: q * scale for s, q in positions.items()}

        return positions
