"""Rotation archetype — sector / style ETF rotation with cross-sectional momentum.

Default universe = 11 GICS sector ETFs. Forecast = z-score of 12-1 month
momentum across the universe. Sizer is RiskSizer with ERC instrument
weights computed from the 60-day covariance matrix.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ...core.events import Frequency
from ..alpha import Alpha, ScalingMixin


DEFAULT_SECTOR_ETFS: list[str] = [
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP",
    "XLI", "XLU", "XLB", "XLRE", "XLC",
]


def equal_risk_contribution(cov: np.ndarray, max_iter: int = 200, tol: float = 1e-8) -> np.ndarray:
    """ERC weights via fixed-point iteration on RC equality.

    Solves for w such that w_i * (Σw)_i is equal across i. Uses the
    classic Maillard et al. iterative scheme (no scipy dependency).
    """
    n = cov.shape[0]
    w = np.ones(n) / n
    for _ in range(max_iter):
        port_var = w @ cov @ w
        if port_var <= 0:
            return w
        rc = w * (cov @ w)
        target = port_var / n
        adj = (target / np.maximum(rc, 1e-12)) ** 0.5
        w_new = w * adj
        w_new = w_new / w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            return w_new
        w = w_new
    return w


class RotationAlpha(Alpha, ScalingMixin):
    """Cross-sectional sector momentum rotation.

    Args:
        symbols: rotation universe (default = 11 GICS sector ETFs).
        rebalance_freq: 'M' (monthly) or 'W-FRI'.
        momentum_lookback: months of return used for momentum (default 12 - skip 1).
        cov_window: covariance window in days (default 60).
        long_only: if True, negative forecasts are clipped to 0 in sizer (alpha keeps sign).
    """

    min_history = 252

    def __init__(
        self,
        symbols: Optional[Sequence[str]] = None,
        rebalance_freq: str = "M",
        momentum_skip: int = 21,
        momentum_lookback: int = 252,
        cov_window: int = 60,
        long_only: bool = True,
        strategy_id: str = "rotation",
    ):
        super().__init__()
        self._init_scaling(window=252)
        self.symbols = list(symbols or DEFAULT_SECTOR_ETFS)
        self.rebalance_freq = rebalance_freq
        self.momentum_skip = momentum_skip
        self.momentum_lookback = momentum_lookback
        self.cov_window = cov_window
        self.long_only = long_only
        self.trigger_freq = Frequency.EOD
        self.strategy_id = strategy_id

    def universe(self, dt, ctx):
        return list(self.symbols)

    def _is_rebalance_day(self, dt: pd.Timestamp) -> bool:
        if self.rebalance_freq == "W-FRI":
            return dt.weekday() == 4
        if self.rebalance_freq == "M":
            return (dt + pd.Timedelta(days=7)).month != dt.month
        return False

    def erc_weights(self, prices: pd.DataFrame) -> dict[str, float]:
        ret = prices[self.symbols].pct_change().tail(self.cov_window).dropna()
        if len(ret) < 20:
            return {s: 1.0 / len(self.symbols) for s in self.symbols}
        cov = ret.cov().values * 252
        w = equal_risk_contribution(cov)
        return {s: float(w[i]) for i, s in enumerate(self.symbols)}

    def forecast(self, dt: pd.Timestamp, ctx) -> dict[str, float]:
        if not self._is_rebalance_day(dt):
            return {}

        prices = ctx.as_of(dt, "price_1d",
                          lookback=self.momentum_lookback + self.momentum_skip + 30,
                          symbols=self.symbols)
        if prices is None or prices.empty:
            return {s: float("nan") for s in self.symbols}

        # 12-1 momentum.
        if len(prices) < self.momentum_lookback + self.momentum_skip:
            return {s: float("nan") for s in self.symbols}
        p_now = prices[self.symbols].shift(self.momentum_skip).iloc[-1]
        p_then = prices[self.symbols].shift(self.momentum_lookback + self.momentum_skip).iloc[-1]
        mom = (p_now / p_then - 1.0)

        # Cross-sectional z-score.
        if mom.std() == 0 or mom.std() != mom.std():
            return {s: float("nan") for s in self.symbols}
        z = (mom - mom.mean()) / mom.std()

        raw = {s: (float(z[s]) if s in z.index and not np.isnan(z[s]) else float("nan"))
               for s in self.symbols}
        if self.long_only:
            raw = {s: max(v, 0.0) if not np.isnan(v) else float("nan") for s, v in raw.items()}
        return self._scale_and_cap(raw)
