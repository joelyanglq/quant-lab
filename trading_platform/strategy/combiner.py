"""Combiners — merge multiple strategies' forecasts into one.

Three implementations:
- WeightedCombiner: linear weighted sum (weights sum to 1).
- LayeredCombiner: multiplicative groups (e.g. stock_picking × regime × timing).
- HandcraftedCombiner: Carver Ch4 weights derived from strategy correlation.

All combiners apply ±2σ winsorization per strategy before combining and
handle NaN with weight renormalization.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


# ── helpers ────────────────────────────────────────────────────────────
def _winsorize(forecasts: dict[str, float], rolling_std: float, mean: float = 0.0,
               n_sigma: float = 2.0) -> dict[str, float]:
    """Clip per-strategy forecasts to mean ± n_sigma * rolling_std."""
    if rolling_std <= 0 or np.isnan(rolling_std):
        return dict(forecasts)
    lo, hi = mean - n_sigma * rolling_std, mean + n_sigma * rolling_std
    return {
        sym: (np.clip(v, lo, hi) if (v is not None and not np.isnan(v)) else float("nan"))
        for sym, v in forecasts.items()
    }


def _all_symbols(forecasts_list: Sequence[dict[str, float]]) -> list[str]:
    seen: set[str] = set()
    for d in forecasts_list:
        seen.update(d.keys())
    return sorted(seen)


# ── handcrafted weights ────────────────────────────────────────────────
def handcraft_weights(corr: np.ndarray) -> np.ndarray:
    """Carver Ch4 handcrafted weights via recursive bisection on correlation.

    Args:
        corr: NxN correlation matrix (symmetric, diag=1).

    Returns:
        N weights summing to 1. Highly-correlated subsets share weight.
    """
    n = corr.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])
    if n == 2:
        rho = float(corr[0, 1])
        # Carver Ch4 table: corr=0 -> 0.5/0.5, corr=1 -> 0.5/0.5 (perfectly redundant),
        # corr=0.9 -> 0.5/0.5 still equal split internally; but the *group* gets
        # less weight when stacked recursively. With just 2 strategies symmetry
        # gives equal weights regardless.
        return np.array([0.5, 0.5])

    # Recursive: split into two halves by similarity.
    if n == 3:
        # Carver Ch4 table 8: identify the most correlated pair, group them.
        i, j = _most_correlated_pair(corr)
        k = [x for x in range(3) if x not in (i, j)][0]
        rho_pair = corr[i, j]
        # Pair vs single; weight depends on average corr.
        # With pair grouped: pair_w = 1/2 * (1 - 0.3 * rho_pair) for split heuristic.
        # Simplified Carver: w_pair = 1 / (1 + sqrt((1+rho_pair)/2)) gives
        # smooth transition from 0.5 (rho=0) to 1/(1+1)=0.5 (rho=1).
        # Use his published 3-asset table approximation:
        #   rho_pair=0.0 -> pair 2*1/3 = 2/3, single 1/3 (equal among 3)
        #   rho_pair=0.9 -> pair total 1/3 (each 1/6), single 2/3
        # Linear interpolation:
        pair_total = (2.0 / 3.0) - (1.0 / 3.0) * max(0.0, rho_pair)
        weights = np.zeros(3)
        weights[i] = pair_total / 2
        weights[j] = pair_total / 2
        weights[k] = 1.0 - pair_total
        return weights

    # n >= 4: split into two groups by hierarchical clustering on correlation.
    groups = _bisect_by_correlation(corr)
    g1, g2 = groups
    sub_w1 = handcraft_weights(corr[np.ix_(g1, g1)])
    sub_w2 = handcraft_weights(corr[np.ix_(g2, g2)])
    # Group-level weight: cross-group correlation determines split.
    cross = corr[np.ix_(g1, g2)].mean() if g1 and g2 else 0.0
    g1_total = 0.5 * (1.0 - 0.3 * max(0.0, cross))
    g2_total = 1.0 - g1_total
    weights = np.zeros(n)
    for idx, w in zip(g1, sub_w1):
        weights[idx] = g1_total * w
    for idx, w in zip(g2, sub_w2):
        weights[idx] = g2_total * w
    return weights / weights.sum()  # numerical safety


def _most_correlated_pair(corr: np.ndarray) -> tuple[int, int]:
    n = corr.shape[0]
    best = (0, 1)
    best_v = -np.inf
    for i in range(n):
        for j in range(i + 1, n):
            if corr[i, j] > best_v:
                best_v = corr[i, j]
                best = (i, j)
    return best


def _bisect_by_correlation(corr: np.ndarray) -> tuple[list[int], list[int]]:
    """Greedy 2-cluster split: seed with most-correlated pair vs farthest, assign rest."""
    n = corr.shape[0]
    if n <= 2:
        return list(range(n)), []
    i, j = _most_correlated_pair(corr)
    # Find a seed for the other cluster: point farthest from {i, j}.
    far = max(
        (k for k in range(n) if k not in (i, j)),
        key=lambda k: -(corr[i, k] + corr[j, k]),
    )
    g1, g2 = [i, j], [far]
    for k in range(n):
        if k in (i, j, far):
            continue
        s1 = max(corr[k, idx] for idx in g1)
        s2 = max(corr[k, idx] for idx in g2)
        (g1 if s1 >= s2 else g2).append(k)
    return sorted(g1), sorted(g2)


# ── ABC ────────────────────────────────────────────────────────────────
class Combiner(ABC):
    @abstractmethod
    def combine(
        self,
        forecasts_list: Sequence[dict[str, float]],
        strategy_ids: Optional[Sequence[str]] = None,
    ) -> dict[str, float]:
        """Combine N strategies' forecasts into one combined forecast dict."""


# ── Weighted ───────────────────────────────────────────────────────────
class WeightedCombiner(Combiner):
    """Linear weighted combination with NaN-aware renormalization."""

    def __init__(
        self,
        weights: Optional[Sequence[float]] = None,
        winsorize_sigma: float = 2.0,
        rolling_std_per_strategy: Optional[Sequence[float]] = None,
    ):
        self.weights = list(weights) if weights is not None else None
        self.winsorize_sigma = winsorize_sigma
        self.rolling_std_per_strategy = rolling_std_per_strategy

    def combine(
        self,
        forecasts_list: Sequence[dict[str, float]],
        strategy_ids: Optional[Sequence[str]] = None,
    ) -> dict[str, float]:
        n = len(forecasts_list)
        if n == 0:
            return {}
        weights = self.weights if self.weights is not None else [1.0 / n] * n
        if len(weights) != n:
            raise ValueError(f"weights len {len(weights)} != strategies {n}")

        if self.rolling_std_per_strategy is not None:
            forecasts_list = [
                _winsorize(d, std, n_sigma=self.winsorize_sigma)
                for d, std in zip(forecasts_list, self.rolling_std_per_strategy)
            ]

        all_syms = _all_symbols(forecasts_list)
        out = {}
        for sym in all_syms:
            num = 0.0
            denom = 0.0
            for d, w in zip(forecasts_list, weights):
                v = d.get(sym, float("nan"))
                if v is None or np.isnan(v):
                    continue
                num += w * v
                denom += w
            out[sym] = (num / denom) if denom > 0 else float("nan")
        return out


# ── Handcrafted ────────────────────────────────────────────────────────
class HandcraftedCombiner(WeightedCombiner):
    """Weights derived from strategy correlation matrix (Carver Ch4)."""

    def __init__(self, corr_matrix: np.ndarray, **kw):
        weights = handcraft_weights(corr_matrix)
        super().__init__(weights=list(weights), **kw)


# ── Layered ────────────────────────────────────────────────────────────
class LayeredCombiner(Combiner):
    """Multiplicative composition of orthogonal signal layers.

    Each `group` is a list of strategy indices into forecasts_list whose
    forecasts are first averaged within the group; group results are then
    multiplied together. If `require_all_groups`, any NaN in any group
    yields NaN for that symbol; otherwise missing groups contribute neutral.

    Multiplicative requires forecasts in [-20,+20] interpreted as a sign-
    bearing magnitude. We multiply then rescale to maintain [-20,+20] range.
    """

    def __init__(
        self,
        groups: Sequence[Sequence[int]],
        require_all_groups: bool = True,
        winsorize_sigma: float = 2.0,
    ):
        self.groups = [list(g) for g in groups]
        self.require_all_groups = require_all_groups
        self.winsorize_sigma = winsorize_sigma

    def combine(
        self,
        forecasts_list: Sequence[dict[str, float]],
        strategy_ids: Optional[Sequence[str]] = None,
    ) -> dict[str, float]:
        all_syms = _all_symbols(forecasts_list)

        # Average within each group.
        group_forecasts: list[dict[str, float]] = []
        for g in self.groups:
            wc = WeightedCombiner()
            sub = [forecasts_list[i] for i in g]
            group_forecasts.append(wc.combine(sub))

        out = {}
        for sym in all_syms:
            vals = []
            any_nan = False
            for gf in group_forecasts:
                v = gf.get(sym, float("nan"))
                if v is None or np.isnan(v):
                    any_nan = True
                    if self.require_all_groups:
                        break
                else:
                    vals.append(v)
            if self.require_all_groups and any_nan:
                out[sym] = float("nan")
                continue
            if not vals:
                out[sym] = float("nan")
                continue
            # Multiplicative: normalize to [-1,1] before product, then rescale.
            normalized = [v / 20.0 for v in vals]
            prod = 1.0
            for v in normalized:
                prod *= v
            out[sym] = float(np.clip(prod * 20.0, -20.0, 20.0))
        return out
