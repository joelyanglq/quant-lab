"""Cross-section multi-factor archetype.

Computes a panel of factors, ranks within sector groups (default), and
produces forecast = sector-neutral z-score of composite factor.

Factor screening uses expanding-window IC (computed by a separate
`FactorScreener`); selected factors are weighted by handcrafted weights
derived from factor correlation. No in-sample top-N selection is allowed.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from ...core.events import Frequency
from ..alpha import Alpha, ScalingMixin
from ..combiner import handcraft_weights


FactorFn = Callable[[pd.DataFrame], pd.DataFrame]
"""A factor function takes a price panel (date × symbol) and returns
a panel of the same shape with the factor value per (date, symbol)."""


def factor_momentum_12_1(close: pd.DataFrame) -> pd.DataFrame:
    """12-1 month momentum (skip last 21 days)."""
    p21 = close.shift(21)
    p273 = close.shift(252 + 21)
    return (p21 / p273 - 1.0)


def factor_short_reversal(close: pd.DataFrame) -> pd.DataFrame:
    """Negative of 1-month return (short-term mean reversion)."""
    return -(close / close.shift(21) - 1.0)


def factor_low_vol(close: pd.DataFrame) -> pd.DataFrame:
    """-1 × 60-day rolling vol (lower vol → higher score)."""
    return -close.pct_change().rolling(60).std()


DEFAULT_FACTORS: dict[str, FactorFn] = {
    "momentum_12_1": factor_momentum_12_1,
    "short_reversal": factor_short_reversal,
    "low_vol": factor_low_vol,
}


def _cs_zscore(panel: pd.DataFrame, sector: Optional[pd.Series] = None) -> pd.DataFrame:
    """Cross-sectional z-score, optionally within sector groups."""
    if sector is None:
        return panel.sub(panel.mean(axis=1), axis=0).div(panel.std(axis=1), axis=0)
    # Per-sector z-score per row.
    out = panel.copy().astype(float) * np.nan
    syms = panel.columns
    sec_groups: dict[str, list[str]] = {}
    for s in syms:
        if s not in sector.index:
            continue
        sec_groups.setdefault(sector[s], []).append(s)
    for sec, group in sec_groups.items():
        sub = panel[group]
        z = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1), axis=0)
        out[group] = z
    return out


class CrossSectionAlpha(Alpha, ScalingMixin):
    """Multi-factor cross-section alpha.

    Args:
        factors: dict of factor_name -> callable. Defaults to 3 demo factors.
        rebalance_freq: 'W-FRI' (weekly Fri close) or 'M' (month-end).
        sector_neutral: if True, rank within GICS sectors.
        min_history_factor: bars of history needed before producing forecast.
        ic_lookback: window in days used to compute factor IC for handcrafted weighting.
        ic_min_train: minimum days of factor history before IC can drive weights;
                      below threshold, equal weights are used.
    """

    min_history = 252

    def __init__(
        self,
        factors: Optional[dict[str, FactorFn]] = None,
        rebalance_freq: str = "W-FRI",
        sector_neutral: bool = True,
        ic_lookback: int = 504,
        ic_min_train: int = 504,
        strategy_id: str = "cross_section",
    ):
        super().__init__()
        self._init_scaling(window=252)
        self.factors = factors or dict(DEFAULT_FACTORS)
        self.rebalance_freq = rebalance_freq
        self.sector_neutral = sector_neutral
        self.ic_lookback = ic_lookback
        self.ic_min_train = ic_min_train
        self.trigger_freq = Frequency.EOD
        self.strategy_id = strategy_id

    def _is_rebalance_day(self, dt: pd.Timestamp) -> bool:
        if self.rebalance_freq == "W-FRI":
            return dt.weekday() == 4
        if self.rebalance_freq == "M":
            # Last business day of month: simplified to dt + 5 days same month.
            return (dt + pd.Timedelta(days=7)).month != dt.month
        if self.rebalance_freq == "Q":
            return dt.month in (3, 6, 9, 12) and (dt + pd.Timedelta(days=7)).month != dt.month
        return False

    def forecast(self, dt: pd.Timestamp, ctx) -> dict[str, float]:
        if not self._is_rebalance_day(dt):
            return {}

        symbols = ctx.universe(dt)
        prices = ctx.as_of(dt, "price_1d", lookback=self.ic_lookback, symbols=symbols)
        if prices is None or prices.empty:
            return {}

        # Compute each factor; final value at dt only.
        factor_panels = {}
        for name, fn in self.factors.items():
            try:
                panel = fn(prices)
                factor_panels[name] = panel
            except Exception:
                continue
        if not factor_panels:
            return {}

        # Sector mapping for sector-neutral z-score.
        sector = None
        if self.sector_neutral:
            try:
                sector = ctx.as_of(dt, "gics_sector")
            except Exception:
                sector = None

        # Cross-sectional z-score at the latest date.
        latest_z = {}
        for name, panel in factor_panels.items():
            z = _cs_zscore(panel, sector=sector if (sector is not None and not sector.empty) else None)
            latest_z[name] = z.iloc[-1] if not z.empty else pd.Series(dtype=float)

        # Determine factor weights.
        factor_names = list(latest_z.keys())
        weights = self._handcraft_factor_weights(prices, factor_panels, factor_names)

        # Combine factors with weights -> composite z.
        combined = pd.Series(0.0, index=symbols)
        denom = pd.Series(0.0, index=symbols)
        for name, w in zip(factor_names, weights):
            v = latest_z[name].reindex(symbols)
            mask = v.notna()
            combined[mask] += w * v[mask]
            denom[mask] += w
        combined = (combined / denom.replace(0, np.nan))

        raw = {s: (float(combined[s]) if s in combined.index and not np.isnan(combined[s]) else float("nan"))
               for s in symbols}
        return self._scale_and_cap(raw)

    def _handcraft_factor_weights(
        self,
        prices: pd.DataFrame,
        factor_panels: dict[str, pd.DataFrame],
        factor_names: list[str],
    ) -> list[float]:
        """Carver-style weights from factor IC correlation. Equal weights if not enough history."""
        if len(prices) < self.ic_min_train:
            return [1.0 / len(factor_names)] * len(factor_names)

        # Compute IC time series (rank correlation of factor with next-period return).
        fwd_ret = prices.pct_change(21).shift(-21)
        ics = {}
        for name in factor_names:
            panel = factor_panels[name]
            joined = panel.iloc[-self.ic_lookback:]
            ic_series = []
            for ts in joined.index:
                if ts not in fwd_ret.index:
                    continue
                fac_row = joined.loc[ts]
                ret_row = fwd_ret.loc[ts]
                pair = pd.concat([fac_row, ret_row], axis=1, keys=["f", "r"]).dropna()
                if len(pair) < 20:
                    continue
                ic_series.append(pair["f"].rank().corr(pair["r"].rank()))
            if ic_series:
                ics[name] = pd.Series(ic_series)

        if len(ics) < 2:
            return [1.0 / len(factor_names)] * len(factor_names)

        ic_df = pd.DataFrame({k: v.reset_index(drop=True) for k, v in ics.items()})
        ic_df = ic_df.reindex(columns=factor_names)
        corr = ic_df.corr().fillna(0.0).values
        weights = handcraft_weights(corr)
        return list(weights)
