"""Pairs (statistical arbitrage) archetype — cointegrated pair mean reversion.

A `PairsAlpha` instance manages a list of cointegrated pairs (loaded from
DataContext key 'cointegration_pairs'). For each pair (A, B) with hedge
ratio β:
    spread[t] = log(P_A[t]) - β * log(P_B[t])
    z[t]      = (spread[t] - rolling_mean) / rolling_std
    forecast_A = -z, forecast_B = +z * β
The signs are chosen so positive z (spread above mean) → short A, long B.

Stop-loss when |z| > stop_z (default 4); revalidation cadence handled by
the offline `pairs_scanner`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ...core.events import Frequency
from ..alpha import Alpha, ScalingMixin


class PairsAlpha(Alpha, ScalingMixin):
    """Z-score based mean-reversion on cointegrated pairs.

    Args:
        spread_window: rolling window for spread z-score (default 60 days).
        entry_z: |z| threshold to enter (default 2.0).
        exit_z: |z| threshold to exit toward 0 (default 0.5).
        stop_z: |z| threshold for hard stop (default 4.0).
        max_pairs: maximum number of active pairs (default 30).
    """

    min_history = 252

    def __init__(
        self,
        spread_window: int = 60,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 4.0,
        max_pairs: int = 30,
        strategy_id: str = "pairs",
    ):
        super().__init__()
        self._init_scaling(window=252)
        self.spread_window = spread_window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.max_pairs = max_pairs
        self.trigger_freq = Frequency.EOD
        self.strategy_id = strategy_id
        self._open_z: dict[tuple, float] = {}  # entry z-score per pair

    def forecast(self, dt: pd.Timestamp, ctx) -> dict[str, float]:
        try:
            pairs = ctx.as_of(dt, "cointegration_pairs")
        except Exception:
            return {}
        if pairs is None or pairs.empty:
            return {}
        pairs = pairs.head(self.max_pairs)

        symbols = sorted(set(pairs["symbol_a"]).union(pairs["symbol_b"]))
        prices = ctx.as_of(dt, "price_1d", lookback=self.spread_window * 4, symbols=symbols)
        if prices is None or prices.empty:
            return {}

        raw: dict[str, float] = {s: 0.0 for s in symbols}
        for _, row in pairs.iterrows():
            a, b = row["symbol_a"], row["symbol_b"]
            beta = float(row.get("hedge_ratio", 1.0))
            if a not in prices.columns or b not in prices.columns:
                continue
            pa = prices[a].dropna()
            pb = prices[b].dropna()
            common = pa.index.intersection(pb.index)
            if len(common) < self.spread_window + 5:
                continue
            log_pa = np.log(pa.loc[common])
            log_pb = np.log(pb.loc[common])
            spread = log_pa - beta * log_pb
            mean = spread.rolling(self.spread_window).mean().iloc[-1]
            std = spread.rolling(self.spread_window).std().iloc[-1]
            if std == 0 or np.isnan(std):
                continue
            z = (spread.iloc[-1] - mean) / std

            pair_key = (a, b)
            in_position = pair_key in self._open_z

            # Stop-loss: force exit (forecast = 0) when |z| beyond stop.
            if in_position and abs(z) > self.stop_z:
                self._open_z.pop(pair_key, None)
                continue
            # Exit when |z| < exit_z.
            if in_position and abs(z) < self.exit_z:
                self._open_z.pop(pair_key, None)
                continue
            # Entry when |z| > entry_z.
            if not in_position and abs(z) >= self.entry_z:
                self._open_z[pair_key] = float(z)

            # Active forecast: only emit if currently in position OR newly entered.
            if pair_key in self._open_z:
                # Sign: positive z → spread high → short A, long B.
                f_a = -float(z)
                f_b = float(z) * beta
                raw[a] = raw.get(a, 0.0) + f_a
                raw[b] = raw.get(b, 0.0) + f_b

        return self._scale_and_cap(raw)
