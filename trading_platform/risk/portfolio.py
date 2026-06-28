"""Portfolio — pure accounting: positions, market value, P&L, equity curve.

Listens to FillEvents to update positions and cash. Listens to bar
updates via update_market() to mark-to-market.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

import pandas as pd

from ..core.events import FillEvent, OrderSide


class Portfolio:
    def __init__(self, initial_capital: float = 100_000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self._positions: dict[str, float] = defaultdict(float)
        self._market_prices: dict[str, float] = {}
        self._equity_curve: list[tuple[pd.Timestamp, float]] = []
        self.fills: list[FillEvent] = []

    def on_fill(self, fill: FillEvent) -> None:
        sign = 1.0 if fill.side == OrderSide.BUY else -1.0
        self._positions[fill.symbol] += sign * fill.quantity
        self.cash -= sign * fill.quantity * fill.fill_price
        self.cash -= fill.commission
        self.fills.append(fill)

    def update_market(self, symbol: str, price: float, ts: pd.Timestamp) -> None:
        self._market_prices[symbol] = price
        # Record equity snapshot once per timestamp (caller responsibility avoids
        # excessive snapshots, but we deduplicate anyway).
        if self._equity_curve and self._equity_curve[-1][0] == ts:
            self._equity_curve[-1] = (ts, self.equity())
        else:
            self._equity_curve.append((ts, self.equity()))

    def positions(self) -> dict[str, float]:
        return {k: v for k, v in self._positions.items() if v != 0}

    def equity(self) -> float:
        mv = sum(qty * self._market_prices.get(sym, 0.0)
                 for sym, qty in self._positions.items())
        return self.cash + mv

    def equity_curve(self) -> pd.DataFrame:
        if not self._equity_curve:
            return pd.DataFrame(columns=["timestamp", "equity"]).set_index("timestamp")
        df = pd.DataFrame(self._equity_curve, columns=["timestamp", "equity"])
        return df.drop_duplicates("timestamp", keep="last").set_index("timestamp")

    def daily_returns(self) -> pd.Series:
        eq = self.equity_curve()["equity"]
        return eq.resample("1D").last().pct_change().dropna()
