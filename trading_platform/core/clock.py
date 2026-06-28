"""Logical clock — backtest advances per bar, live returns wall time.

Strategy code reads `clock.now()` for time-dependent logic and works
unchanged in both modes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Clock(ABC):
    @abstractmethod
    def now(self) -> pd.Timestamp:
        """Return current logical timestamp (UTC)."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(now={self.now()})"


class BacktestClock(Clock):
    """Advances per bar; engine sets `_now` before each strategy callback."""

    def __init__(self, start: pd.Timestamp | None = None):
        self._now = pd.Timestamp(start) if start is not None else pd.Timestamp("1970-01-01", tz="UTC")

    def now(self) -> pd.Timestamp:
        return self._now

    def set(self, ts: pd.Timestamp) -> None:
        self._now = ts


class LiveClock(Clock):
    """Wall-clock UTC time."""

    def now(self) -> pd.Timestamp:
        return pd.Timestamp.utcnow().tz_localize("UTC") if pd.Timestamp.utcnow().tz is None else pd.Timestamp.utcnow()
