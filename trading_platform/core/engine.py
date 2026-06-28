"""Multi-frequency event-driven engine.

Merges multiple bar streams (EOD + 1min + 5min ...) via a min-heap and
dispatches each bar to subscribers. Strategies declare `trigger_freq` and
the engine only invokes their `on_bar` for matching frequencies.

Boundary callbacks (`on_market_close`, `on_week_end`, `on_month_end`) fire
after the last bar of the corresponding period for any subscribed frequency.
"""
from __future__ import annotations

import heapq
import logging
from typing import Iterable, Optional, Protocol

import pandas as pd

from .clock import BacktestClock
from .events import Bar, Frequency

log = logging.getLogger(__name__)


class BarSource(Protocol):
    """Anything that yields Bar in chronological order."""
    def __iter__(self): ...


class _StrategyHook(Protocol):
    trigger_freq: Frequency
    def on_bar(self, bar: Bar) -> None: ...


class Engine:
    """Multi-frequency event loop.

    Args:
        feeds: list of bar sources (each iterable of Bars at one frequency).
        strategies: list of strategy objects with `on_bar` and `trigger_freq`.
        execution_handler: ExecutionHandler instance (drives fills).
        portfolio: Portfolio instance (records positions / equity curve).
        clock: BacktestClock; backtest engine sets it before each callback.
    """

    def __init__(
        self,
        feeds: Iterable[BarSource],
        strategies: Iterable,
        execution_handler,
        portfolio,
        clock: Optional[BacktestClock] = None,
    ):
        self.feeds = list(feeds)
        self.strategies = list(strategies)
        self.execution = execution_handler
        self.portfolio = portfolio
        self.clock = clock or BacktestClock()
        self._last_dt_per_freq: dict[Frequency, pd.Timestamp] = {}

    # ──────────────────────────────────────────────────────────────────
    def _merged_bars(self):
        """Heap-merge bars from all feeds in chronological order."""
        iters = [iter(f) for f in self.feeds]
        heap = []
        for i, it in enumerate(iters):
            try:
                bar = next(it)
                heapq.heappush(heap, (bar.timestamp, i, bar))
            except StopIteration:
                pass
        while heap:
            ts, i, bar = heapq.heappop(heap)
            yield bar
            try:
                nxt = next(iters[i])
                heapq.heappush(heap, (nxt.timestamp, i, nxt))
            except StopIteration:
                pass

    # ──────────────────────────────────────────────────────────────────
    def run(self) -> None:
        prev_day: pd.Timestamp | None = None
        prev_week: pd.Timestamp | None = None
        prev_month: pd.Timestamp | None = None

        for strat in self.strategies:
            if hasattr(strat, "on_start"):
                strat.on_start()

        for bar in self._merged_bars():
            self.clock.set(bar.timestamp)

            # Update mark-to-market with this bar.
            if hasattr(self.portfolio, "update_market"):
                self.portfolio.update_market(bar.symbol, bar.close, bar.timestamp)

            # Drive any pending orders through execution handler.
            if hasattr(self.execution, "process_bar"):
                self.execution.process_bar(bar)

            # Dispatch to strategies whose trigger_freq matches this bar.
            for strat in self.strategies:
                tf = getattr(strat, "trigger_freq", None)
                if tf is None or tf == bar.frequency:
                    try:
                        strat.on_bar(bar)
                    except Exception as e:
                        log.exception("Strategy %s.on_bar failed: %s", type(strat).__name__, e)

            # Boundary callbacks at the granularity of the most coarse subscribed feed.
            day = bar.timestamp.normalize()
            if prev_day is not None and day > prev_day:
                self._fire("on_market_close", prev_day)
                if prev_week is None or self._is_new_week(prev_day, day):
                    self._fire("on_week_end", prev_day)
                    prev_week = day
                if prev_month is None or self._is_new_month(prev_day, day):
                    self._fire("on_month_end", prev_day)
                    prev_month = day
            prev_day = day

        # Final flush.
        if prev_day is not None:
            self._fire("on_market_close", prev_day)

        for strat in self.strategies:
            if hasattr(strat, "on_finish"):
                strat.on_finish()

    # ──────────────────────────────────────────────────────────────────
    def _fire(self, name: str, dt: pd.Timestamp) -> None:
        for strat in self.strategies:
            cb = getattr(strat, name, None)
            if callable(cb):
                try:
                    cb(dt)
                except Exception:
                    log.exception("%s callback failed on %s", name, type(strat).__name__)

    @staticmethod
    def _is_new_week(prev: pd.Timestamp, cur: pd.Timestamp) -> bool:
        return cur.isocalendar().week != prev.isocalendar().week or cur.year != prev.year

    @staticmethod
    def _is_new_month(prev: pd.Timestamp, cur: pd.Timestamp) -> bool:
        return cur.month != prev.month or cur.year != prev.year
