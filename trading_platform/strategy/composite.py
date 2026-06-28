"""CompositeStrategy — orchestrator.

Owns a list of Alphas, a Combiner, a RiskSizer. On each `on_bar`, when
the bar matches the strategy's trigger frequency, it:
    1. Calls each alpha's forecast() for symbols in its universe.
    2. Combines via the combiner.
    3. Sizes via the sizer using current prices, EWMA vol, capital.
    4. Diffs target positions vs current; emits OrderEvents for the diff.

Engine.run() drives the whole thing; this class is what the user
typically constructs.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..core.events import Bar, Frequency, OrderEvent, OrderSide, OrderType
from .alpha import Alpha
from .combiner import Combiner, WeightedCombiner
from .sizer import RiskSizer


class CompositeStrategy:
    """Wires alphas → combiner → sizer → orders.

    Args:
        alphas: list of Alpha instances; their trigger_freq drives evaluation.
        combiner: forecast combiner (defaults to equal-weight WeightedCombiner).
        sizer: RiskSizer instance.
        ctx: DataContext for as_of queries.
        execution: ExecutionHandler to receive orders.
        portfolio: Portfolio to read current positions and equity.
        initial_capital: initial capital in USD.
        trigger_freq: top-level cadence; defaults to EOD.
        strategy_id: identifier propagated into orders.
    """

    def __init__(
        self,
        alphas: Sequence[Alpha],
        combiner: Optional[Combiner],
        sizer: RiskSizer,
        ctx,
        execution,
        portfolio,
        initial_capital: float = 100_000.0,
        trigger_freq: Frequency = Frequency.EOD,
        strategy_id: str = "composite",
    ):
        self.alphas = list(alphas)
        self.combiner = combiner or WeightedCombiner()
        self.sizer = sizer
        self.ctx = ctx
        self.execution = execution
        self.portfolio = portfolio
        self.capital = initial_capital
        self.trigger_freq = trigger_freq
        self.strategy_id = strategy_id
        self._latest_prices: dict[str, float] = {}
        self._last_rebalance_dt: Optional[pd.Timestamp] = None

    def on_bar(self, bar: Bar) -> None:
        # Update price cache.
        self._latest_prices[bar.symbol] = bar.close
        # Avoid repeating rebalance multiple times for same dt across symbols.
        if self._last_rebalance_dt == bar.timestamp:
            return
        if bar.frequency != self.trigger_freq:
            return
        self._rebalance(bar.timestamp)
        self._last_rebalance_dt = bar.timestamp

    def _rebalance(self, dt: pd.Timestamp) -> None:
        # 1. Collect per-alpha forecasts.
        forecasts_list: list[dict[str, float]] = []
        for alpha in self.alphas:
            try:
                f = alpha.forecast(dt, self.ctx)
            except Exception:
                f = {}
            forecasts_list.append(f or {})

        # 2. Combine.
        combined = self.combiner.combine(forecasts_list,
                                         strategy_ids=[a.strategy_id for a in self.alphas])
        if not combined:
            return

        # 3. Get prices and EWMA vol from ctx.
        symbols = [s for s, v in combined.items() if v is not None and not np.isnan(v)]
        if not symbols:
            return
        try:
            ewma = self.ctx.as_of(dt, "ewma_vol", symbols=symbols)
            ewma_dict = ewma.to_dict() if hasattr(ewma, "to_dict") else dict(ewma)
        except Exception:
            ewma_dict = {s: 0.20 for s in symbols}  # fallback

        # 4. Equity / capital.
        equity = self.portfolio.equity() if hasattr(self.portfolio, "equity") else self.capital
        self.sizer.update_equity(equity)

        # 5. Size.
        target_positions = self.sizer.size(
            forecasts=combined,
            prices={s: self._latest_prices.get(s, np.nan) for s in symbols},
            ewma_vol=ewma_dict,
            capital=equity,
            equity=equity,
        )

        # 6. Diff vs current and emit orders.
        current = self.portfolio.positions() if hasattr(self.portfolio, "positions") else {}
        for sym in set(list(target_positions.keys()) + list(current.keys())):
            target = target_positions.get(sym, 0.0)
            cur = current.get(sym, 0.0)
            delta = target - cur
            # Round to nearest share.
            delta = float(np.round(delta))
            if abs(delta) < 1.0:
                continue
            order = OrderEvent(
                timestamp=dt,
                symbol=sym,
                side=OrderSide.BUY if delta > 0 else OrderSide.SELL,
                quantity=abs(delta),
                order_type=OrderType.MKT,
                strategy_id=self.strategy_id,
            )
            self.execution.submit_order(order)
