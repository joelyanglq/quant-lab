"""Smoke test: synthetic 2-symbol buy-and-hold runs end-to-end."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_platform.core.engine import Engine
from trading_platform.core.events import Bar, Frequency, OrderEvent, OrderSide, OrderType
from trading_platform.execution.simulated import SimulatedExecutionHandler
from trading_platform.risk.portfolio import Portfolio


class _InlineBarFeed:
    """Simple iterable feed built from a list of bars."""

    def __init__(self, bars):
        self.bars = bars
        self.frequency = Frequency.EOD

    def __iter__(self):
        return iter(self.bars)


class _BuyAndHold:
    trigger_freq = Frequency.EOD
    strategy_id = "buy_and_hold"

    def __init__(self, execution, portfolio, symbol, qty):
        self.execution = execution
        self.portfolio = portfolio
        self.symbol = symbol
        self.qty = qty
        self._bought = False

    def on_bar(self, bar: Bar):
        if bar.symbol != self.symbol or self._bought:
            return
        self.execution.submit_order(
            OrderEvent(
                timestamp=bar.timestamp,
                symbol=self.symbol,
                side=OrderSide.BUY,
                quantity=self.qty,
                order_type=OrderType.MKT,
                strategy_id=self.strategy_id,
            )
        )
        self._bought = True


def test_buy_and_hold_end_to_end(synthetic_bars):
    """Synthetic 500-day 2-symbol run completes and produces an equity curve."""
    feed = _InlineBarFeed(synthetic_bars)
    portfolio = Portfolio(initial_capital=100_000.0)
    execution = SimulatedExecutionHandler(slippage_model="none")
    execution.on_fill(portfolio.on_fill)
    strat = _BuyAndHold(execution, portfolio, symbol="AAPL", qty=100)

    engine = Engine(
        feeds=[feed],
        strategies=[strat],
        execution_handler=execution,
        portfolio=portfolio,
    )
    engine.run()

    eq = portfolio.equity_curve()
    assert not eq.empty
    assert len(portfolio.fills) == 1
    assert portfolio.positions()["AAPL"] == 100
