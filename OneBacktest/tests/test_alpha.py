"""Tests for Strategy signal mode (on_market_close returns forecasts)."""
import pandas as pd
import pytest
from strategy.base import Strategy
from data.types import Bar


class ConstantSignal(Strategy):
    """Always returns +0.5 for given symbols."""
    def __init__(self, symbols):
        self.symbols = symbols

    def on_market_close(self, dt):
        return {s: 0.5 for s in self.symbols}


class BarCountingSignal(Strategy):
    """Counts bars, returns count/100 capped at 1.0."""
    def __init__(self, symbols):
        self.symbols = symbols
        self.bar_count = 0

    def on_bar(self, bar):
        self.bar_count += 1

    def on_market_close(self, dt):
        f = min(self.bar_count / 100.0, 1.0)
        return {s: f for s in self.symbols}


def test_constant_signal():
    s = ConstantSignal(['AAPL', 'MSFT'])
    result = s.on_market_close(pd.Timestamp('2025-06-01').date())
    assert result == {'AAPL': 0.5, 'MSFT': 0.5}


def test_bar_counting():
    s = BarCountingSignal(['AAPL'])
    bar = Bar(timestamp=pd.Timestamp('2025-06-01'), symbol='AAPL',
              open=100, high=105, low=95, close=102, volume=1000)
    for _ in range(10):
        s.on_bar(bar)
    result = s.on_market_close(pd.Timestamp('2025-06-01').date())
    assert result['AAPL'] == pytest.approx(0.1)


def test_week_end_default_none():
    assert ConstantSignal(['A']).on_week_end(None) is None


def test_month_end_default_none():
    assert ConstantSignal(['A']).on_month_end(None) is None


def test_on_init_default_noop():
    ConstantSignal(['A']).on_init()  # should not raise


def test_on_bar_default_noop():
    ConstantSignal(['A']).on_bar(None)  # should not raise
