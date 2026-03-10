"""Tests for VolTargetSizer."""
import numpy as np
import pandas as pd
import pytest

from strategy.sizer import VolTargetSizer
from data.types import Bar
from data.history import HistoryManager


def _make_history(symbol, prices):
    """Create a HistoryManager pre-loaded with price data."""
    hm = HistoryManager([symbol], max_periods=504)
    for i, p in enumerate(prices):
        bar = Bar(
            timestamp=pd.Timestamp(f'2025-01-{i + 1:02d} 16:00:00'),
            symbol=symbol,
            open=p, high=p, low=p, close=p, volume=1000,
        )
        hm._on_bar(bar)
    return hm


def _random_walk(n=30, start=100.0, daily_vol=0.01, seed=42):
    np.random.seed(seed)
    prices = [start]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + np.random.normal(0, daily_vol)))
    return prices


def test_basic_sizing():
    prices = _random_walk(30)
    hm = _make_history('AAPL', prices)
    latest = {'AAPL': Bar(
        timestamp=pd.Timestamp('2025-01-30 16:00:00'),
        symbol='AAPL', open=prices[-1], high=prices[-1],
        low=prices[-1], close=prices[-1], volume=1000,
    )}
    sizer = VolTargetSizer(target_vol=0.15, vol_lookback=20)
    targets = sizer.size({'AAPL': 0.5}, hm, 100_000.0, latest)
    assert 'AAPL' in targets
    assert targets['AAPL'] > 0


def test_below_min_forecast_yields_empty():
    prices = _random_walk(30)
    hm = _make_history('AAPL', prices)
    latest = {'AAPL': Bar(
        timestamp=pd.Timestamp('2025-01-30 16:00:00'),
        symbol='AAPL', open=prices[-1], high=prices[-1],
        low=prices[-1], close=prices[-1], volume=1000,
    )}
    sizer = VolTargetSizer(target_vol=0.15, min_forecast=0.05)
    assert sizer.size({'AAPL': 0.01}, hm, 100_000.0, latest) == {}


def test_negative_forecast_short():
    prices = _random_walk(30)
    hm = _make_history('AAPL', prices)
    latest = {'AAPL': Bar(
        timestamp=pd.Timestamp('2025-01-30 16:00:00'),
        symbol='AAPL', open=prices[-1], high=prices[-1],
        low=prices[-1], close=prices[-1], volume=1000,
    )}
    sizer = VolTargetSizer(target_vol=0.15)
    targets = sizer.size({'AAPL': -0.5}, hm, 100_000.0, latest)
    assert targets.get('AAPL', 0) < 0


def test_max_leverage_caps():
    # Low vol stock → high vol_scalar, but max_leverage should cap it
    prices = _random_walk(30, daily_vol=0.005)
    hm = _make_history('AAPL', prices)
    latest = {'AAPL': Bar(
        timestamp=pd.Timestamp('2025-01-30 16:00:00'),
        symbol='AAPL', open=prices[-1], high=prices[-1],
        low=prices[-1], close=prices[-1], volume=1000,
    )}
    sizer = VolTargetSizer(target_vol=0.15, vol_lookback=20, max_leverage=0.5)
    targets = sizer.size({'AAPL': 1.0}, hm, 100_000.0, latest)
    if 'AAPL' in targets:
        pos_value = abs(targets['AAPL']) * prices[-1]
        assert pos_value <= 100_000.0 * 0.5 * 1.02  # small tolerance


def test_insufficient_history_skips():
    """Only 2 bars → too few for vol_lookback=20."""
    hm = _make_history('AAPL', [100.0, 101.0])
    latest = {'AAPL': Bar(
        timestamp=pd.Timestamp('2025-01-02 16:00:00'),
        symbol='AAPL', open=101, high=101, low=101, close=101, volume=1000,
    )}
    sizer = VolTargetSizer(vol_lookback=20)
    assert sizer.size({'AAPL': 1.0}, hm, 100_000.0, latest) == {}


def test_empty_forecasts():
    hm = _make_history('AAPL', _random_walk(30))
    assert VolTargetSizer().size({}, hm, 100_000.0, {}) == {}
