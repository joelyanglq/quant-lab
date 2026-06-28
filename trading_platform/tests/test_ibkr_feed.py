"""Test IBKR live data feed — verify LiveDataFeed receives real-time bars.

Requires IBKR TWS / Gateway running on port 7497.
Run with: pytest -m ibkr -v
"""
from __future__ import annotations

import time

import pandas as pd
import pytest

from trading_platform.core.events import Bar

try:
    from trading_platform.data.live_feed import LiveDataFeed
    HAS_IB = True
except ImportError:
    HAS_IB = False


pytestmark = pytest.mark.ibkr


@pytest.mark.skipif(not HAS_IB, reason="ib_insync not installed")
def test_live_feed_receives_bars(require_ibkr):
    """Connect to IBKR, subscribe AAPL, receive at least 1 bar within 90s."""
    feed = LiveDataFeed(
        symbols=["AAPL"],
        port=7497,
        client_id=50,
        bar_size="5 mins",
    )
    feed.connect()
    feed.subscribe()

    received: list[Bar] = []
    deadline = time.time() + 90

    try:
        while time.time() < deadline and len(received) == 0:
            feed.ib.waitOnUpdate(timeout=5)
            for sym, sub in feed._bars_subs:
                while sub:
                    rt_bar = sub.pop(0)
                    bar = Bar(
                        symbol=sym,
                        timestamp=pd.Timestamp(rt_bar.time, tz="UTC"),
                        frequency=feed.frequency,
                        open=float(rt_bar.open_),
                        high=float(rt_bar.high),
                        low=float(rt_bar.low),
                        close=float(rt_bar.close),
                        volume=float(rt_bar.volume or 0.0),
                        source="ibkr_realtime",
                    )
                    received.append(bar)
    finally:
        feed.close()

    assert len(received) >= 1, "No bars received within 90s timeout"

    bar = received[0]
    assert bar.symbol == "AAPL"
    assert isinstance(bar.timestamp, pd.Timestamp)
    assert bar.timestamp.tzinfo is not None
    assert bar.open > 0
    assert bar.high >= bar.low
    assert bar.close > 0
    assert bar.volume >= 0
