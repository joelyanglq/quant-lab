"""Shared test fixtures — synthetic bar streams, mock DataContext, IBKR helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_platform.core.events import Bar, Frequency, OrderEvent, OrderSide, OrderType


def pytest_configure(config):
    config.addinivalue_line("markers", "ibkr: requires IBKR TWS/Gateway on port 7497")
    config.addinivalue_line("markers", "ibkr_live: requires IBKR live on port 7496 (real money)")


@pytest.fixture
def synthetic_bars():
    """Build 500 days of 2-symbol bars with deterministic returns."""
    dates = pd.bdate_range("2020-01-01", periods=500, tz="UTC")
    rng = np.random.default_rng(42)
    aapl = 100 + np.cumsum(rng.normal(0.05, 1.0, len(dates)))
    msft = 200 + np.cumsum(rng.normal(0.05, 1.5, len(dates)))
    bars = []
    for ts, pa, pm in zip(dates, aapl, msft):
        bars.append(Bar("AAPL", ts, Frequency.EOD, pa, pa * 1.01, pa * 0.99, pa, 1e6))
        bars.append(Bar("MSFT", ts, Frequency.EOD, pm, pm * 1.01, pm * 0.99, pm, 1e6))
    return bars


class DictContext:
    """In-memory DataContext for tests."""

    def __init__(self, data: dict):
        self._data = data
        self._syms = data.get("universe", [])

    def universe(self, dt):
        return list(self._syms)

    def as_of(self, dt, key, **kw):
        v = self._data.get(key)
        if callable(v):
            return v(dt, **kw)
        if isinstance(v, pd.DataFrame):
            return v[v.index <= dt]
        return v


@pytest.fixture
def dict_ctx():
    return DictContext


# ── IBKR helpers ───────────────────────────────────────────────────────


def make_test_order(
    symbol: str = "AAPL",
    side: OrderSide = OrderSide.BUY,
    qty: float = 1,
    order_type: OrderType = OrderType.MKT,
    strategy_id: str = "test_order",
    limit_price: float | None = None,
) -> OrderEvent:
    return OrderEvent(
        timestamp=pd.Timestamp.utcnow(),
        symbol=symbol,
        side=side,
        quantity=qty,
        order_type=order_type,
        limit_price=limit_price,
        strategy_id=strategy_id,
    )


def _try_ibkr_connect(port: int, client_id: int = 99) -> bool:
    try:
        from ib_insync import IB
        ib = IB()
        ib.connect("127.0.0.1", port, clientId=client_id, timeout=5)
        ib.disconnect()
        return True
    except Exception:
        return False


@pytest.fixture
def require_ibkr():
    if not _try_ibkr_connect(7497):
        pytest.skip("IBKR TWS/Gateway not available on port 7497")


@pytest.fixture
def require_ibkr_live():
    if not _try_ibkr_connect(7496):
        pytest.skip("IBKR live gateway not available on port 7496")
