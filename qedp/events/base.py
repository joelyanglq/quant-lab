"""
Event type definitions for the EDP framework.

All events inherit from the base Event dataclass and include timestamp and priority.
Priority order: Control > Clock > Fill > Cancel > Order > Signal > Market
"""
from dataclasses import dataclass, field
from typing import Literal, Any
from datetime import datetime
import pandas as pd


# Event priorities (higher number = higher priority)
PRIORITY_CONTROL = 70
PRIORITY_CLOCK = 60
PRIORITY_FILL = 50
PRIORITY_CANCEL = 40
PRIORITY_ORDER = 30
PRIORITY_SIGNAL = 20
PRIORITY_MARKET = 10


@dataclass
class Event:
    """Base event class."""
    ts: pd.Timestamp
    priority: int = 0
    
    def __lt__(self, other):
        """Compare events for priority queue ordering."""
        # First by timestamp (earlier first), then by priority (higher first)
        if self.ts != other.ts:
            return self.ts < other.ts
        return self.priority > other.priority


@dataclass
class MarketEvent(Event):
    """Market data event (tick, bar, or order book update)."""
    symbol: str
    etype: Literal["tick", "bar", "book"]
    payload: dict  # Contains price, volume, bid/ask, etc.
    priority: int = field(default=PRIORITY_MARKET, init=False)


@dataclass
class SignalEvent(Event):
    """Trading signal from strategy."""
    symbol: str
    intent: Literal["open", "close", "reduce", "flip"]
    target: dict  # {"qty": int} or {"weight": float}
    meta: dict = field(default_factory=dict)
    priority: int = field(default=PRIORITY_SIGNAL, init=False)


@dataclass
class OrderEvent(Event):
    """Order submission event."""
    order_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    type: Literal["MKT", "LMT", "STP", "STP_LMT", "ICEBERG"]
    qty: int
    px: float | None = None  # Required for limit orders
    tif: Literal["IOC", "FOK", "DAY", "GTC"] = "DAY"
    meta: dict = field(default_factory=dict)
    priority: int = field(default=PRIORITY_ORDER, init=False)


@dataclass
class CancelEvent(Event):
    """Order cancellation event."""
    order_id: str
    priority: int = field(default=PRIORITY_CANCEL, init=False)


@dataclass
class FillEvent(Event):
    """Order fill/execution event."""
    order_id: str
    symbol: str
    fill_px: float
    fill_qty: int
    fees: float = 0.0
    last_liquidity: Literal["ADD", "REMOVE"] | None = None
    meta: dict = field(default_factory=dict)
    priority: int = field(default=PRIORITY_FILL, init=False)


@dataclass
class AccountEvent(Event):
    """Account status update event."""
    balance: float
    margin: float
    pnl_intraday: float
    risk_limits: dict = field(default_factory=dict)
    priority: int = field(default=PRIORITY_CLOCK, init=False)


@dataclass
class ClockEvent(Event):
    """Time progression and session boundary event."""
    phase: Literal["pre", "open", "close", "after"]
    session_id: str
    priority: int = field(default=PRIORITY_CLOCK, init=False)


@dataclass
class ControlEvent(Event):
    """System control event (pause, resume, stop, reload)."""
    cmd: Literal["pause", "resume", "stop", "reload", "reject"]
    args: dict = field(default_factory=dict)
    priority: int = field(default=PRIORITY_CONTROL, init=False)
