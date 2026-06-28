"""Event types — frozen dataclasses shared across backtest and live runtimes.

All events carry a UTC timestamp. Bar/Order/Fill schemas are identical between
backtest and live (only `source` differs); this lets us run live runtime
unit tests with synthetic injected bars.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd


class Frequency(str, Enum):
    EOD = "1d"
    MIN_1 = "1min"
    MIN_5 = "5min"
    MIN_15 = "15min"
    MIN_30 = "30min"
    HOUR_1 = "1h"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MKT = "MKT"
    LMT = "LMT"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class Bar:
    symbol: str
    timestamp: pd.Timestamp
    frequency: Frequency
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str = "parquet"  # 'parquet' | 'ibkr_realtime'


@dataclass(frozen=True)
class OrderEvent:
    timestamp: pd.Timestamp
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MKT
    limit_price: Optional[float] = None
    strategy_id: Optional[str] = None
    order_id: Optional[str] = None  # filled in by ExecutionHandler


@dataclass(frozen=True)
class FillEvent:
    timestamp: pd.Timestamp
    symbol: str
    side: OrderSide
    quantity: float
    fill_price: float
    commission: float = 0.0
    slippage: float = 0.0
    order_id: Optional[str] = None
    strategy_id: Optional[str] = None


@dataclass(frozen=True)
class SignalEvent:
    """Optional explicit signal event (most strategies skip this and go straight to OrderEvent)."""
    timestamp: pd.Timestamp
    symbol: str
    forecast: float
    strategy_id: str


@dataclass(frozen=True)
class RiskEvent:
    """Emitted by risk subsystem (kill-switch, reconciliation mismatch, anomaly)."""
    timestamp: pd.Timestamp
    severity: str  # 'INFO' | 'WARNING' | 'CRITICAL'
    code: str      # e.g. 'KILL_SWITCH', 'RECON_MISMATCH', 'DISCONNECT'
    message: str
    payload: dict = field(default_factory=dict)
