"""Monitoring — slippage tracking, order audit log, alert generation."""
from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import Optional

import pandas as pd

from ..core.events import FillEvent, OrderEvent

log = logging.getLogger(__name__)


class SlippageMonitor:
    """Track realized vs expected fill price; alert on regime shift."""

    def __init__(self, expected_bps: float = 5.0, alert_multiplier: float = 2.0,
                 window: int = 7 * 50):  # ~7 trading days at 50 fills/day
        self.expected_bps = expected_bps
        self.alert_multiplier = alert_multiplier
        self._slips: deque = deque(maxlen=window)

    def record_fill(self, expected_price: float, fill_price: float, side_sign: int) -> None:
        if expected_price <= 0:
            return
        bps = (fill_price - expected_price) / expected_price * 1e4 * side_sign
        self._slips.append(bps)
        if len(self._slips) >= 20:
            avg = sum(self._slips) / len(self._slips)
            if avg > self.expected_bps * self.alert_multiplier:
                log.warning("Slippage alert: 7d avg = %.1f bps > %.1f bps",
                            avg, self.expected_bps * self.alert_multiplier)

    @property
    def avg_slippage_bps(self) -> float:
        if not self._slips:
            return 0.0
        return sum(self._slips) / len(self._slips)


class OrderAuditLog:
    """Append-only JSONL log of every order/fill event for audit."""

    def __init__(self, log_dir: str | Path = "runtime/logs"):
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self) -> Path:
        return self.dir / f"orders_{pd.Timestamp.utcnow().date().isoformat()}.jsonl"

    def log_order(self, order: OrderEvent, status: str, extra: Optional[dict] = None) -> None:
        entry = {
            "ts": pd.Timestamp.utcnow().isoformat(),
            "type": "order",
            "status": status,
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side.value if hasattr(order.side, "value") else str(order.side),
            "quantity": order.quantity,
            "order_type": order.order_type.value if hasattr(order.order_type, "value") else str(order.order_type),
            "limit_price": order.limit_price,
            "strategy_id": order.strategy_id,
            **(extra or {}),
        }
        self._append(entry)

    def log_fill(self, fill: FillEvent) -> None:
        entry = {
            "ts": pd.Timestamp.utcnow().isoformat(),
            "type": "fill",
            "order_id": fill.order_id,
            "symbol": fill.symbol,
            "side": fill.side.value if hasattr(fill.side, "value") else str(fill.side),
            "quantity": fill.quantity,
            "fill_price": fill.fill_price,
            "commission": fill.commission,
            "slippage": fill.slippage,
            "strategy_id": fill.strategy_id,
        }
        self._append(entry)

    def _append(self, entry: dict) -> None:
        with self._path().open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
