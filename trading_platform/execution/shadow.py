"""ShadowExecutionHandler — connects to IBKR for real-time data but does NOT
submit real orders. Logs would-have-orders to a shadow ledger; daily
reconciliation compares them against simulated orders from EOD bars.

Use case: validate that signal computation works under live data conditions
before risking capital.
"""
from __future__ import annotations

import json
import logging
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd

from ..core.events import Bar, OrderEvent, OrderSide
from .base import ExecutionHandler, OrderId

log = logging.getLogger(__name__)


class ShadowExecutionHandler(ExecutionHandler):
    """Generates orders, logs them, simulates fills at the bar's close."""

    def __init__(self, ledger_path: str | Path = "runtime/state/shadow_ledger.jsonl",
                 commission_per_share: float = 0.005,
                 commission_min: float = 1.0):
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.commission_per_share = commission_per_share
        self.commission_min = commission_min
        self._pending: list[OrderEvent] = []
        self._positions: dict[str, float] = defaultdict(float)
        self._fill_callbacks: list = []

    def submit_order(self, order: OrderEvent) -> OrderId:
        order_id = str(uuid.uuid4())
        new_order = OrderEvent(
            timestamp=order.timestamp,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            order_type=order.order_type,
            limit_price=order.limit_price,
            strategy_id=order.strategy_id,
            order_id=order_id,
        )
        self._pending.append(new_order)
        self._log_to_ledger("submitted", new_order, extra={})
        return order_id

    def cancel_order(self, order_id: OrderId) -> None:
        cancelled = [o for o in self._pending if o.order_id == order_id]
        self._pending = [o for o in self._pending if o.order_id != order_id]
        for o in cancelled:
            self._log_to_ledger("cancelled", o, extra={})

    def get_open_orders(self) -> list[OrderEvent]:
        return list(self._pending)

    def get_positions(self) -> dict[str, float]:
        return dict(self._positions)

    def process_bar(self, bar: Bar) -> None:
        from ..core.events import FillEvent, OrderType

        still = []
        for o in self._pending:
            if o.symbol != bar.symbol:
                still.append(o)
                continue
            fill_price = bar.close
            if o.order_type == OrderType.LMT and o.limit_price is not None:
                if (o.side == OrderSide.BUY and bar.low > o.limit_price) or \
                   (o.side == OrderSide.SELL and bar.high < o.limit_price):
                    still.append(o)
                    continue
                fill_price = o.limit_price
            sign = 1.0 if o.side == OrderSide.BUY else -1.0
            self._positions[o.symbol] += sign * o.quantity
            commission = max(o.quantity * self.commission_per_share, self.commission_min)
            fill = FillEvent(
                timestamp=bar.timestamp,
                symbol=o.symbol,
                side=o.side,
                quantity=o.quantity,
                fill_price=fill_price,
                commission=commission,
                slippage=0.0,
                order_id=o.order_id,
                strategy_id=o.strategy_id,
            )
            self._log_to_ledger("filled", o, extra={"fill_price": fill_price})
            self._emit_fill(fill)
        self._pending = still

    def _log_to_ledger(self, status: str, order: OrderEvent, extra: dict) -> None:
        entry = {
            "ts": order.timestamp.isoformat() if hasattr(order.timestamp, "isoformat") else str(order.timestamp),
            "status": status,
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side.value if hasattr(order.side, "value") else str(order.side),
            "qty": order.quantity,
            "type": order.order_type.value if hasattr(order.order_type, "value") else str(order.order_type),
            "strategy_id": order.strategy_id,
            **extra,
        }
        with self.ledger_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
