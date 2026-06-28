"""LiveExecutionHandler — IBKR live order submission via ib_insync.

Orders are placed immediately upon submit_order(); fills arrive via
IBKR callbacks and are forwarded to registered fill callbacks (Portfolio,
risk manager, etc.).

Port distinguishes paper (7497) from live (7496). Live mode requires
explicit confirmation via runtime CLI flag.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from typing import Optional

import pandas as pd

from ..core.events import FillEvent, OrderEvent, OrderSide, OrderStatus, OrderType
from .base import ExecutionHandler, OrderId

log = logging.getLogger(__name__)

try:
    from ib_insync import IB, Stock, MarketOrder, LimitOrder, Trade
    HAS_IB = True
except Exception:
    HAS_IB = False


class LiveExecutionHandler(ExecutionHandler):
    """IBKR live execution.

    Args:
        port: 7497 paper, 7496 live.
        client_id: unique IBKR client id per process.
        host: IBKR Gateway/TWS host (default 127.0.0.1).
        confirm_live: must be True for port 7496 (real-money guard).
    """

    def __init__(
        self,
        port: int = 7497,
        client_id: int = 2,
        host: str = "127.0.0.1",
        confirm_live: bool = False,
    ):
        if not HAS_IB:
            raise ImportError("ib_insync required for LiveExecutionHandler")
        if port == 7496 and not confirm_live:
            raise RuntimeError(
                "Refusing to connect to live port 7496 without confirm_live=True. "
                "Pass --i-understand-this-uses-real-money to runtime."
            )
        self.port = port
        self.client_id = client_id
        self.host = host
        self.ib = IB()
        self._trades_by_id: dict[OrderId, Trade] = {}
        self._fill_callbacks: list = []
        self._lock = threading.Lock()

    # ── connection ──────────────────────────────────────────────────
    def connect(self, retry_seconds: tuple[int, ...] = (1, 2, 4, 8, 16, 30)) -> None:
        last_err = None
        for d in retry_seconds:
            try:
                self.ib.connect(self.host, self.port, clientId=self.client_id, timeout=10)
                log.info("LiveExecutionHandler connected to %s:%s", self.host, self.port)
                self.ib.execDetailsEvent += self._on_exec_details
                return
            except Exception as e:
                last_err = e
                log.warning("connect failed (%s); retry in %ds", e, d)
                time.sleep(d)
        raise ConnectionError(f"IBKR connect failed: {last_err}")

    def disconnect(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()

    def is_connected(self) -> bool:
        return self.ib.isConnected()

    # ── order submission ────────────────────────────────────────────
    def submit_order(self, order: OrderEvent) -> OrderId:
        if not self.is_connected():
            raise ConnectionError("IBKR not connected; cannot submit order")
        contract = Stock(order.symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)
        action = "BUY" if order.side == OrderSide.BUY else "SELL"
        if order.order_type == OrderType.LMT and order.limit_price is not None:
            ib_order = LimitOrder(action, order.quantity, order.limit_price)
        else:
            ib_order = MarketOrder(action, order.quantity)
        trade = self.ib.placeOrder(contract, ib_order)
        order_id = str(trade.order.orderId)
        with self._lock:
            self._trades_by_id[order_id] = trade
        log.info("Submitted %s %s %d @ %s -> orderId=%s",
                 action, order.symbol, order.quantity, order.order_type, order_id)
        return order_id

    def cancel_order(self, order_id: OrderId) -> None:
        with self._lock:
            trade = self._trades_by_id.get(order_id)
        if trade is not None:
            self.ib.cancelOrder(trade.order)
            log.info("Cancelled order %s", order_id)

    def get_open_orders(self) -> list[OrderEvent]:
        out = []
        with self._lock:
            for oid, trade in self._trades_by_id.items():
                if trade.orderStatus.status in ("Submitted", "PreSubmitted", "PendingSubmit"):
                    out.append(self._trade_to_order_event(oid, trade))
        return out

    def get_positions(self) -> dict[str, float]:
        positions = {}
        for p in self.ib.positions():
            positions[p.contract.symbol] = float(p.position)
        return positions

    @staticmethod
    def _trade_to_order_event(order_id: OrderId, trade: Trade) -> OrderEvent:
        side = OrderSide.BUY if trade.order.action == "BUY" else OrderSide.SELL
        otype = OrderType.LMT if "Limit" in str(trade.order.orderType) else OrderType.MKT
        return OrderEvent(
            timestamp=pd.Timestamp.utcnow(),
            symbol=trade.contract.symbol,
            side=side,
            quantity=float(trade.order.totalQuantity),
            order_type=otype,
            limit_price=getattr(trade.order, "lmtPrice", None) or None,
            order_id=order_id,
        )

    # ── fill callback ───────────────────────────────────────────────
    def _on_exec_details(self, trade: Trade, fill) -> None:
        try:
            event = FillEvent(
                timestamp=pd.Timestamp(fill.time).tz_localize("UTC")
                          if fill.time and fill.time.tzinfo is None else pd.Timestamp(fill.time),
                symbol=trade.contract.symbol,
                side=OrderSide.BUY if trade.order.action == "BUY" else OrderSide.SELL,
                quantity=float(fill.execution.shares),
                fill_price=float(fill.execution.price),
                commission=float(getattr(fill.commissionReport, "commission", 0.0) or 0.0),
                slippage=0.0,
                order_id=str(trade.order.orderId),
            )
            self._emit_fill(event)
        except Exception:
            log.exception("Failed to process exec details")
