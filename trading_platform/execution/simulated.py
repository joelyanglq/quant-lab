"""SimulatedExecutionHandler — backtest fills at next bar's close + slippage.

Slippage models:
    'half_spread'   — fill = close + side * 0.5 * estimated_spread
    'atr_pct'       — fill = close + side * atr_pct * close
    'none'          — fill = close (zero slippage)

Commission model: IB Pro tiered, $0.005/share with $1 minimum, 1% notional cap.
"""
from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Optional

from ..core.events import Bar, FillEvent, OrderEvent, OrderSide, OrderStatus, OrderType
from .base import ExecutionHandler, OrderId


class SimulatedExecutionHandler(ExecutionHandler):
    def __init__(
        self,
        slippage_model: str = "half_spread",
        slippage_bps: float = 5.0,  # 5 bps each side for half_spread
        commission_per_share: float = 0.005,
        commission_min: float = 1.0,
    ):
        self.slippage_model = slippage_model
        self.slippage_bps = slippage_bps
        self.commission_per_share = commission_per_share
        self.commission_min = commission_min
        self._pending: list[OrderEvent] = []
        self._positions: dict[str, float] = defaultdict(float)
        self._fill_callbacks: list = []

    def submit_order(self, order: OrderEvent) -> OrderId:
        order_id = str(uuid.uuid4())
        self._pending.append(
            OrderEvent(
                timestamp=order.timestamp,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                order_type=order.order_type,
                limit_price=order.limit_price,
                strategy_id=order.strategy_id,
                order_id=order_id,
            )
        )
        return order_id

    def cancel_order(self, order_id: OrderId) -> None:
        self._pending = [o for o in self._pending if o.order_id != order_id]

    def get_open_orders(self) -> list[OrderEvent]:
        return list(self._pending)

    def get_positions(self) -> dict[str, float]:
        return dict(self._positions)

    def process_bar(self, bar: Bar) -> None:
        """Fill any pending order for this symbol at this bar's close."""
        still_pending = []
        for order in self._pending:
            if order.symbol != bar.symbol:
                still_pending.append(order)
                continue
            # Fill at this bar's close (assumes order submitted at previous bar).
            fill_price = self._apply_slippage(bar.close, order.side, bar)
            if order.order_type == OrderType.LMT and order.limit_price is not None:
                # Naive: only fill if limit price reached intra-bar.
                if order.side == OrderSide.BUY and order.limit_price < bar.low:
                    still_pending.append(order)
                    continue
                if order.side == OrderSide.SELL and order.limit_price > bar.high:
                    still_pending.append(order)
                    continue
                fill_price = order.limit_price
            commission = self._commission(order.quantity, fill_price)
            sign = 1.0 if order.side == OrderSide.BUY else -1.0
            self._positions[order.symbol] += sign * order.quantity
            fill = FillEvent(
                timestamp=bar.timestamp,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                fill_price=fill_price,
                commission=commission,
                slippage=abs(fill_price - bar.close),
                order_id=order.order_id,
                strategy_id=order.strategy_id,
            )
            self._emit_fill(fill)
        self._pending = still_pending

    def _apply_slippage(self, close: float, side: OrderSide, bar: Bar) -> float:
        if self.slippage_model == "none":
            return close
        if self.slippage_model == "half_spread":
            offset = close * (self.slippage_bps / 1e4)
        elif self.slippage_model == "atr_pct":
            atr = max(bar.high - bar.low, 0.0)
            offset = atr * 0.5
        else:
            offset = 0.0
        return close + offset if side == OrderSide.BUY else close - offset

    def _commission(self, qty: float, price: float) -> float:
        comm = qty * self.commission_per_share
        comm = max(comm, self.commission_min)
        comm = min(comm, 0.01 * qty * price)
        return comm
