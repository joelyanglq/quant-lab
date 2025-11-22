"""Order lifecycle state machine."""
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List
import pandas as pd

from qedp.events.base import OrderEvent, FillEvent, CancelEvent


class OrderStatus(Enum):
    """Order lifecycle states."""
    NEW = "NEW"
    ACCEPTED = "ACCEPTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class Order:
    """Order state tracking."""
    event: OrderEvent
    status: OrderStatus = OrderStatus.NEW
    filled_qty: int = 0
    avg_fill_price: float = 0.0
    fills: List[FillEvent] = field(default_factory=list)
    created_at: pd.Timestamp = field(default_factory=pd.Timestamp.now)
    updated_at: pd.Timestamp = field(default_factory=pd.Timestamp.now)
    
    @property
    def remaining_qty(self) -> int:
        """Remaining quantity to fill."""
        return self.event.qty - self.filled_qty
    
    @property
    def is_complete(self) -> bool:
        """Check if order is in a terminal state."""
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED
        ]
    
    def add_fill(self, fill: FillEvent):
        """Add a fill to this order."""
        self.fills.append(fill)
        self.filled_qty += fill.fill_qty
        
        # Update average fill price
        total_cost = sum(f.fill_qty * f.fill_px for f in self.fills)
        self.avg_fill_price = total_cost / self.filled_qty if self.filled_qty > 0 else 0.0
        
        # Update status
        if self.filled_qty >= self.event.qty:
            self.status = OrderStatus.FILLED
        elif self.filled_qty > 0:
            self.status = OrderStatus.PARTIALLY_FILLED
        
        self.updated_at = fill.ts


class OrderManager:
    """
    Manages order lifecycle and state transitions.
    
    Tracks all orders and their current status.
    """
    
    def __init__(self):
        self._orders: Dict[str, Order] = {}
        self._order_log: List[Dict] = []
    
    def submit_order(self, order_event: OrderEvent) -> Order:
        """Submit a new order."""
        order = Order(event=order_event, status=OrderStatus.NEW)
        self._orders[order_event.order_id] = order
        self._log_event("submit", order)
        return order
    
    def accept_order(self, order_id: str, ts: pd.Timestamp):
        """Mark order as accepted."""
        if order_id in self._orders:
            order = self._orders[order_id]
            order.status = OrderStatus.ACCEPTED
            order.updated_at = ts
            self._log_event("accept", order)
    
    def fill_order(self, fill_event: FillEvent):
        """Process a fill event."""
        order_id = fill_event.order_id
        if order_id not in self._orders:
            return
        
        order = self._orders[order_id]
        order.add_fill(fill_event)
        self._log_event("fill", order)
    
    def cancel_order(self, cancel_event: CancelEvent):
        """Cancel an order."""
        order_id = cancel_event.order_id
        if order_id not in self._orders:
            return
        
        order = self._orders[order_id]
        if not order.is_complete:
            order.status = OrderStatus.CANCELED
            order.updated_at = cancel_event.ts
            self._log_event("cancel", order)
    
    def reject_order(self, order_id: str, ts: pd.Timestamp, reason: str):
        """Reject an order."""
        if order_id in self._orders:
            order = self._orders[order_id]
            order.status = OrderStatus.REJECTED
            order.updated_at = ts
            self._log_event("reject", order, {"reason": reason})
    
    def get_order(self, order_id: str) -> Order | None:
        """Get order by ID."""
        return self._orders.get(order_id)
    
    def get_active_orders(self) -> List[Order]:
        """Get all non-terminal orders."""
        return [o for o in self._orders.values() if not o.is_complete]
    
    def _log_event(self, event_type: str, order: Order, extra: Dict = None):
        """Log order event for audit trail."""
        log_entry = {
            "ts": order.updated_at,
            "event_type": event_type,
            "order_id": order.event.order_id,
            "symbol": order.event.symbol,
            "status": order.status.value,
            "filled_qty": order.filled_qty,
            "remaining_qty": order.remaining_qty,
        }
        if extra:
            log_entry.update(extra)
        self._order_log.append(log_entry)
    
    def get_order_log(self) -> List[Dict]:
        """Get complete order audit log."""
        return self._order_log.copy()
