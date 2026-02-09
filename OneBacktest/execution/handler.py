from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict
import queue

from data.types import Bar
from event import FillEvent, OrderEvent, EventType, OrderSide


class ExecutionHandler(ABC):
    """执行器抽象基类"""

    @abstractmethod
    def execute_order(self, event: OrderEvent):
        raise NotImplementedError("Should implement execute_order()")


class SimulatedExecutionHandler(ExecutionHandler):
    """
    模拟执行器

    直接以最新 close 价成交，无延迟、无滑点。
    """

    def __init__(self, latest_prices: Dict[str, Bar]):
        self.latest_prices = latest_prices
        self.events: queue.Queue = None  # 由 Engine 注入

    def execute_order(self, event: OrderEvent):
        if event.event_type != EventType.ORDER:
            return

        latest = self.latest_prices.get(event.symbol)
        fill_price = latest.close if latest else 0.0

        fill_event = FillEvent(
            event_id=str(id(object())),
            symbol=event.symbol,
            timestamp=latest.timestamp if latest else datetime.now(),
            fill_quantity=event.quantity,
            side=event.side,
            fill_price=fill_price,
            commission=0.0,
            slippage=0.0,
        )
        self.events.put(fill_event)
