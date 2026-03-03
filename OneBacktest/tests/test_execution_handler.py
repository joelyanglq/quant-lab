import queue

import pandas as pd

from data.types import Bar
from event import EventType, OrderEvent, OrderSide, OrderType
from execution.handler import SimulatedExecutionHandler


def test_simulated_execution_generates_fill_from_latest_close():
    latest = {
        "AAPL": Bar(
            timestamp=pd.Timestamp("2025-01-02"),
            symbol="AAPL",
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000,
        )
    }
    handler = SimulatedExecutionHandler(latest)
    handler.events = queue.Queue()

    order = OrderEvent(
        timestamp=pd.Timestamp("2025-01-02"),
        order_id="o1",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10,
    )
    handler.execute_order(order)

    assert not handler.events.empty()
    fill = handler.events.get_nowait()
    assert fill.event_type == EventType.FILL
    assert fill.symbol == "AAPL"
    assert fill.fill_quantity == 10
    assert fill.fill_price == 100.5
    assert fill.side == OrderSide.BUY
