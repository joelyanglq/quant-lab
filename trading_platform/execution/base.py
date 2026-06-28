"""ExecutionHandler ABC — same interface across backtest, shadow, paper, live.

Strategies emit OrderEvent via submit_order(); handler returns an OrderId
synchronously. Fills arrive asynchronously through the on_fill() callback
mechanism — backtest fills happen on next bar in process_bar(); live fills
arrive via IBKR callbacks.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

from ..core.events import Bar, FillEvent, OrderEvent

OrderId = str
FillCallback = Callable[[FillEvent], None]


class ExecutionHandler(ABC):
    @abstractmethod
    def submit_order(self, order: OrderEvent) -> OrderId: ...

    @abstractmethod
    def cancel_order(self, order_id: OrderId) -> None: ...

    @abstractmethod
    def get_open_orders(self) -> list[OrderEvent]: ...

    @abstractmethod
    def get_positions(self) -> dict[str, float]: ...

    def on_fill(self, callback: FillCallback) -> None:
        """Register a fill callback. Multiple callbacks supported."""
        if not hasattr(self, "_fill_callbacks"):
            self._fill_callbacks: list[FillCallback] = []
        self._fill_callbacks.append(callback)

    def _emit_fill(self, fill: FillEvent) -> None:
        for cb in getattr(self, "_fill_callbacks", []):
            try:
                cb(fill)
            except Exception:
                import logging
                logging.getLogger(__name__).exception("Fill callback failed")

    def process_bar(self, bar: Bar) -> None:
        """Optional hook — backtest handler uses this to flush pending orders."""
        pass
