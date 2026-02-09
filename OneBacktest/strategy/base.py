from abc import ABC, abstractmethod
from typing import Dict, Optional
import queue

from data.types import Bar
from event import FillEvent, OrderEvent, OrderType, OrderSide


class Strategy(ABC):
    """
    策略抽象基类 — 生命周期回调 + 下单 API

    回调:
        on_init()          引擎启动时调用一次
        on_bar(bar)        新 bar 到达（必须实现）
        on_fill(fill)      成交回报

    下单:
        buy(symbol, qty)   买入
        sell(symbol, qty)  卖出

    查询:
        get_position(symbol)     当前持仓数量
        get_portfolio_value()    组合总市值
        get_cash()               可用现金
    """

    events: queue.Queue = None              # Engine 注入
    positions: Dict[str, int] = None        # Engine 注入（Portfolio.current_positions 引用）
    holdings: dict = None                   # Engine 注入（Portfolio.current_holdings 引用）
    latest_prices: Dict[str, Bar] = None    # Engine 注入

    # ==================== 生命周期回调 ====================

    def on_init(self):
        """引擎启动时调用一次"""
        pass

    @abstractmethod
    def on_bar(self, bar: Bar):
        """新 bar 到达，核心决策逻辑"""
        ...

    def on_fill(self, fill: FillEvent):
        """成交回报"""
        pass

    # ==================== 下单 API ====================

    def buy(self, symbol: str, quantity: int,
            order_type: OrderType = OrderType.MARKET,
            price: Optional[float] = None):
        """买入"""
        self._send_order(symbol, OrderSide.BUY, quantity, order_type, price)

    def sell(self, symbol: str, quantity: int,
             order_type: OrderType = OrderType.MARKET,
             price: Optional[float] = None):
        """卖出"""
        self._send_order(symbol, OrderSide.SELL, quantity, order_type, price)

    def _send_order(self, symbol: str, side: OrderSide, quantity: int,
                    order_type: OrderType, price: Optional[float]):
        latest = self.latest_prices.get(symbol)
        timestamp = latest.timestamp if latest else None
        order = OrderEvent(
            timestamp=timestamp,
            order_id=str(id(object())),
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
        )
        self.events.put(order)

    # ==================== 查询 ====================

    def get_position(self, symbol: str) -> int:
        """查当前持仓数量"""
        if self.positions is None:
            return 0
        return self.positions.get(symbol, 0)

    def get_portfolio_value(self) -> float:
        """查组合总市值（持仓市值 + 现金）"""
        if self.holdings is None:
            return 0.0
        return self.holdings.get('total', 0.0)

    def get_cash(self) -> float:
        """查可用现金"""
        if self.holdings is None:
            return 0.0
        return self.holdings.get('cash', 0.0)
