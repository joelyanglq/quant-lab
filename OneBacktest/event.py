from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field

class EventType(Enum):
    MARKET = 'market'
    SIGNAL = 'signal'
    ORDER = 'order'
    FILL = 'fill'

@dataclass(frozen=True)
class BaseEvent:
    """
    事件基类
    
    所有事件都应该继承这个类
    frozen=True 保证事件不可变，避免意外修改
    """
    event_type: EventType
    timestamp: datetime
    event_id: str = field(default_factory=lambda: str(id(object())))
    
    def __repr__(self):
        return f"{self.__class__.__name__}(timestamp={self.timestamp}, id={self.event_id[:8]}...)"



class MarketEvent(BaseEvent):
    def __init__(self):
        super().__init__()


class SignalType(Enum):
    LONG = 'long'
    SHORT = 'short'
    EXIT = 'exit'
    LONG_EXIT = 'long_exit'
    SHORT_EXIT = 'short_exit'


class SignalEvent(BaseEvent):
    symbol: str
    signal_type: SignalType
    strength: float = 1.0  # 信号强度 [0, 1]

    def __init__(self, **kwargs):
        if 'event_type' not in kwargs:
            kwargs['event_type'] = EventType.SIGNAL
        object.__setattr__(self, 'event_type', kwargs.pop('event_type'))
        for key, value in kwargs.items():
            object.__setattr__(self, key, value)


class OrderType(Enum):
    """
    https://www.xs.com/zh-Hans/blog/%E8%82%A1%E7%A5%A8%E4%BA%A4%E6%98%93%E8%AE%A2%E5%8D%95%E7%B1%BB%E5%9E%8B/
    """
    MARKET = 'market'
    LIMIT = 'limit'
    STOP = 'stop'
    STOP_LIMIT = 'stop_limit'
    TRAILING_STOP = 'trailing_stop'  # 追踪止损单
    IOC =  'ioc'  # 立即成交或取消
    GTC = 'gtc'  # 取消前有效
    FOK = 'fok'  # 全部成交或取消
    AON = 'aon'  # 立即全部执行或取消订单

class OrderSide(Enum):
    """订单方向"""
    BUY = 'buy'
    SELL = 'sell'


class OrderEvent(BaseEvent):
    """
    message sent to ExecutionHandler.

    我们需要记录什么：
    订单标号：
    订单类型limit order / market order
    股票代码
    股票数量
    价格限制: limit order
    到期日？
    """
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float

    # 价格信息
    price: Optional[float] = None           # 限价单价格
    stop_price: Optional[float] = None      # 止损价格


class FillEvent(BaseEvent):
    """
    成交事件
    
    订单成交时触发
    记录每一笔成交的详细信息
    """
    def __init__(self, **kwargs):
        if 'event_type' not in kwargs:
            kwargs['event_type'] = EventType.FILL
        object.__setattr__(self, 'event_type', kwargs.pop('event_type'))
        for key, value in kwargs.items():
            object.__setattr__(self, key, value)

    @property
    def total_cost(self) -> float:
        """总成本（包含佣金和滑点）"""
        base_cost = self.fill_price * self.fill_quantity
        if self.side == OrderSide.BUY:
            return base_cost + self.commission + self.slippage
        else:
            return base_cost - self.commission - self.slippage

    