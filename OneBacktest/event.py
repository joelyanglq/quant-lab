"""事件定义模块"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


# ==================== 枚举类型 ====================

class EventType(Enum):
    MARKET = 'market'
    SIGNAL = 'signal'
    ORDER = 'order'
    FILL = 'fill'
    CANCEL = 'cancel'


class OrderSide(Enum):
    """订单方向"""
    BUY = 'buy'
    SELL = 'sell'


class OrderType(Enum):
    MARKET = 'market'
    LIMIT = 'limit'
    STOP = 'stop'
    STOP_LIMIT = 'stop_limit'


class SignalType(Enum):
    LONG = 'long'
    SHORT = 'short'
    EXIT = 'exit'
    LONG_EXIT = 'long_exit'
    SHORT_EXIT = 'short_exit'


# ==================== 基类 ====================

@dataclass(frozen=True)
class BaseEvent:
    """
    事件基类（不可变）

    所有事件都应该继承这个类
    frozen=True 保证事件不可变，避免意外修改
    """
    timestamp: datetime
    event_id: str = field(default_factory=lambda: str(id(object())))

    def __repr__(self):
        return f"{self.__class__.__name__}(timestamp={self.timestamp}, id={self.event_id[:8]}...)"


# ==================== 具体事件 ====================

@dataclass(frozen=True)
class MarketEvent(BaseEvent):
    """
    市场数据事件（预留，未来实盘推送用）
    """
    symbol: str = ''
    event_type: EventType = field(default=EventType.MARKET, init=False)


@dataclass(frozen=True)
class SignalEvent(BaseEvent):
    """交易信号事件"""
    symbol: str = ''
    signal_type: SignalType = SignalType.LONG
    strength: float = 1.0  # 信号强度 [0, 1]
    event_type: EventType = field(default=EventType.SIGNAL, init=False)


@dataclass(frozen=True)
class OrderEvent(BaseEvent):
    """订单事件"""
    order_id: str = ''
    symbol: str = ''
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: int = 0
    price: Optional[float] = None       # 限价单价格
    stop_price: Optional[float] = None  # 止损价格
    event_type: EventType = field(default=EventType.ORDER, init=False)

    def print_order(self):
        print(
            f'[Order {self.order_id}]: Symbol={self.symbol}, '
            f'Side={self.side.value}, Qty={self.quantity}, '
            f'Type={self.order_type.value}'
        )


@dataclass(frozen=True)
class FillEvent(BaseEvent):
    """
    成交事件

    一个订单可能产生多个 FillEvent（部分成交）
    费用/滑点由执行器计算后填入，此类仅作为数据载体
    """
    # 身份标识
    fill_id: str = ''               # 成交唯一ID
    order_id: str = ''              # 关联的订单ID

    # 成交信息
    symbol: str = ''
    side: OrderSide = OrderSide.BUY
    fill_price: float = 0.0         # 本次成交价格（已含滑点）
    fill_quantity: int = 0          # 本次成交数量

    # 费用信息（由执行器计算）
    commission: float = 0.0
    slippage: float = 0.0

    # 可选字段
    liquidity: str = 'REMOVE'       # 'ADD'(限价) or 'REMOVE'(市价)
    exchange: str = ''              # 交易所标识
    execution_id: str = ''          # 券商返回的成交编号（实盘用）

    event_type: EventType = field(default=EventType.FILL, init=False)

    @property
    def total_cost(self) -> float:
        """总成本（含佣金）"""
        base_cost = self.fill_price * self.fill_quantity
        if self.side == OrderSide.BUY:
            return base_cost + self.commission
        else:
            return base_cost - self.commission

    @property
    def notional(self) -> float:
        """名义价值"""
        return self.fill_price * self.fill_quantity


@dataclass(frozen=True)
class CancelEvent(BaseEvent):
    """订单取消事件"""
    order_id: str = ''
    reason: str = ''
    event_type: EventType = field(default=EventType.CANCEL, init=False)
