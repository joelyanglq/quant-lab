"""数据结构定义"""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

import pandas as pd


class Frequency(Enum):
    """数据频率"""
    TICK = 'tick'
    MIN_1 = '1m'
    MIN_5 = '5m'
    MIN_15 = '15m'
    MIN_30 = '30m'
    HOUR_1 = '1h'
    HOUR_4 = '4h'
    DAY = '1d'
    WEEK = '1w'
    MONTH = '1M'


class AdjustType(Enum):
    """复权类型"""
    NONE = 'none'
    FORWARD = 'qfq'    # 前复权
    BACKWARD = 'hfq'   # 后复权


@dataclass(frozen=True)
class Bar:
    """
    K线数据（不可变）

    标准化的 OHLCV 数据格式
    frozen=True 保证数据不可变
    """
    timestamp: pd.Timestamp
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int

    # 可选字段
    turnover: Optional[float] = None      # 成交额
    open_interest: Optional[int] = None   # 持仓量（期货）

    def __post_init__(self):
        """数据验证"""
        if self.high < self.low:
            raise ValueError(f"Invalid bar: high({self.high}) < low({self.low}) at {self.timestamp}")
        if self.high < max(self.open, self.close):
            raise ValueError(f"Invalid bar: high < open/close at {self.timestamp}")
        if self.low > min(self.open, self.close):
            raise ValueError(f"Invalid bar: low > open/close at {self.timestamp}")

    def __lt__(self, other):
        """支持 heapq 排序（按时间戳）"""
        return self.timestamp < other.timestamp

    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'turnover': self.turnover,
            'open_interest': self.open_interest,
        }
