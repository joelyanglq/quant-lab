from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum


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
    NONE = 'none'      # 不复权
    FORWARD = 'qfq'    # 前复权
    BACKWARD = 'hfq'   # 后复权


@dataclass
class BarData:
    """
    K线数据结构
    
    标准化的OHLCV数据格式
    """
    symbol: str
    datetime: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    # 可选字段
    turnover: Optional[float] = None      # 成交额
    open_interest: Optional[int] = None   # 持仓量（期货）
    
    def __post_init__(self):
        """数据验证"""
        if self.high < self.low:
            raise ValueError(f"Invalid bar: high < low at {self.datetime}")
        if self.high < self.open or self.high < self.close:
            raise ValueError(f"Invalid bar: high < open/close at {self.datetime}")
        if self.low > self.open or self.low > self.close:
            raise ValueError(f"Invalid bar: low > open/close at {self.datetime}")
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'datetime': self.datetime,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'turnover': self.turnover,
            'open_interest': self.open_interest
        }


class DataFeed(ABC):
    """
    数据源抽象基类
    
    所有数据源都必须实现这个接口
    """
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        frequency: Frequency = Frequency.DAY,
        adjust: AdjustType = AdjustType.NONE
    ) -> pd.DataFrame:
        """
        获取历史K线数据
        
        Returns:
            DataFrame with columns: [open, high, low, close, volume]
            Index: DatetimeIndex
        """
        pass
    
    @abstractmethod
    def get_symbols(self, market: Optional[str] = None) -> List[str]:
        """获取所有可用的股票代码"""
        pass
    
    def get_latest_price(self, symbol: str) -> float:
        """获取最新价格（默认实现）"""
        bars = self.get_bars(
            symbol,
            datetime.now() - timedelta(days=1),
            datetime.now(),
            Frequency.DAY
        )
        if bars.empty:
            raise ValueError(f"No data available for {symbol}")
        return bars.iloc[-1]['close']
    
    def validate_symbol(self, symbol: str) -> bool:
        """验证股票代码是否有效"""
        try:
            self.get_symbols()
            return symbol in self.get_symbols()
        except:
            return False