"""
数据源抽象接口

回测和实盘都实现这个接口
核心理念：Iterator 模式，按时间顺序逐个返回 Bar
"""
from abc import ABC, abstractmethod
from typing import Optional, List

import pandas as pd

from .types import Bar


class DataFeed(ABC):
    """
    数据源抽象基类

    所有数据源（历史回放、实时流、审计回放）都实现这个接口
    """

    @abstractmethod
    def subscribe(self, symbols: List[str], start: pd.Timestamp, end: pd.Timestamp):
        """
        订阅数据流

        Args:
            symbols: 标的列表
            start: 开始时间
            end: 结束时间
        """
        pass

    @abstractmethod
    def next(self) -> Optional[Bar]:
        """
        返回下一个 Bar（按时间顺序）

        多标的数据会按时间戳自动排序
        Returns:
            Bar 对象，数据结束则返回 None
        """
        pass

    @abstractmethod
    def has_next(self) -> bool:
        """是否还有数据"""
        pass

    def reset(self):
        """重置数据流（用于多次回测）"""
        pass

    def get_current_time(self) -> Optional[pd.Timestamp]:
        """获取当前时间"""
        return None
