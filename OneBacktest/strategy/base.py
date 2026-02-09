from abc import ABC, abstractmethod

from data.types import Bar


class Strategy(ABC):
    """
    策略抽象基类

    所有策略实现 calculate_signals(bar)，
    直接接收 Bar 对象，按需生成 SignalEvent 放入 events 队列。
    """

    @abstractmethod
    def calculate_signals(self, bar: Bar):
        """
        接收新的 Bar，计算并生成交易信号。

        Args:
            bar: 最新的 OHLCV Bar
        """
        raise NotImplementedError("Should implement calculate_signals()")
