from collections import deque
from typing import List, Dict
import numpy as np
import queue

from strategy.base import Strategy
from data.types import Bar
from event import SignalEvent, SignalType


class SimpleMAStrategy(Strategy):
    """
    简单均线交叉策略

    内部缓存每个 symbol 的 close 序列，计算短期/长期均线交叉。
    """

    def __init__(self, symbols: List[str], short_window=50, long_window=200):
        self.symbols = symbols
        self.short_window = short_window
        self.long_window = long_window
        self.events: queue.Queue = None  # 由 Engine 注入

        # 每个 symbol 缓存 close 价格
        self._closes: Dict[str, deque] = {
            s: deque(maxlen=long_window + 1) for s in symbols
        }
        self.bought = {s: 'OUT' for s in symbols}

    def calculate_signals(self, bar: Bar):
        if bar.symbol not in self._closes:
            return

        self._closes[bar.symbol].append(bar.close)
        closes = self._closes[bar.symbol]

        if len(closes) <= self.long_window:
            return

        short_ma = np.mean(list(closes)[-self.short_window:])
        long_ma = np.mean(list(closes)[-self.long_window:])
        s = bar.symbol

        if short_ma > long_ma and self.bought[s] == 'OUT':
            print(f"LONG Signal for {s} at {bar.timestamp}")
            signal = SignalEvent(
                symbol=s, timestamp=bar.timestamp,
                signal_type=SignalType.LONG, strength=1.0,
            )
            self.events.put(signal)
            self.bought[s] = 'LONG'

        elif short_ma < long_ma and self.bought[s] == 'LONG':
            print(f"EXIT Signal for {s} at {bar.timestamp}")
            signal = SignalEvent(
                symbol=s, timestamp=bar.timestamp,
                signal_type=SignalType.EXIT, strength=1.0,
            )
            self.events.put(signal)
            self.bought[s] = 'OUT'
