from collections import deque
from typing import List, Dict
import numpy as np

from strategy.base import Strategy
from data.types import Bar


class SimpleMAStrategy(Strategy):
    """
    简单均线交叉策略

    短期均线上穿长期均线时买入，下穿时平仓。
    """

    def __init__(self, symbols: List[str], short_window=50, long_window=200):
        self.symbols = symbols
        self.short_window = short_window
        self.long_window = long_window

        self._closes: Dict[str, deque] = {
            s: deque(maxlen=long_window + 1) for s in symbols
        }

    def on_bar(self, bar: Bar):
        s = bar.symbol
        if s not in self._closes:
            return

        self._closes[s].append(bar.close)
        closes = self._closes[s]

        if len(closes) <= self.long_window:
            return

        short_ma = np.mean(list(closes)[-self.short_window:])
        long_ma = np.mean(list(closes)[-self.long_window:])
        pos = self.get_position(s)

        if short_ma > long_ma and pos == 0:
            print(f"LONG: {s} at {bar.timestamp}")
            self.buy(s, 100)

        elif short_ma < long_ma and pos > 0:
            print(f"EXIT: {s} at {bar.timestamp}")
            self.sell(s, pos)
