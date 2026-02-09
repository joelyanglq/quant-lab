from typing import List
import queue

from strategy.base import Strategy
from data.types import Bar
from event import SignalEvent, SignalType


class BuyAndHoldStrategy(Strategy):
    """
    买入持有策略

    第一根 bar 出现时买入，持有到结束。
    """

    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.events: queue.Queue = None  # 由 Engine 注入
        self.bought = {s: False for s in symbols}

    def calculate_signals(self, bar: Bar):
        s = bar.symbol
        if s not in self.bought:
            return

        if not self.bought[s]:
            signal = SignalEvent(
                symbol=s, timestamp=bar.timestamp,
                signal_type=SignalType.LONG, strength=1.0,
            )
            self.events.put(signal)
            self.bought[s] = True
