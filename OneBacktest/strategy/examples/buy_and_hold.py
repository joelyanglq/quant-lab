from datetime import datetime
import queue

from strategy.base import Strategy
from event import SignalEvent, MarketEvent, SignalType

class BuyAndHoldStrategy(Strategy):
    """
    Buy and Hold Strategy.
    Buys on the first available bar and holds until the end.
    """
    def __init__(self, data_handler, events: queue.Queue):
        self.data_handler = data_handler
        self.events = events
        self.symbol_list = self.data_handler.symbols
        self.bought = self._calculate_initial_bought()

    def _calculate_initial_bought(self):
        """
        Adds keys to the bought dictionary for all symbols
        and sets them to False.
        """
        bought = {}
        for s in self.symbol_list:
            bought[s] = False
        return bought

    def calculate_signals(self, event: MarketEvent):
        """
        Generates a new SignalEvent based on the MAC.
        """
        if event.event_type == 'market':
            for s in self.symbol_list:
                if not self.bought[s]:
                    bars = self.data_handler.get_latest_bars(s, N=1)
                    if bars is not None and not bars.empty:
                        # Buy!
                        signal = SignalEvent(symbol=s, timestamp=bars.index[-1], signal_type=SignalType.LONG, strength=1.0)
                        self.events.put(signal)
                        self.bought[s] = True
