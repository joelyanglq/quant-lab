from datetime import datetime
import numpy as np
import pandas as pd
import queue

from strategy.base import Strategy
from event import SignalEvent, MarketEvent, SignalType

class SimpleMAStrategy(Strategy):
    """
    Simple Moving Average Crossover Strategy.
    """
    def __init__(self, data_handler, events: queue.Queue, short_window=50, long_window=200):
        self.data_handler = data_handler
        self.events = events
        self.short_window = short_window
        self.long_window = long_window
        self.symbol_list = self.data_handler.symbols
        self.bought = self._calculate_initial_bought()

    def _calculate_initial_bought(self):
        """
        Adds keys to the bought dictionary for all symbols
        and sets them to 'OUT'.
        """
        bought = {}
        for s in self.symbol_list:
            bought[s] = 'OUT'
        return bought

    def calculate_signals(self, event: MarketEvent):
        """
        Generates a new SignalEvent based on the MAC.
        """
        if event.event_type == 'market':
            for s in self.symbol_list:
                bars = self.data_handler.get_latest_bars(s, N=self.long_window+1)
                
                if bars is not None and len(bars) > self.long_window:
                    # Calculate MAs
                    short_ma = np.mean(bars['close'].iloc[-self.short_window:])
                    long_ma = np.mean(bars['close'].iloc[-self.long_window:])
                    
                    dt = bars.index[-1]
                    
                    symbol = s
                    strength = 1.0
                    
                    if short_ma > long_ma and self.bought[s] == 'OUT':
                        print(f"LONG Signal for {s} at {dt}")
                        signal = SignalEvent(symbol=symbol, timestamp=dt, signal_type=SignalType.LONG, strength=strength)
                        self.events.put(signal)
                        self.bought[s] = 'LONG'
                        
                    elif short_ma < long_ma and self.bought[s] == 'LONG':
                        print(f"EXIT Signal for {s} at {dt}")
                        signal = SignalEvent(symbol=symbol, timestamp=dt, signal_type=SignalType.EXIT, strength=strength)
                        self.events.put(signal)
                        self.bought[s] = 'OUT'
