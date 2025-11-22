from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Dict
import pandas as pd
import queue

from event import MarketEvent
from .manager import DataManager
from .base import Frequency

class DataHandler(ABC):
    """
    DataHandler is an abstract base class providing an interface for
    all subsequent (inherited) data handlers (both live and historic).

    The goal of a (derived) DataHandler object is to output a generated
    set of bars (OHLCVI) for each symbol requested.

    This will replicate how a live strategy would function as current
    market data would be sent "down the pipe". Thus a historic and live
    system will be treated identically by the rest of the backtesting suite.
    """

    @abstractmethod
    def get_latest_bars(self, symbol: str, N: int = 1) -> pd.DataFrame:
        """
        Returns the last N bars from the latest_symbol list,
        or fewer if less bars are available.
        """
        raise NotImplementedError("Should implement get_latest_bars()")

    @abstractmethod
    def update_bars(self):
        """
        Pushes the latest bar to the latest_symbol_data structure
        for all symbols in the symbol list.
        """
        raise NotImplementedError("Should implement update_bars()")

class HistoricDataHandler(DataHandler):
    """
    HistoricDataHandler is designed to read CSV files for
    each requested symbol from disk and provide an interface
    to obtain the "latest" bar in a manner identical to a live
    trading interface.
    """

    def __init__(self, events: queue.Queue, data_manager: DataManager, 
                 symbols: List[str], start: datetime, end: datetime, frequency: Frequency = Frequency.DAY):
        """
        Initializes the historic data handler.

        Parameters:
        events: The Event Queue.
        data_manager: DataManager instance to fetch data.
        symbols: A list of symbol strings.
        start: Start datetime.
        end: End datetime.
        frequency: Data frequency.
        """
        self.events = events
        self.data_manager = data_manager
        self.symbols = symbols
        self.start = start
        self.end = end
        self.frequency = frequency

        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.continue_backtest = True
        self.bar_index = 0
        
        # Load data
        self._load_data()

    def _load_data(self):
        """
        Loads data from DataManager and converts to generator/iterator.
        """
        comb_index = None
        for s in self.symbols:
            # Load data using DataManager
            # Note: DataManager.get_bars returns a DataFrame
            df = self.data_manager.get_bars(s, self.start, self.end, self.frequency)
            
            # Reindex to ensure all symbols have the same index (handling missing data)
            if comb_index is None:
                comb_index = df.index
            else:
                comb_index = comb_index.union(df.index)
            
            self.symbol_data[s] = df
            self.latest_symbol_data[s] = []

        # Reindex all dataframes to the combined index and forward fill
        for s in self.symbols:
            self.symbol_data[s] = self.symbol_data[s].reindex(index=comb_index, method='pad').iterrows()

        self.comb_index = comb_index

    def _get_new_bar(self, symbol):
        """
        Returns the latest bar from the data feed as a tuple of 
        (sybmol, datetime, open, low, high, close, volume).
        """
        try:
            bar = next(self.symbol_data[symbol])
            return bar
        except StopIteration:
            self.continue_backtest = False
            return None

    def get_latest_bars(self, symbol, N=1):
        """
        Returns the last N bars from the latest_symbol list,
        or N-k if less available.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            return []
        else:
            return pd.DataFrame(bars_list[-N:], columns=['open', 'high', 'low', 'close', 'volume', 'turnover', 'open_interest'])

    def update_bars(self):
        """
        Pushes the latest bar to the latest_symbol_data structure
        for all symbols in the symbol list.
        """
        for s in self.symbols:
            try:
                bar = next(self.symbol_data[s])
            except StopIteration:
                self.continue_backtest = False
                return
            else:
                if bar is not None:
                    # bar is a tuple (index, Series)
                    timestamp = bar[0]
                    data = bar[1]
                    
                    # Append to latest_symbol_data
                    self.latest_symbol_data[s].append(data)
                    
                    # Create MarketEvent
                    # Note: We might want to optimize this to send one event per timestamp for all symbols
                    # But for now, one event per symbol per timestamp is fine for simplicity
                    # Actually, usually MarketEvent signals that "new data is available"
                    # It doesn't necessarily need to carry the data if the strategy queries the handler.
                    # But let's stick to the event carrying timestamp.
                    pass
        
        # After updating all symbols for this timestamp, generate a MarketEvent
        # We assume all symbols are aligned on the same timestamp (comb_index)
        if self.continue_backtest:
             # Get the current timestamp from the iterator? 
             # Since we are iterating generators, it's a bit tricky to know the "current" timestamp 
             # if we just called next() on all of them.
             # However, we reindexed them to comb_index.
             
             # Let's verify if we are still in bounds
             if self.bar_index < len(self.comb_index):
                 timestamp = self.comb_index[self.bar_index]
                 self.bar_index += 1
                 self.events.put(MarketEvent(timestamp=timestamp))
             else:
                 self.continue_backtest = False
