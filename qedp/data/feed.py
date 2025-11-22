"""Data feed abstractions for market data."""
from abc import ABC, abstractmethod
from typing import List, Callable, Iterator
import pandas as pd

from qedp.events.base import MarketEvent


class DataFeed(ABC):
    """
    Abstract base class for data feeds.
    
    Supports both pull (iterator) and push (callback) patterns.
    """
    
    def __init__(self, name: str):
        self.name = name
        self._subscribers: List[Callable[[MarketEvent], None]] = []
    
    @abstractmethod
    def subscribe(self, symbol: str, fields: List[str] | None = None, depth: int = 1):
        """
        Subscribe to market data for a symbol.
        
        Args:
            symbol: Symbol to subscribe to
            fields: List of fields to include (None = all)
            depth: Order book depth (for book data)
        """
        pass
    
    @abstractmethod
    def __iter__(self) -> Iterator[MarketEvent]:
        """Return iterator over market events."""
        pass
    
    def add_subscriber(self, callback: Callable[[MarketEvent], None]):
        """Add a callback to be notified of market events."""
        self._subscribers.append(callback)
    
    def _notify(self, event: MarketEvent):
        """Notify all subscribers of a market event."""
        for callback in self._subscribers:
            callback(event)


class FileFeed(DataFeed):
    """
    File-based data feed (CSV, Parquet, HDF5).
    
    Loads data from files and replays it as MarketEvent stream.
    """
    
    def __init__(
        self,
        name: str,
        file_path: str,
        file_format: str = "parquet",
        symbols: List[str] | None = None
    ):
        super().__init__(name)
        self.file_path = file_path
        self.file_format = file_format
        self.symbols = symbols or []
        self._data: pd.DataFrame | None = None
        self._iterator: Iterator | None = None
        
    def subscribe(self, symbol: str, fields: List[str] | None = None, depth: int = 1):
        """Subscribe to symbol data."""
        if symbol not in self.symbols:
            self.symbols.append(symbol)
    
    def _load_data(self):
        """Load data from file."""
        if self._data is not None:
            return
            
        if self.file_format == "parquet":
            self._data = pd.read_parquet(self.file_path)
        elif self.file_format == "csv":
            self._data = pd.read_csv(self.file_path, parse_dates=['timestamp'])
        elif self.file_format == "hdf5":
            self._data = pd.read_hdf(self.file_path)
        else:
            raise ValueError(f"Unsupported file format: {self.file_format}")
        
        # Ensure timestamp column
        if 'timestamp' not in self._data.columns and self._data.index.name == 'timestamp':
            self._data = self._data.reset_index()
        
        # Sort by timestamp
        self._data = self._data.sort_values('timestamp')
    
    def __iter__(self) -> Iterator[MarketEvent]:
        """Iterate over market events."""
        self._load_data()
        
        for _, row in self._data.iterrows():
            # Convert row to MarketEvent
            event = MarketEvent(
                ts=pd.Timestamp(row['timestamp']),
                symbol=row.get('symbol', self.symbols[0] if self.symbols else 'UNKNOWN'),
                etype="bar",  # Assume bar data from file
                payload={
                    'open': row.get('open'),
                    'high': row.get('high'),
                    'low': row.get('low'),
                    'close': row.get('close'),
                    'volume': row.get('volume'),
                }
            )
            yield event


class ReplayFeed(DataFeed):
    """
    Replay feed that preserves original timestamps.
    
    Used for audit replay and consistency testing.
    """
    
    def __init__(self, name: str, event_log_path: str):
        super().__init__(name)
        self.event_log_path = event_log_path
        self._events: List[MarketEvent] = []
    
    def subscribe(self, symbol: str, fields: List[str] | None = None, depth: int = 1):
        """Subscribe to symbol data."""
        pass  # Replay uses logged events
    
    def _load_events(self):
        """Load events from log file."""
        # TODO: Implement JSONL event log parsing
        pass
    
    def __iter__(self) -> Iterator[MarketEvent]:
        """Iterate over logged events."""
        self._load_events()
        for event in self._events:
            yield event
