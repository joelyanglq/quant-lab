"""Bar aggregation from tick/trade data."""
from typing import Dict, List
from collections import defaultdict
import pandas as pd

from qedp.events.base import MarketEvent


class BarAggregator:
    """
    Aggregates tick/trade data into bars of various frequencies.
    
    Supports multiple frequencies simultaneously (1s, 5s, 1m, etc.)
    """
    
    def __init__(
        self,
        frequencies: List[str],  # e.g., ["1s", "5s", "1m"]
        alignment: str = "right"  # "left" or "right" closed
    ):
        self.frequencies = [pd.Timedelta(f) for f in frequencies]
        self.alignment = alignment
        
        # Buffer for each symbol and frequency
        self._buffers: Dict[str, Dict[pd.Timedelta, List[MarketEvent]]] = defaultdict(
            lambda: defaultdict(list)
        )
        
        # Last bar timestamp for each symbol and frequency
        self._last_bar_ts: Dict[str, Dict[pd.Timedelta, pd.Timestamp]] = defaultdict(dict)
    
    def process_tick(self, tick: MarketEvent) -> List[MarketEvent]:
        """
        Process a tick and return any completed bars.
        
        Args:
            tick: Tick-level MarketEvent
            
        Returns:
            List of bar MarketEvents (one per frequency if bar completed)
        """
        if tick.etype != "tick":
            return []
        
        symbol = tick.symbol
        bars = []
        
        for freq in self.frequencies:
            # Determine bar boundary
            bar_ts = self._get_bar_timestamp(tick.ts, freq)
            
            # Check if we crossed into a new bar
            last_ts = self._last_bar_ts[symbol].get(freq)
            
            if last_ts is not None and bar_ts > last_ts:
                # Emit completed bar
                bar_event = self._create_bar(symbol, last_ts, freq)
                if bar_event:
                    bars.append(bar_event)
                
                # Clear buffer
                self._buffers[symbol][freq].clear()
            
            # Add tick to buffer
            self._buffers[symbol][freq].append(tick)
            self._last_bar_ts[symbol][freq] = bar_ts
        
        return bars
    
    def _get_bar_timestamp(self, ts: pd.Timestamp, freq: pd.Timedelta) -> pd.Timestamp:
        """Get the bar timestamp for a given tick timestamp."""
        if self.alignment == "right":
            # Right-closed: bar ends at timestamp
            return ts.floor(freq) + freq
        else:
            # Left-closed: bar starts at timestamp
            return ts.floor(freq)
    
    def _create_bar(self, symbol: str, bar_ts: pd.Timestamp, freq: pd.Timedelta) -> MarketEvent | None:
        """Create a bar event from buffered ticks."""
        ticks = self._buffers[symbol][freq]
        if not ticks:
            return None
        
        # Extract prices from tick payloads
        prices = [t.payload.get('price', 0) for t in ticks if 'price' in t.payload]
        volumes = [t.payload.get('volume', 0) for t in ticks if 'volume' in t.payload]
        
        if not prices:
            return None
        
        # Create OHLCV bar
        bar_payload = {
            'open': prices[0],
            'high': max(prices),
            'low': min(prices),
            'close': prices[-1],
            'volume': sum(volumes) if volumes else 0,
            'frequency': str(freq)
        }
        
        return MarketEvent(
            ts=bar_ts,
            symbol=symbol,
            etype="bar",
            payload=bar_payload
        )
    
    def flush(self, symbol: str | None = None) -> List[MarketEvent]:
        """
        Flush remaining bars (e.g., at end of session).
        
        Args:
            symbol: Symbol to flush (None = all symbols)
            
        Returns:
            List of final bar events
        """
        bars = []
        symbols = [symbol] if symbol else list(self._buffers.keys())
        
        for sym in symbols:
            for freq in self.frequencies:
                last_ts = self._last_bar_ts[sym].get(freq)
                if last_ts:
                    bar_event = self._create_bar(sym, last_ts, freq)
                    if bar_event:
                        bars.append(bar_event)
        
        return bars
