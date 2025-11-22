"""YFinance data feed implementation."""
import yfinance as yf
import pandas as pd
from typing import List, Iterator

from qedp.data.feed import DataFeed
from qedp.events.base import MarketEvent


class YFinanceDataFeed(DataFeed):
    """
    Data feed using Yahoo Finance.
    
    Downloads historical data and streams it as MarketEvents.
    """
    
    def __init__(
        self,
        symbols: List[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        interval: str = "1d"
    ):
        super().__init__("yfinance")
        self.symbols = symbols
        self.start = start
        self.end = end
        self.interval = interval
        self._data: Dict[str, pd.DataFrame] = {}
        
    def subscribe(self, symbol: str, fields: List[str] | None = None, depth: int = 1):
        """Subscribe to symbol data."""
        if symbol not in self.symbols:
            self.symbols.append(symbol)
    
    def _download_data(self):
        """Download data from Yahoo Finance."""
        if self._data:
            return  # Already downloaded
        
        for symbol in self.symbols:
            df = yf.download(
                symbol,
                start=self.start,
                end=self.end,
                interval=self.interval,
                auto_adjust=True,
                progress=False
            )
            
            if not df.empty:
                # Standardize column names
                df = df.rename(columns={
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                df['symbol'] = symbol
                df = df.reset_index()
                df = df.rename(columns={'Date': 'timestamp', 'Datetime': 'timestamp'})
                self._data[symbol] = df
    
    def __iter__(self) -> Iterator[MarketEvent]:
        """Iterate over market events."""
        self._download_data()
        
        # Merge all symbols and sort by timestamp
        all_data = []
        for symbol, df in self._data.items():
            all_data.append(df)
        
        if not all_data:
            return
        
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.sort_values('timestamp')
        
        for _, row in combined.iterrows():
            event = MarketEvent(
                ts=pd.Timestamp(row['timestamp']),
                symbol=row['symbol'],
                etype="bar",
                payload={
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume'],
                }
            )
            yield event
