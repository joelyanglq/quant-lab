import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import List, Optional

from .base import DataFeed, Frequency, AdjustType

class YFinanceDataFeed(DataFeed):
    """
    DataFeed implementation using yfinance.
    """
    def __init__(self):
        super().__init__("yfinance")

    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        frequency: Frequency = Frequency.DAY,
        adjust: AdjustType = AdjustType.FORWARD
    ) -> pd.DataFrame:
        """
        Fetch historical data from Yahoo Finance.
        """
        # Map Frequency to yfinance interval
        interval_map = {
            Frequency.MIN_1: '1m',
            Frequency.MIN_5: '5m',
            Frequency.MIN_15: '15m',
            Frequency.MIN_30: '30m',
            Frequency.HOUR_1: '1h',
            Frequency.DAY: '1d',
            Frequency.WEEK: '1wk',
            Frequency.MONTH: '1mo'
        }
        
        interval = interval_map.get(frequency, '1d')
        
        # yfinance expects string dates or datetime objects
        # auto_adjust=True does Close adjustment. 
        # For full adjustment (splits/dividends) we might need actions=True and calculate manually if needed,
        # but yfinance's auto_adjust is usually sufficient for backtesting on adjusted close.
        # However, yfinance returns 'Close' as adjusted close if auto_adjust=True.
        
        df = yf.download(
            symbol,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=(adjust != AdjustType.NONE),
            progress=False
        )
        
        if df.empty:
            return pd.DataFrame()

        # Standardize columns
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Ensure index is datetime
        df.index = pd.to_datetime(df.index)
        
        return df[['open', 'high', 'low', 'close', 'volume']]

    def get_symbols(self, market: Optional[str] = None) -> List[str]:
        """
        YFinance doesn't provide a list of all symbols.
        Return an empty list or raise NotImplementedError.
        """
        return []
