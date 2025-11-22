from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime

from .base import DataFeed, Frequency, AdjustType
from .parquet_storage import ParquetStorage

class DataManager:
    """
    数据管理器
    
    统一管理数据的获取、缓存和更新
    这是业务层应该使用的主要接口
    """
    
    def __init__(
        self,
        data_feed: DataFeed,
        storage: Optional[ParquetStorage] = None,
        use_cache: bool = True
    ):
        self.data_feed = data_feed
        self.storage = storage
        self.use_cache = use_cache
        
        # 内存缓存
        self._memory_cache: Dict[str, pd.DataFrame] = {}
    
    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        frequency: Frequency = Frequency.DAY,
        adjust: AdjustType = AdjustType.FORWARD,
        force_update: bool = False
    ) -> pd.DataFrame:
        """
        获取K线数据（智能缓存）
        
        查找顺序：内存缓存 -> 本地存储 -> 数据源
        """
        
        cache_key = f"{symbol}_{frequency.value}_{adjust.value}"
        
        # 1. 检查内存缓存
        if not force_update and cache_key in self._memory_cache:
            df = self._memory_cache[cache_key]
            # 检查日期范围是否满足
            if df.index.min() <= start and df.index.max() >= end:
                return df.loc[start:end]
        
        # 2. 检查本地存储
        if self.use_cache and self.storage and not force_update:
            try:
                if self.storage.exists(symbol, frequency.value):
                    df = self.storage.load(symbol, frequency=frequency.value)
                    
                    # 检查是否需要更新
                    if df.index.max() >= end:
                        self._memory_cache[cache_key] = df
                        return df.loc[start:end]
            except Exception as e:
                print(f"Error loading from storage: {e}")
        
        # 3. 从数据源获取
        df = self.data_feed.get_bars(symbol, start, end, frequency, adjust)
        
        if df.empty:
            raise ValueError(f"No data available for {symbol}")
        
        # 4. 保存到存储和缓存
        if self.storage:
            try:
                self.storage.save(symbol, df, frequency.value)
            except Exception as e:
                print(f"Error saving to storage: {e}")
        
        self._memory_cache[cache_key] = df
        
        return df
    
    def update_data(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        frequency: Frequency = Frequency.DAY
    ):
        """
        批量更新数据
        
        适合每日定时任务
        """
        for symbol in symbols:
            try:
                print(f"Updating {symbol}...")
                df = self.data_feed.get_bars(symbol, start, end, frequency)
                if not df.empty and self.storage:
                    self.storage.save(symbol, df, frequency.value)
            except Exception as e:
                print(f"Error updating {symbol}: {e}")
    
    def preload(self, symbols: List[str], start: datetime, end: datetime):
        """预加载数据到内存"""
        for symbol in symbols:
            try:
                self.get_bars(symbol, start, end)
            except Exception as e:
                print(f"Error preloading {symbol}: {e}")