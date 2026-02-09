"""
历史数据源

从 ParquetStorage 加载历史数据
使用 heapq 优先级队列自动合并多标的时间线
"""
import heapq
from typing import List, Optional

import pandas as pd

from ..feed import DataFeed
from ..types import Bar
from ..storage.parquet import ParquetStorage


class HistoricFeed(DataFeed):
    """
    历史数据源（回测用）

    从 ParquetStorage 加载数据，用 heapq 按时间戳排序
    多标的数据自动合并到同一条时间线

    Parquet 文件格式要求：
    - 文件名: {symbol}_{frequency}.parquet（由 ParquetStorage 管理）
    - index: DatetimeIndex（时间戳）
    - 列: open, high, low, close, volume（小写）

    使用方式：
        storage = ParquetStorage('./data_cache')
        feed = HistoricFeed(storage, frequency='1d')
        feed.subscribe(['AAPL', 'SPY'], start, end)
        while feed.has_next():
            bar = feed.next()
    """

    def __init__(self, storage: ParquetStorage, frequency: str = '1d'):
        """
        Args:
            storage: ParquetStorage 实例
            frequency: 数据频率（用于匹配文件名）
        """
        self.storage = storage
        self.frequency = frequency
        self._heap: list = []       # heapq: [(timestamp, counter, Bar)]
        self._counter = 0           # 保证同一时间戳的稳定排序
        self._current_time: Optional[pd.Timestamp] = None
        self._symbols: List[str] = []

    def subscribe(self, symbols: List[str], start: pd.Timestamp, end: pd.Timestamp):
        """从 Parquet 文件加载数据并放入优先级队列"""
        self._heap = []
        self._counter = 0
        self._symbols = list(symbols)

        for symbol in symbols:
            # 从 ParquetStorage 加载
            df = self.storage.load(symbol, start, end, self.frequency)

            if df.empty:
                print(f"Warning: No data for {symbol} in [{start}, {end}]")
                continue

            # 确保列名小写
            df.columns = df.columns.str.lower()

            # 验证必需列
            required = ['open', 'high', 'low', 'close', 'volume']
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"Parquet for {symbol} missing columns: {missing}")

            # 转换为 Bar 并放入 heapq
            for ts, row in df.iterrows():
                bar = Bar(
                    timestamp=pd.Timestamp(ts),
                    symbol=symbol,
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=int(row['volume']),
                )
                heapq.heappush(self._heap, (bar.timestamp, self._counter, bar))
                self._counter += 1

        print(f"HistoricFeed: loaded {self._counter} bars for {symbols}")

    def next(self) -> Optional[Bar]:
        """返回时间戳最早的下一个 Bar"""
        if not self._heap:
            return None
        ts, _, bar = heapq.heappop(self._heap)
        self._current_time = ts
        return bar

    def has_next(self) -> bool:
        return len(self._heap) > 0

    def reset(self):
        """重置（需要重新 subscribe）"""
        self._heap = []
        self._counter = 0
        self._current_time = None

    def get_current_time(self) -> Optional[pd.Timestamp]:
        return self._current_time
