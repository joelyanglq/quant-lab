"""
DataHandler - 混合设计（Iterator + Query）

- 从 DataFeed（Iterator）拉取数据
- 内部缓存历史 bar（deque 限制长度）
- 提供 get_latest_bars() 给策略查询
- 生成 MarketEvent 推送到事件队列
"""
import queue
from collections import deque
from typing import List, Optional, Dict

import pandas as pd

from .feed import DataFeed
from .types import Bar
from event import MarketEvent


class DataHandler:
    """
    数据处理器（中间层）

    职责：
    1. 从 DataFeed 逐个拉取 Bar
    2. 缓存每个标的最近 N 个 bar（供策略查询指标）
    3. 向事件队列发送 MarketEvent

    使用方式：
        handler = DataHandler(data_feed, events, cache_size=200)
        while handler.update():
            pass  # 引擎会从事件队列消费 MarketEvent
    """

    def __init__(
        self,
        data_feed: DataFeed,
        events: queue.Queue,
        cache_size: int = 200,
    ):
        self.data_feed = data_feed
        self.events = events
        self.cache_size = cache_size

        # {symbol: deque([Bar, Bar, ...], maxlen=cache_size)}
        self._bars_cache: Dict[str, deque] = {}
        self._current_time: Optional[pd.Timestamp] = None
        self.continue_backtest = True

    def update(self) -> bool:
        """
        从 DataFeed 拉取下一个 Bar，缓存并生成 MarketEvent

        Returns:
            True  - 成功拉取到新数据
            False - 数据已耗尽
        """
        if not self.data_feed.has_next():
            self.continue_backtest = False
            return False

        bar = self.data_feed.next()
        if bar is None:
            self.continue_backtest = False
            return False

        # 缓存
        if bar.symbol not in self._bars_cache:
            self._bars_cache[bar.symbol] = deque(maxlen=self.cache_size)
        self._bars_cache[bar.symbol].append(bar)

        # 更新当前时间
        self._current_time = bar.timestamp

        # 发送 MarketEvent
        event = MarketEvent(timestamp=bar.timestamp, symbol=bar.symbol)
        self.events.put(event)

        return True

    # ==================== 查询接口（给策略用） ====================

    def get_latest_bars(self, symbol: str, N: int = 1) -> List[Bar]:
        """
        返回最近 N 个 Bar

        Args:
            symbol: 标的代码
            N: 需要的 bar 数量

        Returns:
            Bar 列表（按时间正序），不足 N 个则返回已有的全部
        """
        bars = self._bars_cache.get(symbol)
        if bars is None:
            return []
        return list(bars)[-N:]

    def get_latest_bar(self, symbol: str) -> Optional[Bar]:
        """返回最新的一个 Bar"""
        bars = self._bars_cache.get(symbol)
        if not bars:
            return None
        return bars[-1]

    def get_latest_bars_df(self, symbol: str, N: int = 1) -> pd.DataFrame:
        """
        返回最近 N 个 Bar 的 DataFrame（方便计算指标）

        Returns:
            DataFrame，列: open, high, low, close, volume
        """
        bars = self.get_latest_bars(symbol, N)
        if not bars:
            return pd.DataFrame()
        return pd.DataFrame([b.to_dict() for b in bars]).set_index('timestamp')

    def get_current_time(self) -> Optional[pd.Timestamp]:
        """获取当前时间"""
        return self._current_time

    @property
    def symbols(self) -> List[str]:
        """已缓存的标的列表"""
        return list(self._bars_cache.keys())
