"""
data/ 模块

架构：
    DataFeed (抽象接口)
        └── HistoricFeed (从 Parquet 文件加载历史数据)

    HistoryManager (数据层 handler)
        - Engine 每根 bar 调用 _on_bar() 累积 1d 滚动窗口
        - 策略通过 self.history 访问 panel / get / panel_1min 等

    ParquetStorage (存储后端)
        - 读写 Parquet 文件

目录结构：
    data/
    ├── __init__.py
    ├── types.py           # Bar, Frequency, AdjustType
    ├── feed.py            # DataFeed 抽象基类
    ├── history.py         # HistoryManager (1d 缓存 + 1min 按需加载)
    ├── fundamentals.py    # Point-in-Time 基本面加载
    ├── prices.py          # 价格面板加载
    ├── sources/
    │   └── historic.py    # HistoricFeed 实现
    └── storage/
        └── parquet.py     # ParquetStorage
"""
from .types import Bar, Frequency, AdjustType
from .feed import DataFeed
from .sources.historic import HistoricFeed
from .storage.parquet import ParquetStorage
from .history import HistoryManager
from .fundamentals import build_fundamental_panel, build_shares_panel, build_quarterly_series
from .prices import load_price_panel, load_index_symbols

__all__ = [
    'Bar',
    'Frequency',
    'AdjustType',
    'DataFeed',
    'HistoricFeed',
    'ParquetStorage',
    'HistoryManager',
    'build_fundamental_panel',
    'build_shares_panel',
    'build_quarterly_series',
    'load_price_panel',
    'load_index_symbols',
]
