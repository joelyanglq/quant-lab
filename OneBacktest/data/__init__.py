"""
data/ 模块

架构：
    DataFeed (抽象接口)
        └── HistoricFeed (从 Parquet 文件加载历史数据)
    
    DataHandler (中间层：Iterator + Query)
        - 内部从 DataFeed 拉取 Bar
        - 对外提供 get_latest_bars() 查询
        - 向事件队列发送 MarketEvent
    
    ParquetStorage (存储后端)
        - 读写 Parquet 文件

目录结构：
    data/
    ├── __init__.py
    ├── types.py           # Bar, Frequency, AdjustType
    ├── feed.py            # DataFeed 抽象基类
    ├── handler.py         # DataHandler 混合层
    ├── sources/
    │   └── historic.py    # HistoricFeed 实现
    └── storage/
        └── parquet.py     # ParquetStorage
"""
from .types import Bar, Frequency, AdjustType
from .feed import DataFeed
from .sources.historic import HistoricFeed
from .handler import DataHandler
from .storage.parquet import ParquetStorage

__all__ = [
    'Bar',
    'Frequency',
    'AdjustType',
    'DataFeed',
    'HistoricFeed',
    'DataHandler',
    'ParquetStorage',
]
